from collections import defaultdict
from datetime import datetime
import json , unsloth , argparse, os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from datasets import load_dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default='vbe', help="One of 3 predefined model paths, or your local checkpoint paths of 1 of 3 models")
parser.add_argument("--task", type=str, default='ftstd', help="One of 4 tasks, finetune or evaluate, similarity or standard")
parser.add_argument("--config", type=str, default=None, help="Path to config file, if not provided, decide by model name")
parser.add_argument("--save-folder", type=str, default=None, help="Save finetuned model to a folder, if not provided, saved as model-name-task-time")

args = parser.parse_args()

model_set = defaultdict(lambda: None,
                        {
                            'vbe' : "bkai-foundation-models/vietnamese-bi-encoder",
                            'gte' : "Alibaba-NLP/gte-multilingual-mlm-base",
                            'qw7' : "Qwen/Qwen2.5-7B"
                        })

model_name = model_set[args.model]
if model_name is None: 
    assert args.config is not None, "Config must be provided for checkpoint model"
    with open(args.config, 'r') as f:
        all_config = json.load(f) 
    model_name = args.model 
else:
    with open(f"./config/{model_name.split("/")[-1]}.json", 'r') as f:
        all_config = json.load(f) 

task_set = defaultdict(lambda: None, 
                       {
                           "ftsim": 0,
                           "ftstd": 1,
                           "evsim": 2,
                           "evstd": 3,
                           "ftllm": 4,
                           "evllm": 5
                       })
task = task_set[args.task]
assert task is not None, "task must be 1 of 4 predefined tasks"

def format_save_folder_name(model_name):
    if "/" in model_name:
        name_ = model_name.split("/")[-1]
    elif "\\" in model_name:
        name_ = model_name.split("\\")[-1]
    else:
        name_ = model_name
    
    now = datetime.now()
    # Extract date, month, hour, and minute
    current_date = now.day
    current_month = now.month
    current_hour = now.hour
    current_minute = now.minute

    return f'{name_}-{current_date}-{current_month}-{current_hour}-{current_minute}'

class StandardFinetuneEmbeddingModelRunner:
    def __init__(self):
        self.dataset = load_dataset("tridm/UIT-VSMEC")
        self.num_labels = 7 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=self.num_labels,trust_remote_code=True)
        if args.save_folder is not None:
            self.output_dir = args.save_folder
        else: self.output_dir = format_save_folder_name(model_name)
        self.label_list = ['Anger', 'Disgust', 'Enjoyment', 'Fear', 'Other', 'Sadness', 'Surprise']
        self.label2id = {
            'Anger': 0,
            'Disgust': 1,
            'Enjoyment': 2,
            'Fear': 3,
            'Other': 4,
            'Sadness': 5,
            'Surprise': 6
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.label2id = {label: idx for idx, label in enumerate(sorted(self.label_list))}
        
        self.dataset = self.dataset.map(self.encode_labels)
        self.preprocess_config = all_config['standard']['tokenizer'] 
        self.dataset = self.dataset.map(self.preprocess_function, batched=True)
        self.trainset = self.dataset["train"]
        self.training_args = TrainingArguments(output_dir = self.output_dir, **all_config['standard']['train'])
        self.trainer = Trainer(
            model = self.model,
            args = self.training_args,
            train_dataset = self.trainset,
            processing_class = self.tokenizer
        )

    def encode_labels(self,example):
        example["labels"] = self.label2id[example["Emotion"]]
        return example
    
    def preprocess_function(self,examples):
        return self.tokenizer(examples["Sentence"], **self.preprocess_config)
    
    def train(self):
        self.trainer.train()
        self.trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

class StandardEvaluateEmbeddingModelRunner(StandardFinetuneEmbeddingModelRunner):
    def __init__(self):
        super().__init__()
        self.testset = self.dataset['test']
        self.training_args = TrainingArguments(output_dir=self.output_dir, **all_config['standard']['evaluate'])
        self.trainer = Trainer(
            model = self.model,
            args = self.training_args,
            eval_dataset=self.testset,
            processing_class = self.tokenizer,
            compute_metrics = self.compute_metric
        )
        pass        

    def compute_metric(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
    
        # Compute precision, recall, f1-score
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

        # Compute confusion matrix
        conf_matrix = confusion_matrix(labels, preds)

        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=self.label_list, yticklabels=self.label_list)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")

        # Save the image
        image_path = f"{self.output_dir}/confusion_matrix.png"
        plt.savefig(image_path)
        plt.close()

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    
    def evaluate(self):
        result = self.trainer.evaluate()
        print(result)

class ContrastiveDataset(Dataset):
    def __init__(self, data, tokenizer, label_list, max_length=258):
        self.data = data
        self.tokenizer = tokenizer
        self.labels = label_list
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]["Sentence"]
        emotion = self.data[idx]["Emotion"]

        # Positive: (sentence, correct label)
        input_enc = self.tokenizer(sentence, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        label_enc = self.tokenizer(emotion, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")

        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "label_ids": label_enc["input_ids"].squeeze(0),
            "label_attention_mask": label_enc["attention_mask"].squeeze(0),
            "emotion": emotion
        }

class ConstrastiveFinetuneEmbeddingModelRunner:
    def __init__(self):
        if args.save_folder is not None:
            self.output_dir = args.save_folder
        else: self.output_dir = format_save_folder_name(model_name)

        self.config = all_config['similarity']['train']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code = True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.label_list_vi = ['Tức giận', 'Kinh tởm', 'Thích thú', 'Sợ hãi', 'Khác', 'Buồn bã', 'Bất ngờ']
        self.label_list_en = ['Anger', 'Disgust', 'Enjoyment', 'Fear', 'Other', 'Sadness', 'Surprise']
        self.label_map = {k: v for k,v in zip(self.label_list_vi, self.label_list_en)}
        self.label2id = {
            'Anger': 0,
            'Disgust': 1,
            'Enjoyment': 2,
            'Fear': 3,
            'Other': 4,
            'Sadness': 5,
            'Surprise': 6
        }

        self.dataset = load_dataset("tridm/UIT-VSMEC")
        self.trainset = ContrastiveDataset(self.dataset['train'].select(range(10)), self.tokenizer, self.label_list_vi)
        self.trainloader = DataLoader(self.trainset, batch_size=self.config['batch_size'], shuffle=self.config['shuffle'])
        self.optimizer = AdamW(self.model.parameters(), lr = self.config['learning_rate'])
        pass
    
    @staticmethod
    def cosine_similarity(a,b):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return torch.matmul(a, b.T)
    @staticmethod
    def contrastive_loss(sent_embs, label_embs):
        sim = ConstrastiveFinetuneEmbeddingModelRunner.cosine_similarity(sent_embs, label_embs)  # [B, B]
        labels = torch.arange(sim.size(0)).to(sim.device)
        return F.cross_entropy(sim, labels)
    
    def train(self):
        self.model.to(self.device)
        self.model.train()
        for epoch in range(self.config['epochs']):
            total_loss = 0.0
            pbar = tqdm(self.trainloader, desc=f"Epoch {epoch+1}")

            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                label_ids = batch["label_ids"].to(self.device)
                label_attention_mask = batch["label_attention_mask"].to(self.device)

                # Get embeddings ([CLS] token)
                sent_out = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
                label_out = self.model(input_ids=label_ids, attention_mask=label_attention_mask).last_hidden_state[:, 0]

                loss = ConstrastiveFinetuneEmbeddingModelRunner.contrastive_loss(sent_out, label_out)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

            print(f"Epoch {epoch+1} - Avg loss: {total_loss / len(self.trainset):.4f}")
            
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

class ConstrastiveEvaluateEmbeddingModelRunner(ConstrastiveFinetuneEmbeddingModelRunner):
    def __init__(self):
        super().__init__()
        self.config = all_config['similarity']['evaluate']
        self.testset = ContrastiveDataset(self.dataset['test'], self.tokenizer, self.label_list_vi)
        self.testloader = DataLoader(self.testset, batch_size=self.config['batch_size'])

    def evaluate(self):
        self.model.eval()
        self.model.to(self.device)

        preds = []
        labels = []

        with torch.no_grad():
            label_embs = []
            for label in self.label_list_vi:
                tokens = self.tokenizer(label, return_tensors="pt", truncation=True, padding=True).to(self.device)
                label_emb = self.model(**tokens).last_hidden_state[:, 0]
                label_embs.append(F.normalize(label_emb, dim=-1))
            label_embs = torch.cat(label_embs, dim=0)  # [num_labels, hidden]

        for batch in tqdm(self.testloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            true_emotion = batch["emotion"]

            with torch.no_grad():
                sentence_emb = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
                sentence_emb = F.normalize(sentence_emb, dim=-1)
                sims = ConstrastiveFinetuneEmbeddingModelRunner.cosine_similarity(sentence_emb, label_embs)  # [1, num_labels]
                pred_idx = torch.argmax(sims, dim=1).item()
                pred_label = self.label_list_en[pred_idx]

            preds.append(pred_label)
            labels.append(true_emotion[0])

        preds = [self.label2id[pred] for pred in preds]
        labels = [self.label2id[label] for label in labels]

        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        # precision, recall, f1 = 0,0,0
        # print("before plt")
        # cm = confusion_matrix(labels, preds, labels=[self.label2id[label] for label in self.label_list_en])

        # plt.figure(figsize=(6, 5))
        # print("before sns")
        # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.label_list, yticklabels=self.label_list)
        # plt.xlabel("Predicted Label")
        # plt.ylabel("True Label")
        # plt.title("Confusion Matrix")

        # # Save the image
        # image_path = f"{self.output_dir}/confusion_matrix.png"
        # plt.savefig(image_path)
        # plt.close()
        print(f"\nTop-1 Accuracy: {accuracy:.4f}, {precision:.4f}, {recall:.4f}, {f1:.4f}")

class FinetuneLLM:
    def __init__(self):
        pass

class EvaluateLLM:
    def __init__(self):
        pass

if __name__ == "__main__":
    if task == 1: # finetune standard
        runner = StandardFinetuneEmbeddingModelRunner()
        runner.train()
    elif task == 3: # evaluate standard
        runner = StandardEvaluateEmbeddingModelRunner()
        runner.evaluate()
    elif task == 0: # finetune constrastive
        runner = ConstrastiveFinetuneEmbeddingModelRunner()
        runner.train()
    elif task == 2: # evaluate constrastive
        runner = ConstrastiveEvaluateEmbeddingModelRunner()
        runner.evaluate()