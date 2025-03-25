from collections import defaultdict
from datetime import datetime
import json , evaluate, argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default='vbe', help="One of 3 predefined model paths, or your local checkpoint paths of 1 of 3 models")
parser.add_argument("--task", type=str, default='ftstd', help="One of 4 tasks, finetune or evaluate, similarity or standard")
parser.add_argument("--config", type=str, default=None, help="Path to config file, if not provided, decide by model name")
parser.add_argument("--save-loss", type=int, default=1, help="Save running loss to a json file or not, use with finetune mode only")
parser.add_argument("--save-folder", type=str, default=None, help="Save finetuned model to a folder, if not provided, saved as model-name-task-time")
parser.add_argument("--confusion-matrix", type=int, default=1, help="Save confusion matrix to an image file or not, use with evaluation mode only")

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
    with open(f"./config/{model_name.split("/")[-1]}.json") as f:
        all_config = json.load(f) 

task_set = defaultdict(lambda: None, 
                       {
                           "ftsim": 0,
                           "ftstd": 1,
                           "evsim": 2,
                           "evstd": 3
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
        self.traning_args = TrainingArguments(output_dir = self.output_dir, **all_config['standard']['train'])
        self.trainer = Trainer(
            model = self.model,
            args = self.traning_args,
            train_dataset = self.trainset,
            tokenizer = self.tokenizer
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

class StandardEvaluateEmbeddingModelRunner:
    def __init__(self):
        pass        

class FinetuneLLM:
    def __init__(self):
        pass

class EvaluateLLM:
    def __init__(self):
        pass

if __name__ == "__main__":
    if task == 1:
        runner = StandardFinetuneEmbeddingModelRunner()
        runner.train()