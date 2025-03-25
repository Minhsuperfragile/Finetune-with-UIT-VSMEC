#@title First test mTGE family model without semantics understanding
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
import numpy as np
import evaluate

output_dir = "./bkai-embedding-ft"

# 1. Load tokenizer and classification head model
model_name = "bkai-foundation-models/vietnamese-bi-encoder"
num_labels = 7  # binary classification
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels,trust_remote_code=True)

# 2. Load a classification dataset (IMDB as example)
dataset = load_dataset("tridm/UIT-VSMEC")

# 2. Fixed emotion-to-id mapping
label_list = ['Anger', 'Disgust', 'Enjoyment', 'Fear', 'Other', 'Sadness', 'Surprise']
label2id = {
    'Anger': 0,
    'Disgust': 1,
    'Enjoyment': 2,
    'Fear': 3,
    'Other': 4,
    'Sadness': 5,
    'Surprise': 6
}
id2label = {v: k for k, v in label2id.items()}
label2id = {label: idx for idx, label in enumerate(sorted(label_list))}
print("Label mapping:", label2id)

# 2. Map Emotion strings to integers
def encode_labels(example):
    example["labels"] = label2id[example["Emotion"]]
    return example

dataset = dataset.map(encode_labels)
# 3. Preprocess (tokenize text)
def preprocess_function(examples):
    return tokenizer(examples["Sentence"], truncation=True, padding="max_length", max_length=258)

tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]
train_subset = train_dataset.shuffle(seed=42).select(range(500))
test_subset = test_dataset.shuffle(seed=42).select(range(200))
# 5. Load metric
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

# 6. TrainingArguments
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_steps=100,
    push_to_hub=False,
    report_to="none",
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset, # use subset for speed
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 8. Train
trainer.train()

# 9. Evaluate
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# 10. Save the final model
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
