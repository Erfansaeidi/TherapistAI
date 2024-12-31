from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, 
                          DataCollatorWithPadding)
import torch
import pandas as pd

# Load the dataset
df = pd.read_csv('dataset/test.csv')

# Load dataset
dataset = load_dataset("imdb")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Ensure the tokenizer has a pad_token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to the selected device

# Tokenize dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True, max_length=512)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Prepare datasets for training
tokenized_datasets = tokenized_datasets.remove_columns(["text"])  # Keep only input IDs and labels
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

# Ensure labels are tensors
def convert_labels_to_tensor(batch):
    batch["label"] = torch.tensor(batch["label"], dtype=torch.long)
    return batch

train_dataset = train_dataset.map(convert_labels_to_tensor)
eval_dataset = eval_dataset.map(convert_labels_to_tensor)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    report_to="none",  # Avoid unnecessary logging to external services
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train model
if __name__ == "__main__":
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving the model...")
        trainer.save_model("./saved_model_interrupt")
        print("Model saved.")

# Save model after training
trainer.save_model("./saved_model")
