from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from preprocess import load_and_preprocess_data

# Load the trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("../models/saved_model")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load and preprocess the dataset
dataset = load_and_preprocess_data("../dataset/test.csv")

# Tokenize dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True, max_length=512)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Evaluate the model
def evaluate_model(model, dataset):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for example in dataset:
            inputs = tokenizer(example["text"], return_tensors="pt", padding=True, truncation=True, max_length=512)
            labels = torch.tensor(example["label"]).unsqueeze(0)
            outputs = model(**inputs)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

if __name__ == "__main__":
    accuracy = evaluate_model(model, tokenized_dataset)
    print(f"Model accuracy: {accuracy}")
