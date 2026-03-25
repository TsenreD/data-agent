#!/usr/bin/env python3
"""
Active Learning Training Script
Train classification model to determine whether text contains a full problem, or an incomplete one.
When annotation label is 1, problem is considered complete, with label 0 marking incomplete problems.
Uses DeepPavlov/rubert-base-cased with a classification head on top.
"""

from pathlib import Path
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F

# Set deterministic seed
SEED = 13
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load config
config_path = Path('/home/user/data-agent/data/active_learning/training_job_config.json')
config = json.loads(config_path.read_text())

# Paths
train_path = Path(config['train_path'])
val_path = Path(config['val_path'])
model_path = Path(config['model_path'])
metrics_path = Path(config['metrics_path'])

# Hyperparameters
EPOCHS = 5
BATCH_SIZE = 20
LEARNING_RATE = 2e-4
MAX_LEN = 256

print("Config loaded:")
print(f"  train_path: {train_path}")
print(f"  val_path: {val_path}")
print(f"  model_path: {model_path}")
print(f"  metrics_path: {metrics_path}")
print(f"  epochs: {EPOCHS}, batch_size: {BATCH_SIZE}")

# Load data from JSONL files
def load_jsonl(path):
    lines = path.read_text().strip().split('\n')
    data = []
    for line in lines:
        if line:
            data.append(json.loads(line))
    return data

train_data = load_jsonl(train_path)
val_data = load_jsonl(val_path)

print(f"Train samples: {len(train_data)}")
print(f"Val samples: {len(val_data)}")

# Create dataset class
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text'][:5000]  # Truncate very long texts
        label = int(item['annotation_label'])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Load tokenizer and model
model_name = "DeepPavlov/rubert-base-cased"
print(f"Loading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Loading model: {model_name}")
base_model = AutoModel.from_pretrained(model_name)

# Freeze all model parameters
for param in base_model.parameters():
    param.requires_grad = False

# Get hidden size
hidden_size = base_model.config.hidden_size
print(f"Hidden size: {hidden_size}")

# Create datasets and dataloaders
train_dataset = TextDataset(train_data, tokenizer, MAX_LEN)
val_dataset = TextDataset(val_data, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# Setup training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

base_model = base_model.to(device)

# Use class weights for imbalanced data
class_counts = [98, 2]  # positive, negative
total = sum(class_counts)
class_weights = torch.tensor([total / (2 * c) for c in class_counts], dtype=torch.float32).to(device)
print(f"Class weights: {class_weights}")

# Create classifier
classifier_layer = torch.nn.Linear(hidden_size, 2).to(device)
optimizer = torch.optim.Adam(classifier_layer.parameters(), lr=LEARNING_RATE)

# Set models to eval mode
base_model.eval()
classifier_layer.train()

loss_history = []
best_val_loss = float('inf')

print("Starting training...")

for epoch in range(EPOCHS):
    train_loss_total = 0
    num_batches = 0
    
    # Training loop - use iterator
    train_iter = iter(train_loader)
    while True:
        try:
            batch = next(train_iter)
        except StopIteration:
            break
            
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Get BERT outputs
        outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Classifier
        logits = classifier_layer(cls_output)
        
        # Loss
        loss = F.cross_entropy(logits, labels, weight=class_weights)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        train_loss_total += loss.item()
        num_batches += 1
    
    train_loss = train_loss_total / num_batches
    
    # Validation
    val_loss_total = 0
    all_preds = []
    all_labels = []
    
    val_iter = iter(val_loader)
    while True:
        try:
            batch = next(val_iter)
        except StopIteration:
            break
            
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = classifier_layer(cls_output)
        
        loss = F.cross_entropy(logits, labels, weight=class_weights)
        val_loss_total += loss.item()
        
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    val_loss = val_loss_total / len(val_loader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    
    loss_history.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'val_loss': val_loss
    })
    
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro F1: {val_f1:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(classifier_layer.state_dict(), str(model_path))
        print(f"  -> Model saved")

print("Training complete!")

# Final evaluation on validation set
base_model.eval()
classifier_layer.load_state_dict(torch.load(str(model_path)))

all_preds = []
all_labels = []

val_iter = iter(val_loader)
while True:
    try:
        batch = next(val_iter)
    except StopIteration:
        break
        
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)
    
    outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
    cls_output = outputs.last_hidden_state[:, 0, :]
    logits = classifier_layer(cls_output)
    
    preds = torch.argmax(logits, dim=1)
    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

final_accuracy = accuracy_score(all_labels, all_preds)
final_f1 = f1_score(all_labels, all_preds, average='macro')

print(f"Final metrics:")
print(f"  Accuracy: {final_accuracy}")
print(f"  Macro F1: {final_f1}")

# Create metrics JSON
metrics = {
    "accuracy": final_accuracy,
    "macro_f1": final_f1,
    "evaluated_rows": len(val_data),
    "train_rows": len(train_data),
    "val_rows": len(val_data),
    "backend": "DeepPavlov/rubert-base-cased",
    "loss_history": loss_history
}

# Save metrics
metrics_path.write_text(json.dumps(metrics, indent=2))
print(f"\nMetrics saved to: {metrics_path}")

# Verify files exist
print(f"\nVerifying output files:")
print(f"  Model path exists: {model_path.exists()}")
print(f"  Metrics path exists: {metrics_path.exists()}")

if model_path.exists():
    print(f"  Model file size: {model_path.stat().st_size} bytes")

print("\nDone!")
