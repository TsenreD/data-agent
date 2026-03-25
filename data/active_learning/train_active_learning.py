#!/usr/bin/env python3
"""
Active Learning Training Script for Text Classification
Task: Determine whether text contains a full problem (complete) or an incomplete one.
Label 1 = complete problem, Label 0 = incomplete problem
Model: DeepPavlov/rubert-base-cased with classification head only
"""

from pathlib import Path
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import f1_score

# Set random seeds for determinism
SEED = 13
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def load_jsonl(path):
    """Load JSONL file."""
    lines = path.read_text().strip().split('\n')
    return [json.loads(line) for line in lines]


class TextDataset(Dataset):
    """PyTorch Dataset for text classification."""
    
    def __init__(self, data, feature_col, target_col):
        self.texts = [d[feature_col] for d in data]
        self.labels = [int(d[target_col]) for d in data]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class TextClassifier(nn.Module):
    """Text classifier with frozen encoder and trainable classification head."""
    
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        # Freeze encoder parameters - only train classification head
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        # Encoder is frozen - no grad tracking
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits


def train():
    """Main training function."""
    # Read config
    config_path = Path('/home/user/data-agent/data/active_learning/training_job_config.json')
    config = json.loads(config_path.read_text())
    
    train_path = Path(config['train_path'])
    val_path = Path(config['val_path'])
    model_path = Path(config['model_path'])
    metrics_path = Path(config['metrics_path'])
    target_column = config['target_column']
    feature_columns = config['feature_columns']
    
    # Load data
    train_data = load_jsonl(train_path)
    val_data = load_jsonl(val_path)
    
    print(f"Train rows: {len(train_data)}")
    print(f"Val rows: {len(val_data)}")
    
    # Load tokenizer and model
    model_name = "DeepPavlov/rubert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = TextClassifier(model_name, num_labels=2)
    model.to(device)
    
    # Create datasets
    train_dataset = TextDataset(train_data, feature_columns[0], target_column)
    val_dataset = TextDataset(val_data, feature_columns[0], target_column)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.classifier.parameters(), lr=2e-5)
    
    # Training loop
    num_epochs = 5
    loss_history = []
    
    # Disable gradient computation globally for inference
    torch.set_grad_enabled(False)
    
    for epoch in range(num_epochs):
        # Training - enable gradients
        torch.set_grad_enabled(True)
        model.train()
        train_loss = 0.0
        for texts, labels in train_loader:
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = torch.tensor(labels).to(device)
            
            optimizer.zero_grad()
            logits = model(**inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation - disable gradients
        torch.set_grad_enabled(False)
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        for texts, labels in val_loader:
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels_tensor = torch.tensor(labels).to(device)
            
            logits = model(**inputs)
            loss = criterion(logits, labels_tensor)
            val_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels_tensor).sum().item()
            total += labels_tensor.size(0)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels_tensor.cpu().numpy().tolist())
        
        val_loss /= len(val_loader)
        accuracy = correct / total
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        
        loss_history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss)
        })
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.4f}, Macro F1: {macro_f1:.4f}")
    
    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Calculate final metrics
    model.eval()
    all_preds = []
    all_labels = []
    
    for texts, labels in val_loader:
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels)
    
    final_accuracy = sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
    final_macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    # Save metrics
    metrics = {
        'accuracy': float(final_accuracy),
        'macro_f1': float(final_macro_f1),
        'evaluated_rows': int(len(val_data)),
        'train_rows': int(len(train_data)),
        'val_rows': int(len(val_data)),
        'backend': 'pytorch',
        'loss_history': loss_history
    }
    
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {metrics_path}")
    print(f"Final metrics: {metrics}")


if __name__ == '__main__':
    train()
