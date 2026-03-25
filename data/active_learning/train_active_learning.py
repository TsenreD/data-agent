#!/usr/bin/env python3
"""
Active Learning Classifier Training Script
Trains a fasttext-based classifier to determine whether text contains a full problem or an incomplete one.
"""
import json
import pathlib
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# Set random seeds for determinism
SEED = 13
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def load_jsonl(path):
    """Load JSONL file."""
    content = pathlib.Path(path).read_text()
    lines = content.strip().split('\n')
    data = []
    for line in lines:
        if line.strip():
            data.append(json.loads(line))
    return data


def get_label(label):
    """Convert label to binary: 1 if incomplete (none/invalid), 0 otherwise."""
    if label is None:
        return 1
    if isinstance(label, str):
        lower_label = label.lower()
        if lower_label == 'none' or lower_label == 'invalid':
            return 1
    return 0


def tokenize(text):
    """Simple character-level tokenization for fasttext-style embeddings."""
    text = text.lower()
    tokens = []
    text_padded = '<' + text + '>'
    for n in range(3, 7):
        for i in range(len(text_padded) - n + 1):
            tokens.append(text_padded[i:i+n])
    return tokens


class TextDataset(Dataset):
    """Custom Dataset for text classification."""
    
    def __init__(self, texts, labels, vocab, max_len=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        indices = self.text_to_indices(text)
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    
    def text_to_indices(self, text):
        tokens = tokenize(text)
        indices = [self.vocab.get(t, self.vocab['<unk>']) for t in tokens]
        # Pad or truncate
        if len(indices) < self.max_len:
            indices = indices + [self.vocab['<pad>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        return indices


class FastTextClassifier(nn.Module):
    """FastText-style classifier with classification head."""
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=64, num_classes=2):
        super(FastTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        # Mask padding
        mask = (x != 0).float().unsqueeze(-1)  # (batch, seq_len, 1)
        embedded = embedded * mask
        # Average pooling
        summed = embedded.sum(dim=1)  # (batch, embed_dim)
        lengths = mask.sum(dim=1).clamp(min=1)  # (batch, 1)
        pooled = summed / lengths  # (batch, embed_dim)
        
        # Classification head
        x = torch.relu(self.fc1(pooled))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_epoch(model, loader, criterion, optimizer):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (texts, labels) in enumerate(loader):
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for texts, labels in loader:
        outputs = model(texts)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(loader), correct / total, all_preds, all_labels


def main():
    # Read config
    config_path = pathlib.Path('/home/user/data-agent/data/active_learning/training_job_config.json')
    config_content = config_path.read_text()
    config = json.loads(config_content)
    
    print("Config loaded:")
    print(f"  train_path: {config['train_path']}")
    print(f"  val_path: {config['val_path']}")
    print(f"  task_prompt: {config['task_prompt'][:100]}...")
    print(f"  model_path: {config['model_path']}")
    print(f"  metrics_path: {config['metrics_path']}")
    print(f"  random_seed: {config['random_seed']}")
    
    # Load data
    train_data = load_jsonl(config['train_path'])
    val_data = load_jsonl(config['val_path'])
    
    print(f"\nLoaded {len(train_data)} training samples, {len(val_data)} validation samples")
    
    # Process data
    train_texts = []
    train_labels = []
    for item in train_data:
        text = item.get('text', '')
        label = item.get('label', None)
        binary_label = get_label(label)
        train_texts.append(text)
        train_labels.append(binary_label)
    
    val_texts = []
    val_labels = []
    for item in val_data:
        text = item.get('text', '')
        label = item.get('label', None)
        binary_label = get_label(label)
        val_texts.append(text)
        val_labels.append(binary_label)
    
    print(f"Training label distribution: 0={train_labels.count(0)}, 1={train_labels.count(1)}")
    print(f"Validation label distribution: 0={val_labels.count(0)}, 1={val_labels.count(1)}")
    
    # Task prompt for context
    task_prompt = config['task_prompt']
    print(f"\nTask prompt: {task_prompt[:200]}...")
    
    # Build vocabulary from training data
    word_counts = Counter()
    for text in train_texts:
        tokens = tokenize(text)
        word_counts.update(tokens)
    
    # Keep most common words (vocab size based on small dataset)
    vocab_size = min(5000, len(word_counts))
    most_common = word_counts.most_common(vocab_size)
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
    vocab['<pad>'] = 0
    vocab['<unk>'] = 1
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, vocab)
    val_dataset = TextDataset(val_texts, val_labels, vocab)
    
    # Create data loaders
    batch_size = 20
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    # Initialize model - small size for small dataset
    model = FastTextClassifier(
        vocab_size=len(vocab),
        embedding_dim=100,
        hidden_dim=64,
        num_classes=2
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 5
    best_val_acc = 0
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch': []
    }
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_preds, val_labels_list = evaluate(model, val_loader, criterion)
        
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model
            torch.save(model.state_dict(), config['model_path'])
            print(f"  -> Saved best model with val_acc: {val_acc:.4f}")
    
    # Final evaluation
    final_val_loss, final_val_acc, final_preds, final_labels = evaluate(model, val_loader, criterion)
    print(f"\nFinal validation accuracy: {final_val_acc:.4f}")
    
    # Save metrics
    metrics['final_val_acc'] = final_val_acc
    metrics['final_val_loss'] = final_val_loss
    metrics['best_val_acc'] = best_val_acc
    
    # Write metrics to JSON
    metrics_path = pathlib.Path(config['metrics_path'])
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {config['metrics_path']}")
    
    print("\nTraining complete!")
    
    return {
        'success': True,
        'model_path': config['model_path'],
        'metrics_path': config['metrics_path'],
        'notes': f'Trained fasttext classifier for {num_epochs} epochs with batch size {batch_size}. Best validation accuracy: {best_val_acc:.4f}'
    }


if __name__ == '__main__':
    result = main()
    print("\nFinal result:")
    print(json.dumps(result, indent=2))
