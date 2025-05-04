#!/usr/bin/env python
"""
Train LSTM + Attention model with VGG features for binary classification (NCP vs Normal)
on full dataset without validation or early stopping.
"""

import argparse
import os
import time
import random

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from collections import Counter

from datasets import DatasetSequence2DBinary
from utils.download import download_from_blob, download_from_blob_with_access_key
from utils.log_config import get_custom_logger

logger = get_custom_logger('train_lstm_attention_vgg_no_val')

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class VGGFeatureExtractor(nn.Module):
    """VGG16-BN backbone, classifier head removed."""
    def __init__(self, weights_path: str):
        super().__init__()
        backbone = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        embedding_dim = backbone.classifier[6].in_features
        backbone.classifier[6] = nn.Identity()
        state = torch.load(weights_path, map_location='cpu')
        backbone.load_state_dict(state, strict=False)
        self.backbone = backbone
        self.embedding_dim = embedding_dim

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.backbone(x)

class TemporalAttention(nn.Module):
    """Attention over time dimension."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, outputs: torch.Tensor):
        energies = self.score(outputs).squeeze(-1)
        weights = torch.softmax(energies, dim=1)
        context = (outputs * weights.unsqueeze(-1)).sum(dim=1)
        return context

class LSTMAttentionClassifier(nn.Module):
    """LSTM + temporal attention sequence classifier."""
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 1, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attn = TemporalAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        outputs, _ = self.lstm(x)
        context = self.attn(outputs)
        return self.fc(self.dropout(context))


def train(train_loader, feat_ext, clf, criterion, optimizer, device, epochs, chunk_size):
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        logger.info(f"--- Epoch {epoch}/{epochs} ---")
        clf.train()
        running_loss = correct = total = 0

        for imgs, labels in train_loader:
            B, S, C, H, W = imgs.shape
            imgs, labels = imgs.to(device), labels.to(device)
            flat = imgs.view(B * S, C, H, W)

            feats_chunks = []
            with torch.no_grad():
                for start in range(0, flat.size(0), chunk_size):
                    end = start + chunk_size
                    with autocast():
                        feats_chunks.append(feat_ext(flat[start:end]))
                feats = torch.cat(feats_chunks, dim=0).view(B, S, -1)

            logits = clf(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * B
            preds = logits.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += B

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        mlflow.log_metric('train_loss', epoch_loss, step=epoch)
        mlflow.log_metric('train_accuracy', epoch_acc, step=epoch)
        logger.info(f"Train loss={epoch_loss:.4f}, acc={epoch_acc:.2f}%")

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.2f}s")

    # save final model
    os.makedirs('outputs', exist_ok=True)
    model_path = f"outputs/lstm_attn_seq{args.sequence_length}_{epochs}ep.pth"
    torch.save(clf.state_dict(), model_path)
    logger.info(f"Model saved: {model_path}")


def main():
    parser = argparse.ArgumentParser("Train LSTM+Attention VGG binary (no val)")
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--vgg_model_uri', type=str, required=True)
    parser.add_argument('--sequence_length', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--chunk_size', type=int, default=64,
                        help='Feature extraction chunk size')
    args = parser.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    load_dotenv()
    acct = os.getenv('AZURE_STORAGE_ACCOUNT')
    key = os.getenv('AZURE_STORAGE_KEY')
    cont = os.getenv('BLOB_CONTAINER')
    logger.info("Downloading training dataset from blob...")
    download_from_blob(acct, key, cont, args.train_dir)

    vgg_path = 'models/vgg_binary.pth'
    download_from_blob_with_access_key(args.vgg_model_uri, key, vgg_path)


    # Data
    train_ds = DatasetSequence2DBinary(args.train_dir, args.sequence_length)
    train_labels = train_ds.labels
    cnt = Counter(train_labels)
    logger.info(f"Train label distribution: {cnt}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )

    # Models
    feat_ext = VGGFeatureExtractor(vgg_path).to(device)
    feat_ext.eval()
    for p in feat_ext.parameters(): p.requires_grad = False

    clf = LSTMAttentionClassifier(input_dim=feat_ext.embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clf.parameters(), lr=args.lr, weight_decay=1e-5)

    mlflow.start_run()
    train(
        train_loader, feat_ext, clf, criterion, optimizer,
        device, args.epochs, args.chunk_size
    )
    mlflow.end_run()

if __name__ == '__main__':
    main()
