#!/usr/bin/env python
"""
Train LSTM-VGG binary sequence classifier on full dataset without validation or early stopping.
"""

import torch.multiprocessing as mp
# switch multiprocessing strategy
mp.set_sharing_strategy('file_system')

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
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from collections import Counter

from datasets import DatasetSequence2DBinary
from utils.download import download_from_blob, download_from_blob_with_access_key
from utils.log_config import get_custom_logger

logger = get_custom_logger('train_lstm_vgg_no_val')

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class VGGFeatureExtractor(nn.Module):
    """VGG16-BN feature extractor (classifier head removed)."""
    def __init__(self, weights_path: str):
        super().__init__()
        self.vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        embedding_dim = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Identity()
        state = torch.load(weights_path, map_location='cpu')
        self.vgg.load_state_dict(state, strict=False)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.vgg(x)

class LSTMClassifier(nn.Module):
    """LSTM-based sequence classifier."""
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 1, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # last time step
        return self.fc(out)


def train(train_loader, feat_ext, model, criterion, optimizer, device, epochs, chunk_size):
    start = time.time()
    for epoch in range(1, epochs + 1):
        logger.info(f"--- Epoch {epoch}/{epochs} ---")
        model.train()
        total_loss = total_correct = total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            B, S, C, H, W = imgs.size()
            flat = imgs.view(B * S, C, H, W)

            # chunked feature extraction
            feats_chunks = []
            with torch.no_grad():
                for start in range(0, flat.size(0), chunk_size):
                    end = start + chunk_size
                    with autocast():
                        feats_chunks.append(feat_ext(flat[start:end]))
            # concatenate and cast back to float32
            feats = torch.cat(feats_chunks, dim=0).view(B, S, -1).float()

            logits = lstm(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            preds = logits.argmax(dim=1)
            total_correct += preds.eq(labels).sum().item()
            total += B

        epoch_loss = total_loss / total
        epoch_acc = 100.0 * total_correct / total
        mlflow.log_metric('train_loss', epoch_loss, step=epoch)
        mlflow.log_metric('train_accuracy', epoch_acc, step=epoch)
        logger.info(f"Train loss={epoch_loss:.4f}, acc={epoch_acc:.2f}%")

    elapsed = time.time() - start
    logger.info(f"Training completed in {elapsed:.2f}s")

    os.makedirs('outputs', exist_ok=True)
    save_path = f"outputs/lstm_vgg_{epochs}ep.pth"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train LSTM-VGG binary model (no validation)")
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--vgg_model_uri', type=str, required=True)
    parser.add_argument('--sequence_length', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--chunk_size', type=int, default=64)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    load_dotenv()
    account = os.getenv('AZURE_STORAGE_ACCOUNT')
    key = os.getenv('AZURE_STORAGE_KEY')
    container = os.getenv('BLOB_CONTAINER')

    logger.info("Downloading training dataset from blob storage...")
    download_from_blob(account, key, container, args.train_dir)

    vgg_path = 'models/vgg_binary.pth'
    download_from_blob_with_access_key(args.vgg_model_uri, key, vgg_path)

    # prepare data loader
    train_ds = DatasetSequence2DBinary(args.train_dir, args.sequence_length)
    labels = train_ds.labels
    cnt = Counter(labels)
    logger.info(f"Train label distribution: {cnt}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )

    # model setup
    feat_ext = VGGFeatureExtractor(vgg_path).to(device)
    feat_ext.eval()
    for p in feat_ext.parameters(): p.requires_grad = False

    model = LSTMClassifier(input_dim=feat_ext.embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    mlflow.start_run()
    train(
        train_loader, feat_ext, model,
        criterion, optimizer,
        device, args.epochs, args.chunk_size
    )
    mlflow.end_run()

if __name__ == '__main__':
    main()
