#!/usr/bin/env python
"""
Train LSTM-VGG binary sequence classifier on the full dataset (no validation).

This script:
  - Switches PyTorch multiprocessing to file-system sharing.
  - Downloads only the training folder from Azure Blob Storage.
  - Uses a pretrained VGG16-BN (classifier head removed) to extract per-slice embeddings.
  - Trains an LSTM on fixed-length CT sequences for binary classification (NCP vs Normal).
  - Logs train_loss and train_accuracy to MLflow each epoch.
  - Saves the final model at the end.
"""

import torch.multiprocessing as mp
# avoid /dev/shm limits
mp.set_sharing_strategy('file_system')

import argparse
import os
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import mlflow
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from dotenv import load_dotenv

from datasets import DatasetSequence2DBinary
from utils.download import download_from_blob_with_access_key, download_from_blob
from utils.log_config import get_custom_logger

logger = get_custom_logger('train_lstm_vgg_full')

class VGGFeatureExtractor(nn.Module):
    """VGG16-BN backbone feature extractor (classifier head removed)."""
    def __init__(self, weights_path: str):
        super().__init__()
        backbone = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        self.embedding_dim = backbone.classifier[6].in_features
        backbone.classifier[6] = nn.Identity()
        state = torch.load(weights_path, map_location='cpu')
        backbone.load_state_dict(state, strict=False)
        self.backbone = backbone

    def forward(self, x):
        # x: [B*seq_len, 1, H, W] → convert to 3-ch and resize
        x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(x, size=(224, 224),
                                      mode='bilinear', align_corners=False)
        return self.backbone(x)

class LSTMClassifier(nn.Module):
    """LSTM sequence classifier."""
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 1, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [B, S, input_dim]
        outputs, _ = self.lstm(x)
        last = outputs[:, -1, :]             # take last time step
        return self.fc(self.dropout(last))

def train_full(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device={device}, seq_len={args.sequence_length}")

    # Download training data
    load_dotenv()
    acct, key, cont = (
        os.getenv('AZURE_STORAGE_ACCOUNT'),
        os.getenv('AZURE_STORAGE_KEY'),
        os.getenv('BLOB_CONTAINER'),
    )
    logger.info("Downloading full training set...")
    download_from_blob(acct, key, cont, args.train_dir)

    # Download pretrained VGG weights
    vgg_path = "models/vgg_backbone.pth"
    os.makedirs(os.path.dirname(vgg_path), exist_ok=True)
    logger.info("Downloading VGG weights...")
    download_from_blob_with_access_key(args.vgg_model_uri, key, vgg_path)

    # Prepare dataset + loader
    train_ds = DatasetSequence2DBinary(args.train_dir, args.sequence_length)
    labels = train_ds.labels
    dist = Counter(labels)
    logger.info(f"Train label distribution: {dist}")

    if args.use_sampler:
        weights = [1.0 / dist[int(l)] for l in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                        replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=2, pin_memory=True)
        class_weights = None
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=2, pin_memory=True)
        if len(dist) == 2:
            total = sum(dist.values())
            class_weights = [total / dist[i] for i in range(2)]
            logger.info(f"Using class weights={class_weights}")
        else:
            class_weights = None

    # Build models
    feat_ext = VGGFeatureExtractor(vgg_path).to(device)
    feat_ext.eval()
    for p in feat_ext.parameters():
        p.requires_grad = False

    clf = LSTMClassifier(input_dim=feat_ext.embedding_dim).to(device)
    if class_weights:
        cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(clf.parameters(),
                           lr=args.learning_rate, weight_decay=1e-5)
    scaler = GradScaler()

    # MLflow run
    mlflow.start_run()
    start_time = time.time()

    chunk_size = 64
    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"=== Epoch {epoch}/{args.num_epochs} ===")
        clf.train()
        running_loss = running_corr = running_total = 0

        for imgs, lbls in train_loader:
            B, S, C, H, W = imgs.shape
            imgs, lbls = imgs.to(device), lbls.to(device)
            flat = imgs.view(B * S, C, H, W)

            # extract VGG features in chunks
            feats_list = []
            with torch.no_grad():
                for i in range(0, flat.size(0), chunk_size):
                    end = i + chunk_size
                    feats_list.append(feat_ext(flat[i:end]))
            feats = torch.cat(feats_list, dim=0).view(B, S, -1)

            optimizer.zero_grad()
            with autocast():
                logits = clf(feats)
                loss = criterion(logits, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = logits.argmax(dim=1)
            running_loss    += loss.item() * B
            running_corr    += (preds == lbls).sum().item()
            running_total   += B

        train_loss = running_loss / running_total
        train_acc  = running_corr / running_total
        logger.info(f"[Train] Loss={train_loss:.6f}, Acc={train_acc:.4f}")

        mlflow.log_metric('train_loss', train_loss, step=epoch)
        mlflow.log_metric('train_accuracy', train_acc, step=epoch)

    elapsed = time.time() - start_time
    logger.info(f"Training finished in {elapsed:.2f}s")

    # Save final model
    os.makedirs('outputs', exist_ok=True)
    out_path = f"outputs/lstm_vgg_full_{args.num_epochs}ep.pth"
    torch.save(clf.state_dict(), out_path)
    logger.info(f"Model saved: {out_path}")

    mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LSTM-VGG on full dataset (no validation)"
    )
    parser.add_argument('--train_dir',       type=str, required=True,
                        help="Folder with full training CT sequences")
    parser.add_argument('--vgg_model_uri',   type=str, required=True,
                        help="Blob URI for pretrained VGG weights")
    parser.add_argument('--sequence_length', type=int, default=30,
                        help="Number of slices per sequence")
    parser.add_argument('--num_epochs',      type=int, default=20,
                        help="Total epochs to train")
    parser.add_argument('--batch_size',      type=int, default=16)
    parser.add_argument('--learning_rate',   type=float, default=5e-4)
    parser.add_argument('--use_sampler',     action='store_true',
                        help="Use weighted sampling over classes")
    args = parser.parse_args()

    train_full(args)
