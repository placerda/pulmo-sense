#!/usr/bin/env python
"""
Train LSTM + Attention model with VGG features for binary classification (NCP vs Normal).

This script:
  - Downloads training and validation data from Azure Blob Storage.
  - Extracts per-slice embeddings via a pretrained VGG16-BN (classifier head removed).
  - Trains an LSTM with temporal attention on fixed-length CT sequences.
  - Logs train/validation metrics (loss, accuracy, recall, precision, F1, AUC) to MLflow.
  - Saves the best model and confusion matrix based on validation recall.
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
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
)
from collections import Counter

from datasets import DatasetSequence2DBinary
from utils.download import download_from_blob
from utils.log_config import get_custom_logger

logger = get_custom_logger('train_lstm_attention_vgg_binary')

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
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        logger.info(
            f"Loaded VGG weights from {weights_path}. Missing keys: {missing}, Unexpected: {unexpected}"
        )

    def forward(self, x):
        # x: [B*seq_len, 1, H, W]
        x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.backbone(x)

class TemporalAttention(nn.Module):
    """Attention over time dimension."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, outputs: torch.Tensor):
        # outputs: [B, seq_len, hidden_dim]
        energies = self.score(outputs).squeeze(-1)         # [B, seq_len]
        weights = torch.softmax(energies, dim=1)          # [B, seq_len]
        context = (outputs * weights.unsqueeze(-1)).sum(dim=1)  # [B, hidden_dim]
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
        # x: [B, seq_len, input_dim]
        outputs, _ = self.lstm(x)
        context = self.attn(outputs)                    # [B, hidden_dim]
        return self.fc(self.dropout(context))           # [B, num_classes]


def train(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Download data
    load_dotenv()
    acct = os.getenv('AZURE_STORAGE_ACCOUNT')
    key = os.getenv('AZURE_STORAGE_KEY')
    cont = os.getenv('BLOB_CONTAINER')
    logger.info("Downloading datasets from blob storage...")
    download_from_blob(acct, key, cont, args.train_dir)
    download_from_blob(acct, key, cont, args.val_dir)

    # Datasets and loaders
    train_ds = DatasetSequence2DBinary(args.train_dir, args.sequence_length)
    val_ds   = DatasetSequence2DBinary(args.val_dir,   args.sequence_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    logger.info(f"Loaded train sequences: {len(train_ds)}, val: {len(val_ds)}")

    # Models
    feat_ext = VGGFeatureExtractor(args.vgg_weights).to(device)
    feat_ext.eval()
    for p in feat_ext.parameters():
        p.requires_grad = False

    clf = LSTMAttentionClassifier(input_dim=feat_ext.embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clf.parameters(), lr=args.lr, weight_decay=1e-5)

    best_recall = 0.0
    best_epoch = 0
    best_metrics = {}
    best_model_path = best_cm_path = None
    no_improve = 0
    start_time = time.time()

    # Mixed-precision chunk settings
    chunk_size = 64

    mlflow.start_run()
    for epoch in range(1, args.epochs + 1):
        logger.info(f"--- Epoch {epoch}/{args.epochs} ---")

        # Training
        clf.train()
        train_loss = train_correct = train_total = 0
        for imgs, labels in train_loader:
            B, S, C, H, W = imgs.shape
            imgs, labels = imgs.to(device), labels.to(device)
            flat = imgs.view(B * S, C, H, W)

            # Chunked feature extraction
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

            train_loss += loss.item() * B
            preds = logits.argmax(dim=1)
            train_correct += preds.eq(labels).sum().item()
            train_total += B

        train_loss /= train_total
        train_acc = 100.0 * train_correct / train_total
        mlflow.log_metric('train_loss', train_loss, step=epoch)
        mlflow.log_metric('train_accuracy', train_acc, step=epoch)
        logger.info(f"Train loss={train_loss:.4f}, acc={train_acc:.2f}%")

        # Validation
        clf.eval()
        val_loss = val_correct = val_total = 0
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                B, S, C, H, W = imgs.shape
                imgs, labels = imgs.to(device), labels.to(device)
                flat = imgs.view(B * S, C, H, W)

                feats_chunks = []
                for start in range(0, flat.size(0), chunk_size):
                    end = start + chunk_size
                    with autocast():
                        feats_chunks.append(feat_ext(flat[start:end]))
                feats = torch.cat(feats_chunks, dim=0).view(B, S, -1)

                logits = clf(feats)
                batch_loss = criterion(logits, labels)
                val_loss += batch_loss.item() * B

                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                val_correct += preds.eq(labels).sum().item()
                val_total += B

                all_labels.extend(labels.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())

        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total
        all_labels = np.array(all_labels)
        all_probs = np.vstack(all_probs)
        val_preds = all_probs.argmax(axis=1)

        val_recall = recall_score(all_labels, val_preds, average='binary', pos_label=1, zero_division=0)
        val_precision = precision_score(all_labels, val_preds, average='binary', pos_label=1, zero_division=0)
        val_f1 = f1_score(all_labels, val_preds, average='binary', pos_label=1, zero_division=0)
        try:
            fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1], pos_label=1)
            val_auc = auc(fpr, tpr)
        except Exception as e:
            logger.error("Error computing AUC: %s", e)
            val_auc = 0.0

        # Log validation metrics
        metrics = {
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_recall': val_recall,
            'val_precision': val_precision,
            'val_f1': val_f1,
            'val_auc': val_auc
        }
        for name, value in metrics.items():
            mlflow.log_metric(name, value, step=epoch)
        logger.info(
            f"Val loss={val_loss:.4f}, acc={val_acc:.2f}%, recall={val_recall:.2f}, "
            f"prec={val_precision:.2f}, f1={val_f1:.2f}, auc={val_auc:.2f}"
        )
        logger.info(f"Val label distribution: {Counter(all_labels.tolist())}")

        # Save best by recall
        prefix = f"attn_lstm_seq{args.sequence_length}_{epoch}ep_{val_recall:.3f}rec"
        os.makedirs('outputs', exist_ok=True)
        if val_recall > best_recall:
            best_recall = val_recall
            best_epoch = epoch
            best_metrics = metrics
            best_model_path = f"outputs/{prefix}.pth"
            torch.save(clf.state_dict(), best_model_path)
            logger.info(f"Saved new best model: {best_model_path}")

            cm = confusion_matrix(all_labels, val_preds)
            disp = ConfusionMatrixDisplay(cm)
            disp.plot()  # default colormap
            best_cm_path = f"outputs/{prefix}_cm.png"
            plt.savefig(best_cm_path); plt.close()
            logger.info(f"Saved confusion matrix: {best_cm_path}")

            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info("Early stopping triggered.")
                break

    elapsed = time.time() - start_time
    logger.info(f"Training complete in {elapsed:.2f}s, best recall={best_recall:.3f} at epoch {best_epoch}")
    mlflow.log_metrics({'best_recall': best_recall, 'best_epoch': best_epoch})
    mlflow.end_run()


def main():
    parser = argparse.ArgumentParser("Train LSTM+Attention VGG binary")
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--vgg_weights', type=str, required=True)
    parser.add_argument('--sequence_length', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=3)
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
