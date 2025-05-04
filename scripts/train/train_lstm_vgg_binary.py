#!/usr/bin/env python
"""
Train LSTM-VGG binary sequence classifier

This script uses a pretrained VGG16-BN (without its final classifier)
 to extract feature embeddings for each slice in a CT sequence, then trains
 an LSTM on fixed-length sequences for binary classification (NCP vs Normal).
 It downloads train/val folders from Azure Blob Storage, logs metrics to MLflow,
 and saves the best model and confusion matrix based on recall.
"""

import torch.multiprocessing as mp
# switch PyTorch multiprocessing to file-system mode to avoid /dev/shm limits
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
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
)
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from collections import Counter

from datasets import DatasetSequence2DBinary
from utils.download import download_from_blob, download_from_blob_with_access_key
from utils.log_config import get_custom_logger

my_logger = get_custom_logger('train_lstm_vgg_binary')

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure reproducibility
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
        missing, unexpected = self.vgg.load_state_dict(state, strict=False)
        self.embedding_dim = embedding_dim
        my_logger.info(
            f"Loaded VGG weights from {weights_path}. Missing keys: {missing}, Unexpected keys: {unexpected}"
        )

    def forward(self, x):
        # x: [B, 1, H, W] -> repeat to 3 channels, resize to 224x224
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
        out, _ = self.lstm(x)                   # [B, S, hidden_dim]
        out = self.dropout(out[:, -1, :])       # last time step
        return self.fc(out)


def train_model(train_loader: DataLoader,
                val_loader: DataLoader,
                num_epochs: int,
                lr: float,
                vgg_weights: str,
                seq_len: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    my_logger.info(f"Training on {device}, seq_len={seq_len}")

    # Feature extractor
    feat_ext = VGGFeatureExtractor(vgg_weights).to(device)
    feat_ext.eval()
    for param in feat_ext.parameters():
        param.requires_grad = False

    # LSTM classifier
    lstm = LSTMClassifier(input_dim=feat_ext.embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm.parameters(), lr=lr, weight_decay=1e-5)

    best_recall = 0.0
    best_metrics = {}
    best_model_path = best_cm_path = None
    patience = 3
    no_improve = 0
    start_time = time.time()

    # Chunking for feature extraction
    chunk_size = 64

    for epoch in range(1, num_epochs + 1):
        my_logger.info(f"--- Epoch {epoch}/{num_epochs} ---")
        # ----------
        # TRAINING
        # ----------
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
            total_samples += B

        train_loss = total_loss / total_samples
        train_acc = 100.0 * total_correct / total_samples
        mlflow.log_metric('train_loss', train_loss, step=epoch)
        mlflow.log_metric('train_accuracy', train_acc, step=epoch)
        my_logger.info(f"Train loss={train_loss:.4f}, acc={train_acc:.2f}%")

        # ------------
        # VALIDATION
        # ------------
        lstm.eval()
        val_loss = val_correct = val_count = 0
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                B, S, C, H, W = imgs.size()
                flat = imgs.view(B * S, C, H, W)

                feats_chunks = []
                for start in range(0, flat.size(0), chunk_size):
                    end = start + chunk_size
                    with autocast():
                        feats_chunks.append(feat_ext(flat[start:end]))
                feats = torch.cat(feats_chunks, dim=0).view(B, S, -1).float()

                logits = lstm(feats)
                batch_loss = criterion(logits, labels)
                val_loss += batch_loss.item() * B

                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                val_correct += preds.eq(labels).sum().item()
                val_count += B

                all_labels.extend(labels.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())

        val_loss /= val_count
        val_acc = 100.0 * val_correct / val_count
        all_labels = np.array(all_labels)
        all_probs  = np.array(all_probs)
        val_preds  = all_probs.argmax(axis=1)

        val_recall    = recall_score(all_labels, val_preds, average='binary', pos_label=1, zero_division=0)
        val_precision = precision_score(all_labels, val_preds, average='binary', pos_label=1, zero_division=0)
        val_f1        = f1_score(all_labels, val_preds, average='binary', pos_label=1, zero_division=0)
        try:
            fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1], pos_label=1)
            val_auc = auc(fpr, tpr)
        except Exception as e:
            my_logger.error("Error computing AUC: %s", e)
            val_auc = 0.0

        mlflow.log_metric('val_loss', val_loss, step=epoch)
        mlflow.log_metric('val_accuracy', val_acc, step=epoch)
        mlflow.log_metric('val_recall', val_recall, step=epoch)
        mlflow.log_metric('val_precision', val_precision, step=epoch)
        mlflow.log_metric('val_f1', val_f1, step=epoch)
        mlflow.log_metric('val_auc', val_auc, step=epoch)

        my_logger.info(
            f"Val loss={val_loss:.4f}, acc={val_acc:.2f}%, recall={val_recall:.2f}, "
            f"precision={val_precision:.2f}, f1={val_f1:.2f}, auc={val_auc:.2f}"
        )
        my_logger.info(f"Val label distribution: {Counter(all_labels.tolist())}")

        prefix = f"lstm_vgg_seq{seq_len}_{epoch}ep_{lr:.5f}lr_{val_recall:.3f}rec"
        os.makedirs('outputs', exist_ok=True)
        model_path = f"outputs/{prefix}.pth"

        if val_recall > best_recall:
            best_recall = val_recall
            best_metrics = {'loss': val_loss, 'acc': val_acc,
                            'precision': val_precision, 'f1': val_f1, 'auc': val_auc}
            best_model_path = model_path
            torch.save(lstm.state_dict(), model_path)
            my_logger.info(f"Saved new best model: {model_path}")

            cm = confusion_matrix(all_labels, val_preds)
            disp = ConfusionMatrixDisplay(cm)
            disp.plot()
            cm_path = f"outputs/{prefix}_confmat.png"
            plt.savefig(cm_path); plt.close()
            best_cm_path = cm_path
            my_logger.info(f"Saved confusion matrix: {cm_path}")

            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                my_logger.info("Early stopping triggered.")
                break

    elapsed = time.time() - start_time
    my_logger.info(f"Training complete in {elapsed:.2f}s")
    mlflow.log_metric('best_recall', best_recall)
    mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.items()})
    my_logger.info(f"Best metrics: {best_metrics}")
    my_logger.info(f"Model: {best_model_path}, CM: {best_cm_path}")


def main():
    parser = argparse.ArgumentParser(description="Train LSTM-VGG sequence model")
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir",   type=str, required=True)
    parser.add_argument("--vgg_model_uri", type=str, required=True)
    parser.add_argument("--sequence_length", type=int, required=True)
    parser.add_argument("--num_epochs",   type=int, default=20)
    parser.add_argument("--batch_size",   type=int, default=16)
    parser.add_argument("--learning_rate",type=float, default=0.0005)
    args = parser.parse_args()
    my_logger.info(f"Args: {args}")

    set_seed(42)
    load_dotenv()
    account   = os.getenv("AZURE_STORAGE_ACCOUNT")
    key       = os.getenv("AZURE_STORAGE_KEY")
    container = os.getenv("BLOB_CONTAINER")
    my_logger.info("Downloading data from blob storage")
    download_from_blob(account, key, container, args.train_dir)
    download_from_blob(account, key, container, args.val_dir)

    vgg_model_path = "models/vgg_binary_best.pth"
    download_from_blob_with_access_key(args.vgg_model_uri, key, vgg_model_path)

    mlflow.start_run()
    train_ds = DatasetSequence2DBinary(args.train_dir, args.sequence_length)
    val_ds   = DatasetSequence2DBinary(args.val_dir,   args.sequence_length)

    # reduced worker count to avoid shared-memory bus errors
    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=2,
                              pin_memory=True)

    my_logger.info(f"Loaded train seqs={len(train_ds)}, val seqs={len(val_ds)}")

    train_model(
        train_loader,
        val_loader,
        num_epochs=args.num_epochs,
        lr=args.learning_rate,
        vgg_weights=vgg_model_path,
        seq_len=args.sequence_length
    )
    mlflow.end_run()
    my_logger.info("MLflow run ended")

if __name__ == '__main__':
    main()
