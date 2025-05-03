#!/usr/bin/env python
"""
Train LSTM-VGG binary sequence classifier

This script uses a pretrained VGG16-BN (without its final classifier)
to extract feature embeddings for each slice in a CT sequence, then trains
an LSTM on fixed-length sequences for binary classification (NCP vs Normal).
It downloads train/val folders from Azure Blob Storage, logs metrics to MLflow,
and saves the best model and confusion matrix based on recall.
"""

import argparse
import os
import time

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
from torch.utils.data import DataLoader

from datasets import DatasetSequence2DBinary
from utils.download import download_from_blob
from utils.log_config import get_custom_logger

my_logger = get_custom_logger('train_lstm_vgg_binary')


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
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, num_classes=2, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1])
        return self.fc(out)


def train_model(train_loader, val_loader, num_epochs, lr, vgg_weights, seq_len):
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

    for epoch in range(1, num_epochs + 1):
        my_logger.info(f"--- Epoch {epoch}/{num_epochs} ---")
        # Training
        lstm.train()
        total_loss = total_correct = total_samples = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            B, S, C, H, W = imgs.shape
            flat = imgs.view(B * S, C, H, W)
            with torch.no_grad():
                feats = feat_ext(flat).view(B, S, -1)
            logits = lstm(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            preds = logits.argmax(1)
            total_correct += preds.eq(labels).sum().item()
            total_samples += B

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        mlflow.log_metric('train_loss', train_loss, step=epoch)
        mlflow.log_metric('train_accuracy', train_acc, step=epoch)
        my_logger.info(f"Train loss={train_loss:.4f}, acc={train_acc:.4f}")

        # Validation
        lstm.eval()
        val_loss = val_correct = val_count = 0
        all_labels, all_probs = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                B, S, C, H, W = imgs.shape
                flat = imgs.view(B * S, C, H, W)
                feats = feat_ext(flat).view(B, S, -1)
                logits = lstm(feats)

                val_loss += criterion(logits, labels).item() * B
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(1)
                val_correct += preds.eq(labels).sum().item()
                val_count += B

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        val_loss /= val_count
        val_acc = val_correct / val_count
        all_labels = np.array(all_labels)
        all_probs = np.vstack(all_probs)
        val_preds = all_probs.argmax(axis=1)

        val_recall = recall_score(all_labels, val_preds, average='binary', pos_label=1, zero_division=0)
        val_precision = precision_score(all_labels, val_preds, average='binary', pos_label=1, zero_division=0)
        val_f1 = f1_score(all_labels, val_preds, average='binary', pos_label=1, zero_division=0)
        try:
            fpr, tpr, _ = roc_curve(all_labels, all_probs[:,1], pos_label=1)
            val_auc = auc(fpr, tpr)
        except Exception:
            val_auc = 0.0

        for name, val in [('val_loss', val_loss), ('val_accuracy', val_acc),
                          ('val_recall', val_recall), ('val_precision', val_precision),
                          ('val_f1', val_f1), ('val_auc', val_auc)]:
            mlflow.log_metric(name, val, step=epoch)
        my_logger.info(
            f"Val loss={val_loss:.4f}, acc={val_acc:.4f}, recall={val_recall:.2f}, "
            f"precision={val_precision:.2f}, f1={val_f1:.2f}, auc={val_auc:.2f}"
        )

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
            disp.plot(cmap=plt.cm.Blues)
            cm_path = f"outputs/{prefix}_confmat.png"
            plt.savefig(cm_path)
            plt.close()
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
    parser.add_argument("--train_dir", type=str, required=True, help="Training folder")
    parser.add_argument("--val_dir", type=str, required=True, help="Validation folder")
    parser.add_argument("--vgg_model_path", type=str, required=True, help="VGG weights .pth path")
    parser.add_argument("--sequence_length", type=int, required=True, help="Frames per sequence")
    parser.add_argument("--num_epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate")
    args = parser.parse_args()
    my_logger.info(f"Args: {args}")

    load_dotenv()
    account = os.getenv("AZURE_STORAGE_ACCOUNT")
    key = os.getenv("AZURE_STORAGE_KEY")
    container = os.getenv("BLOB_CONTAINER")
    my_logger.info("Downloading data from blob storage")
    download_from_blob(account, key, container, args.train_dir)
    download_from_blob(account, key, container, args.val_dir)

    mlflow.start_run()
    train_ds = DatasetSequence2DBinary(args.train_dir, args.sequence_length)
    val_ds = DatasetSequence2DBinary(args.val_dir, args.sequence_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    my_logger.info(f"Loaded train seqs={len(train_ds)}, val seqs={len(val_ds)}")

    train_model(train_loader, val_loader,
                num_epochs=args.num_epochs,
                lr=args.learning_rate,
                vgg_weights=args.vgg_model_path,
                seq_len=args.sequence_length)
    mlflow.end_run()
    my_logger.info("MLflow run ended")

if __name__ == '__main__':
    main()
