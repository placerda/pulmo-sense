#!/usr/bin/env python
"""
Train MAE-based binary model

This script trains a binary classifier using a pretrained Masked Autoencoder (MAE)
from timm. It handles single-channel input by replicating to 3 channels,
resizes to 224×224, downloads separate training and validation directories,
and evaluates on sequences of slices.

The training and validation loops log metrics to MLflow at each epoch. The best model
(by validation recall) is saved, along with its confusion matrix. After training,
we log the best-model metrics, including the epoch, loss, accuracy, recall, precision,
F1, and AUC, to MLflow and print their values and file locations.
"""

import argparse
import os
import time

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from datasets.raster_dataset import Dataset2DBinary
from datasets import DatasetSequence2DBinary
from utils.download import download_from_blob
from utils.log_config import get_custom_logger

my_logger = get_custom_logger('train_mae_binary')


class MAEModel(nn.Module):
    """
    MAE wrapper for binary classification.
    """
    def __init__(self, model_name: str, num_classes: int = 2):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.model(x)


def train_model(train_loader, val_seq_loader, num_epochs, learning_rate, model_name):
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    my_logger.info(f"Starting MAE ({model_name}) binary training on {device}")

    model = MAEModel(model_name=model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    patience = 3
    no_improve = 0
    best_recall = 0.0
    best_epoch = 0
    best_metrics = {'loss': None, 'acc': None, 'precision': None, 'f1': None, 'auc': None}
    best_model_path = None
    best_cm_path = None

    for epoch in range(1, num_epochs + 1):
        my_logger.info(f"=== Epoch {epoch}/{num_epochs} ===")

        # — Training —
        model.train()
        total_loss = total_correct = total_samples = 0
        for i, (inputs, _, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += preds.eq(labels).sum().item()
            total_samples += labels.size(0)

            if i % 50 == 0:
                batch_acc = preds.eq(labels).float().mean().item()
                my_logger.info(
                    f"[Train] Batch {i}/{len(train_loader)} — "
                    f"Loss={loss.item():.4f}, Acc={batch_acc:.4f}"
                )

        train_loss = total_loss / total_samples
        train_acc = 100.0 * total_correct / total_samples
        my_logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%")
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)

        # — Validation (sequence-level) —
        model.eval()
        val_loss = val_correct = val_total = 0
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for seq_slices, seq_labels in val_seq_loader:
                B, L, C, H, W = seq_slices.shape
                seq_slices = seq_slices.to(device)
                labels = seq_labels.to(device)

                flat = seq_slices.view(B * L, C, H, W)
                logits = model(flat).view(B, L, -1)
                seq_logits = logits.mean(dim=1)

                loss = criterion(seq_logits, labels)
                val_loss += loss.item() * B

                probs = torch.softmax(seq_logits, dim=1)
                preds = probs.argmax(dim=1)
                val_correct += preds.eq(labels).sum().item()
                val_total += B

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        val_loss = val_loss / val_total
        val_acc = 100.0 * val_correct / val_total
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        val_recall = recall_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0)
        val_precision = precision_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0)
        try:
            fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1], pos_label=1)
            val_auc = auc(fpr, tpr)
        except Exception as e:
            my_logger.error(f"AUC computation error: {e}")
            val_auc = 0.0

        my_logger.info(
            f"Epoch {epoch} Validation — "
            f"Loss={val_loss:.4f}, Acc={val_acc:.2f}%, "
            f"Recall={val_recall:.2f}, Precision={val_precision:.2f}, "
            f"F1={val_f1:.2f}, AUC={val_auc:.2f}"
        )
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        mlflow.log_metric("val_recall", val_recall, step=epoch)
        mlflow.log_metric("val_precision", val_precision, step=epoch)
        mlflow.log_metric("val_f1", val_f1, step=epoch)
        mlflow.log_metric("val_auc", val_auc, step=epoch)

        # — Save best model by recall —
        if val_recall > best_recall:
            best_recall = val_recall
            best_epoch = epoch
            best_metrics = {
                'loss': val_loss,
                'acc': val_acc,
                'precision': val_precision,
                'f1': val_f1,
                'auc': val_auc
            }
            prefix = f"mae_{model_name}_{epoch}epoch_{learning_rate:.5f}lr_{val_recall:.3f}rec"
            os.makedirs("outputs", exist_ok=True)
            best_model_path = f"outputs/{prefix}.pth"
            torch.save(model.state_dict(), best_model_path)
            my_logger.info(f"New best model saved: {best_model_path}")

            cm = confusion_matrix(all_labels, all_preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap=plt.cm.Blues)
            best_cm_path = f"outputs/{prefix}_confmat.png"
            plt.savefig(best_cm_path)
            plt.close()
            my_logger.info(f"Confusion matrix saved: {best_cm_path}")
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            my_logger.info("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    my_logger.info(f"Training completed in {total_time:.2f}s; "
                   f"best recall={best_recall:.3f} at epoch {best_epoch}")

    # — Log best-model metrics to MLflow —
    final_metrics = {
        "best_model_epoch":    best_epoch,
        "best_model_loss":     best_metrics['loss'],
        "best_model_acc":      best_metrics['acc'],
        "best_model_recall":   best_recall,
        "best_model_precision":best_metrics['precision'],
        "best_model_f1":       best_metrics['f1'],
        "best_model_auc":      best_metrics['auc']
    }
    mlflow.log_metrics(final_metrics, step=best_epoch)

    my_logger.info("Best Model Metrics:")
    for name, value in final_metrics.items():
        my_logger.info(f"  {name} = {value}")

    if best_model_path:
        my_logger.info(f"Best model file: {best_model_path}")
        my_logger.info(f"Best CM file   : {best_cm_path}")


def main():
    parser = argparse.ArgumentParser(description="Train MAE binary model")
    parser.add_argument("--train_dir", type=str, required=True, help="Training data directory")
    parser.add_argument("--val_dir",   type=str, required=True, help="Validation data directory")
    parser.add_argument("--num_epochs", type=int, default=20, help="Epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--sequence_length", type=int, default=30, help="Slices per sequence for validation")
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224.mae", help="timm MAE model name")
    args = parser.parse_args()
    my_logger.info(f"Args: {args}")

    load_dotenv()
    account   = os.getenv("AZURE_STORAGE_ACCOUNT")
    key       = os.getenv("AZURE_STORAGE_KEY")
    container = os.getenv("BLOB_CONTAINER")
    my_logger.info("Downloading training and validation data...")
    download_from_blob(account, key, container, args.train_dir)
    download_from_blob(account, key, container, args.val_dir)

    mlflow.start_run()
    my_logger.info("Loading datasets...")
    train_dataset   = Dataset2DBinary(args.train_dir)
    val_seq_dataset = DatasetSequence2DBinary(dataset_folder=args.val_dir,
                                              sequence_length=args.sequence_length)

    train_loader   = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_seq_loader = DataLoader(val_seq_dataset, batch_size=args.batch_size, shuffle=False)
    my_logger.info(
        f"Loaded {len(train_dataset)} train samples and {len(val_seq_dataset)} validation sequences."
    )

    train_model(
        train_loader,
        val_seq_loader,
        num_epochs    = args.num_epochs,
        learning_rate = args.learning_rate,
        model_name    = args.model_name
    )

    mlflow.end_run()
    my_logger.info("MLflow run ended")


if __name__ == "__main__":
    main()
