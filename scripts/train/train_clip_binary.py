#!/usr/bin/env python
"""
Train CLIP-based binary classification model

This script trains a binary classifier using a pretrained HuggingFace CLIP vision encoder.
It handles single-channel input by replicating to 3 channels, resizing to 224×224,
downloads separate training and validation directories, and evaluates on sequences of slices.
"""

import argparse
import os
import time

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from dotenv import load_dotenv
from transformers import CLIPModel

from datasets.raster_dataset import Dataset2DBinary
from datasets import DatasetSequence2DBinary
from utils.download import download_from_blob
from utils.log_config import get_custom_logger

my_logger = get_custom_logger('train_clip_binary')


class CLIPBinaryModel(nn.Module):
    """
    CLIP wrapper for binary classification.
    """
    def __init__(self, model_name: str, num_classes: int = 2):
        super().__init__()
        # load vision encoder
        self.clip = CLIPModel.from_pretrained(model_name)
        hidden_size = self.clip.config.vision_config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # replicate single channel to 3 and resize
        x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        vision_outputs = self.clip.vision_model(pixel_values=x)
        pooled = vision_outputs.pooler_output
        return self.classifier(pooled)


def train_model(train_loader, val_seq_loader, num_epochs, learning_rate, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    my_logger.info(f"Starting CLIP ({model_name}) binary training on {device}")
    model = CLIPBinaryModel(model_name=model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    patience = 3
    no_improve = 0
    best_recall = 0.0
    best_epoch = 0
    best_metrics = {}
    best_model_path = None
    best_cm_path = None
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        my_logger.info(f"=== Epoch {epoch}/{num_epochs} ===")

        # — Training —
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for inputs, _, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            running_correct += preds.eq(labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total
        train_acc = 100.0 * running_correct / running_total
        mlflow.log_metric('train_loss', train_loss, step=epoch)
        mlflow.log_metric('train_accuracy', train_acc, step=epoch)
        my_logger.info(f"Train Loss={train_loss:.4f}, Acc={train_acc:.2f}%")

        # — Validation (sequence-level) —
        model.eval()
        all_labels, all_preds, all_probs = [], [], []
        with torch.no_grad():
            for seq_slices, seq_labels in val_seq_loader:
                B, L, C, H, W = seq_slices.shape
                flat = seq_slices.view(B * L, C, H, W).to(device)
                logits = model(flat)
                probs = torch.softmax(logits, dim=1).view(B, L, -1)
                avg_prob = probs.mean(dim=1)
                preds = avg_prob.argmax(dim=1)

                all_labels.extend(seq_labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(avg_prob.cpu().numpy())

        val_recall = recall_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0)
        val_precision = precision_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0)
        try:
            fpr, tpr, _ = roc_curve(all_labels, [p[1] for p in all_probs], pos_label=1)
            val_auc = auc(fpr, tpr)
        except Exception as e:
            my_logger.error(f"AUC computation failed: {e}")
            val_auc = 0.0

        mlflow.log_metric('val_recall', val_recall, step=epoch)
        mlflow.log_metric('val_precision', val_precision, step=epoch)
        mlflow.log_metric('val_f1', val_f1, step=epoch)
        mlflow.log_metric('val_auc', val_auc, step=epoch)
        my_logger.info(f"Val Recall={val_recall:.2f}, Precision={val_precision:.2f}, F1={val_f1:.2f}, AUC={val_auc:.2f}")

        # — Save best model by recall —
        prefix = f"clip_{model_name.replace('/', '_')}_{epoch}ep_{learning_rate:.5f}lr_{val_recall:.3f}rec"
        os.makedirs('outputs', exist_ok=True)
        model_path = f"outputs/{prefix}.pth"
        if val_recall > best_recall:
            best_recall = val_recall
            best_epoch = epoch
            best_metrics = {'loss': train_loss, 'acc': train_acc, 'precision': val_precision, 'f1': val_f1, 'auc': val_auc}
            best_model_path = model_path
            torch.save(model.state_dict(), model_path)
            my_logger.info(f"New best model saved: {model_path}")

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
    my_logger.info(f"Training completed in {total_time:.2f}s")

    # — Log best-model metrics —
    mlflow.log_metrics({
        'best_epoch': best_epoch,
        **{f"best_{k}": v for k, v in best_metrics.items()}
    }, step=best_epoch)
    my_logger.info(f"Best Epoch: {best_epoch}, Metrics: {best_metrics}")


def main():
    parser = argparse.ArgumentParser(description='Train CLIP binary model')
    parser.add_argument('--train_dir', type=str, required=True, help='Training data directory')
    parser.add_argument('--val_dir',   type=str, required=True, help='Validation data directory')
    parser.add_argument('--num_epochs', type=int,   default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int,   default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--sequence_length', type=int, default=30, help='Slices per sequence for validation')
    parser.add_argument('--model_name', type=str, default='openai/clip-vit-base-patch32', help='CLIP model name')
    args = parser.parse_args()
    my_logger.info(f"Args: {args}")

    load_dotenv()
    account = os.getenv('AZURE_STORAGE_ACCOUNT')
    key = os.getenv('AZURE_STORAGE_KEY')
    container = os.getenv('BLOB_CONTAINER')
    my_logger.info('Downloading data from blob storage')
    download_from_blob(account, key, container, args.train_dir)
    download_from_blob(account, key, container, args.val_dir)

    mlflow.start_run()
    train_ds = Dataset2DBinary(args.train_dir)
    val_seq_ds = DatasetSequence2DBinary(dataset_folder=args.val_dir, sequence_length=args.sequence_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_seq_loader = DataLoader(val_seq_ds, batch_size=args.batch_size, shuffle=False)
    my_logger.info(f"Loaded {len(train_ds)} train samples and {len(val_seq_ds)} validation sequences.")

    train_model(
        train_loader,
        val_seq_loader,
        num_epochs    = args.num_epochs,
        learning_rate = args.learning_rate,
        model_name    = args.model_name
    )

    mlflow.end_run()
    my_logger.info('MLflow run ended')

if __name__ == '__main__':
    main()
