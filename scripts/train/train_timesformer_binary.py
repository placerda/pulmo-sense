#!/usr/bin/env python3
"""
Train TimeSformer-based binary classification model

This script:
  - Downloads training and validation CT-image sequences from Azure Blob Storage.
  - Uses a pretrained TimeSformerForVideoClassification from Hugging Face to extract spatiotemporal features.
  - Handles single-channel input by replicating to RGB frames.
  - Trains for fixed-length sequences and logs metrics to MLflow.
  - Implements early stopping based on validation recall and saves the best model and confusion matrix.
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import mlflow
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
)
from PIL import Image

from transformers import (
    TimesformerConfig,
    TimesformerForVideoClassification,
    AutoImageProcessor
)
from datasets.raster_dataset import DatasetSequence2DBinary
from utils.download import download_from_blob
from utils.log_config import get_custom_logger

logger = get_custom_logger('train_timesformer_binary')

def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    processor: AutoImageProcessor,
    model: TimesformerForVideoClassification,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    patience: int
):
    best_recall = 0.0
    epochs_no_improve = 0
    best_epoch = 0
    best_metrics = {}
    best_dir = None

    mlflow.start_run()
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        logger.info(f"=== Epoch {epoch}/{num_epochs} ===")
        # --- Training ---
        model.train()
        train_loss = train_correct = train_total = 0
        for seq, labels in train_loader:
            # seq: [B, L, 1, H, W]
            B, L, C, H, W = seq.shape
            # to RGB and to numpy for processor
            frames = seq.repeat(1, 1, 3, 1, 1).permute(0,1,3,4,2).cpu().numpy()
            videos = [[Image.fromarray(frames[b,i]) for i in range(L)] for b in range(B)]
            inputs = processor(videos, return_tensors="pt").to(device)

            labels = labels.to(device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * B
            preds = logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += B

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        mlflow.log_metric('train_loss', avg_train_loss, step=epoch)
        mlflow.log_metric('train_accuracy', train_acc, step=epoch)
        logger.info(f"Train Loss={avg_train_loss:.4f}, Acc={train_acc:.2f}")

        # --- Validation ---
        model.eval()
        val_loss = val_correct = val_total = 0
        all_labels, all_probs = [], []
        with torch.no_grad():
            for seq, labels in val_loader:
                B, L, C, H, W = seq.shape
                frames = seq.repeat(1, 1, 3, 1, 1).permute(0,1,3,4,2).cpu().numpy()
                videos = [[Image.fromarray(frames[b,i]) for i in range(L)] for b in range(B)]
                inputs = processor(videos, return_tensors="pt").to(device)

                labels = labels.to(device)
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                val_loss += loss.item() * B
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += B

                all_labels.extend(labels.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        preds = np.argmax(all_probs, axis=1)
        recall = recall_score(all_labels, preds, average='binary', pos_label=1, zero_division=0)
        precision = precision_score(all_labels, preds, average='binary', pos_label=1, zero_division=0)
        f1 = f1_score(all_labels, preds, average='binary', pos_label=1, zero_division=0)
        try:
            fpr, tpr, _ = roc_curve(all_labels, [p[1] for p in all_probs], pos_label=1)
            auc_score = auc(fpr, tpr)
        except Exception:
            auc_score = 0.0

        # log metrics
        mlflow.log_metrics({
            'val_loss': avg_val_loss,
            'val_accuracy': val_acc,
            'val_recall': recall,
            'val_precision': precision,
            'val_f1': f1,
            'val_auc': auc_score
        }, step=epoch)
        logger.info(
            f"Val Loss={avg_val_loss:.4f}, Acc={val_acc:.2f}, Recall={recall:.2f}, "
            f"Prec={precision:.2f}, F1={f1:.2f}, AUC={auc_score:.2f}"
        )

        # early stopping & save best
        if recall > best_recall:
            best_recall = recall
            best_epoch = epoch
            best_metrics = {'loss': avg_val_loss, 'acc': val_acc, 'precision': precision, 'f1': f1, 'auc': auc_score}
            best_dir = f"outputs/timesformer_best_ep{epoch}"
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            logger.info(f"Saved best model to {best_dir}")
            # save confusion matrix
            cm = confusion_matrix(all_labels, preds)
            disp = ConfusionMatrixDisplay(cm)
            disp.plot(cmap=plt.cm.Blues)
            plt.savefig(os.path.join(best_dir, 'confmat.png'))
            plt.close()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info('Early stopping.')
                break

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.2f}s; best recall {best_recall:.3f} at epoch {best_epoch}")

    mlflow.log_metrics({
        'best_epoch': best_epoch,
        'best_recall': best_recall,
        'best_precision': best_metrics.get('precision'),
        'best_f1': best_metrics.get('f1'),
        'best_auc': best_metrics.get('auc')
    }, step=best_epoch)
    mlflow.end_run()


def main():
    parser = argparse.ArgumentParser(description='Train TimeSformer binary model')
    parser.add_argument('--train_dir', type=str, required=True, help='Training data directory')
    parser.add_argument('--val_dir', type=str, required=True, help='Validation data directory')
    parser.add_argument('--sequence_length', type=int, default=30, help='Frames per sequence')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--model_name_or_path', type=str,
                        default='facebook/timesformer-base-finetuned-k400',
                        help='Pretrained TimeSformer model')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    args = parser.parse_args()

    # Download and prepare
    load_dotenv()
    train_ds = DatasetSequence2DBinary(args.train_dir, args.sequence_length)
    val_ds = DatasetSequence2DBinary(args.val_dir, args.sequence_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # load model & processor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = TimesformerConfig.from_pretrained(args.model_name_or_path, num_labels=2)
    model = TimesformerForVideoClassification.from_pretrained(
        args.model_name_or_path, config=config, ignore_mismatched_sizes=True
    ).to(device)
    processor = AutoImageProcessor.from_pretrained(args.model_name_or_path)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    # run training
    train_model(
        train_loader, val_loader,
        device, processor, model,
        criterion, optimizer,
        num_epochs=args.num_epochs,
        patience=args.patience
    )

if __name__ == '__main__':
    main()
