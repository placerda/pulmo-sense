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
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
import mlflow
from dotenv import load_dotenv
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
)
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from datasets import DatasetSequence2DBinary
from utils.download import download_from_blob, download_from_blob_with_access_key
from utils.log_config import get_custom_logger

logger = get_custom_logger('train_lstm_vgg_binary')


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
        self.embedding_dim = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Identity()
        state = torch.load(weights_path, map_location='cpu')
        missing, unexpected = self.vgg.load_state_dict(state, strict=False)
        logger.info(
            f"Loaded VGG weights from {weights_path}. "
            f"Missing keys: {missing}, Unexpected keys: {unexpected}"
        )

    def forward(self, x):
        # x: [B, 1, H, W] → repeat to 3 channels and resize
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
    logger.info(f"Training on {device}, sequence_length={seq_len}")

    # Feature extractor
    feat_ext = VGGFeatureExtractor(vgg_weights).to(device)
    feat_ext.eval()
    for p in feat_ext.parameters():
        p.requires_grad = False

    # LSTM classifier
    model = LSTMClassifier(input_dim=feat_ext.embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_recall = 0.0
    best_metrics = {}
    best_model_path = best_cm_path = None
    patience = 3
    no_improve = 0
    start_time = time.time()

    chunk_size = 64  # for batching feature extraction

    for epoch in range(1, num_epochs + 1):
        logger.info(f"--- Epoch {epoch}/{num_epochs} ---")
        # ---- TRAIN ----
        model.train()
        running_loss = running_correct = running_total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            B, S, C, H, W = imgs.shape
            flat = imgs.view(B * S, C, H, W)

            # chunked feature extraction
            feats_chunks = []
            with torch.no_grad():
                for start in range(0, flat.size(0), chunk_size):
                    end = min(start + chunk_size, flat.size(0))
                    with autocast():
                        feats_chunks.append(feat_ext(flat[start:end]))
            feats = torch.cat(feats_chunks, dim=0).view(B, S, -1).float()

            logits = model(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * B
            preds = logits.argmax(dim=1)
            running_correct += preds.eq(labels).sum().item()
            running_total += B

        train_loss = running_loss / running_total
        train_acc = 100.0 * running_correct / running_total
        logger.info(f"[Train] Loss={train_loss:.6f}, Acc={train_acc:.2f}%")
        mlflow.log_metric('train_loss', train_loss, step=epoch)
        mlflow.log_metric('train_accuracy', train_acc, step=epoch)

        # ---- VALID ----
        model.eval()
        val_loss = val_correct = val_total = 0
        all_labels, all_preds, all_probs = [], [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                B, S, C, H, W = imgs.shape
                flat = imgs.view(B * S, C, H, W)

                feats_chunks = []
                for start in range(0, flat.size(0), chunk_size):
                    end = min(start + chunk_size, flat.size(0))
                    with autocast():
                        feats_chunks.append(feat_ext(flat[start:end]))
                feats = torch.cat(feats_chunks, dim=0).view(B, S, -1).float()

                logits = model(feats)
                loss = criterion(logits, labels)

                val_loss += loss.item() * B
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                val_correct += preds.eq(labels).sum().item()
                val_total += B

                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())

        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total
        lbl_arr = np.array(all_labels)
        pred_arr = np.array(all_preds)
        prob_arr = np.array(all_probs)

        val_recall    = recall_score(lbl_arr, pred_arr, average='binary', pos_label=1, zero_division=0)
        val_precision = precision_score(lbl_arr, pred_arr, average='binary', pos_label=1, zero_division=0)
        val_f1        = f1_score(lbl_arr, pred_arr, average='binary', pos_label=1, zero_division=0)
        try:
            fpr, tpr, _ = roc_curve(lbl_arr, prob_arr[:,1], pos_label=1)
            val_auc = auc(fpr, tpr)
        except:
            val_auc = 0.0

        logger.info(
            f"[Val] Loss={val_loss:.6f}, Acc={val_acc:.2f}%, "
            f"Recall={val_recall:.2f}, Prec={val_precision:.2f}, "
            f"F1={val_f1:.2f}, AUC={val_auc:.2f}"
        )
        logger.info(f"Val label distribution: {Counter(lbl_arr.tolist())}")

        mlflow.log_metrics({
            'val_loss':      val_loss,
            'val_accuracy':  val_acc,
            'val_recall':    val_recall,
            'val_precision': val_precision,
            'val_f1':        val_f1,
            'val_auc':       val_auc
        }, step=epoch)

        # ---- SAVE BEST & EARLY STOPPING ----
        prefix = f"lstm_vgg_seq{seq_len}_{epoch}ep_{lr:.5f}lr_{val_recall:.3f}rec"
        os.makedirs('outputs', exist_ok=True)
        model_path = f"outputs/{prefix}.pth"

        if val_recall > best_recall:
            best_recall = val_recall
            best_metrics = {
                'loss':      val_loss,
                'acc':       val_acc,
                'recall':    val_recall,
                'precision': val_precision,
                'f1':        val_f1,
                'auc':       val_auc
            }
            best_model_path = model_path
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved: {best_model_path}")

            cm = confusion_matrix(lbl_arr, pred_arr)
            disp = ConfusionMatrixDisplay(cm)
            disp.plot(cmap=plt.cm.Blues)
            best_cm_path = f"outputs/{prefix}_confmat.png"
            plt.savefig(best_cm_path)
            plt.close()
            logger.info(f"Confusion matrix saved: {best_cm_path}")

            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping triggered.")
                break

    elapsed = time.time() - start_time
    logger.info(f"Training complete in {elapsed:.2f}s, best recall={best_recall:.3f}")

    # ---- FINAL BEST MODEL SUMMARY ----
    logger.info("Best Model Metrics:")
    for k, v in best_metrics.items():
        logger.info(f"  {k} = {v}")
    if best_model_path and best_cm_path:
        logger.info(f"Best model file: {best_model_path}")
        logger.info(f"Best CM file   : {best_cm_path}")

    mlflow.log_metric('best_recall', best_recall)
    mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.items()})


def main():
    parser = argparse.ArgumentParser(description="Train LSTM-VGG sequence model")
    parser.add_argument("--train_dir",        type=str, required=True)
    parser.add_argument("--val_dir",          type=str, required=True)
    parser.add_argument("--vgg_model_uri",    type=str, required=True)
    parser.add_argument("--sequence_length",  type=int, default=30)
    parser.add_argument("--num_epochs",       type=int, default=20)
    parser.add_argument("--batch_size",       type=int, default=16)
    parser.add_argument("--learning_rate",    type=float, default=0.0005)
    args = parser.parse_args()

    logger.info(f"Args: {args}")
    set_seed(42)

    load_dotenv()
    acct, key, cont = (
        os.getenv("AZURE_STORAGE_ACCOUNT"),
        os.getenv("AZURE_STORAGE_KEY"),
        os.getenv("BLOB_CONTAINER"),
    )
    logger.info("Downloading data from blob storage")
    download_from_blob(acct, key, cont, args.train_dir)
    download_from_blob(acct, key, cont, args.val_dir)

    vgg_model_path = "models/vgg_binary_best.pth"
    download_from_blob_with_access_key(args.vgg_model_uri, key, vgg_model_path)

    mlflow.start_run()
    train_ds = DatasetSequence2DBinary(args.train_dir, args.sequence_length)
    val_ds   = DatasetSequence2DBinary(args.val_dir,   args.sequence_length)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    logger.info(f"Loaded train seqs={len(train_ds)}, val seqs={len(val_ds)}")

    train_model(
        train_loader,
        val_loader,
        num_epochs=args.num_epochs,
        lr=args.learning_rate,
        vgg_weights=vgg_model_path,
        seq_len=args.sequence_length
    )
    mlflow.end_run()
    logger.info("MLflow run ended")


if __name__ == '__main__':
    main()
