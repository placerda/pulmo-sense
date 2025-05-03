#!/usr/bin/env python
"""
Train VGG Binary Model with mixed precision and memory-friendly validation.

This script trains a VGG model for binary classification,
using separate training and validation datasets provided as inputs.
Validation is done sequence-level in small chunks to reduce peak GPU memory.
"""

import argparse
import os
import time
from collections import Counter

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from dotenv import load_dotenv
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
import matplotlib.pyplot as plt

from datasets.raster_dataset import Dataset2DBinary
from datasets import DatasetSequence2DBinary
from utils.log_config import get_custom_logger
from utils.download import download_from_blob

my_logger = get_custom_logger('train_vgg_binary')


class VGG_Net(nn.Module):
    """
    Wrapper around pretrained VGG16-BN for binary classification.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # input x: [batch, 1, H, W] → repeat to 3 channels
        x = x.repeat(1, 3, 1, 1)
        return self.model(x)


def train_model(train_loader, val_seq_loader, device, num_epochs, learning_rate, class_weights=None):
    start_time = time.time()
    my_logger.info("Starting VGG binary training")
    my_logger.info(f"Using device: {device}")

    model = VGG_Net(num_classes=2).to(device)
    if class_weights is not None:
        cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=cw)
        my_logger.info(f"Using class weights: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scaler = GradScaler()

    best_recall = -1.0
    best_epoch = 0
    best_loss = best_acc = best_precision = best_f1 = best_auc = None
    best_model_path = best_cm_path = None
    early_stopping_patience = 3
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        my_logger.info(f"=== Epoch {epoch}/{num_epochs} ===")

        # ---------
        # TRAINING
        # ---------
        model.train()
        running_loss = running_correct = running_total = 0

        for i, (inputs, _, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # forward in mixed precision
            with autocast():
                outputs = model(inputs)
            # compute loss in full precision
            loss = criterion(outputs.float(), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * labels.size(0)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

            if i % 50 == 0:
                batch_acc = (preds == labels).float().mean().item()
                my_logger.info(
                    f"[Train] Batch {i}/{len(train_loader)} – "
                    f"Loss={loss.item():.6f}, Acc={batch_acc:.4f}"
                )

        train_loss = running_loss / running_total
        train_acc  = 100.0 * running_correct / running_total
        my_logger.info(f"Epoch {epoch}: Train Loss={train_loss:.6f}, Train Acc={train_acc:.2f}%")
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)

        # --------------
        # VALIDATION
        # --------------
        model.eval()
        val_loss = val_corr = val_total = 0
        all_seq_labels = []
        all_seq_preds  = []
        all_seq_probs  = []

        with torch.no_grad():
            for seq_slices, seq_label in val_seq_loader:
                B, L, C, H, W = seq_slices.shape
                seq_slices = seq_slices.to(device)
                labels = seq_label.to(device)

                # chunked forward pass to save memory
                chunk_size = 10
                logits_chunks = []
                for start in range(0, L, chunk_size):
                    end = min(start + chunk_size, L)
                    flat = seq_slices[:, start:end].reshape(-1, C, H, W)
                    with autocast():
                        out = model(flat)
                    logits_chunks.append(out.view(B, -1, 2))
                slice_logits = torch.cat(logits_chunks, dim=1)  # [B, L, 2]

                seq_logits = slice_logits.mean(dim=1)           # [B, 2]
                # compute loss in full precision
                loss = criterion(seq_logits.float(), labels)
                val_loss += loss.item() * B

                prob  = torch.softmax(seq_logits, dim=1)
                preds = prob.argmax(dim=1)
                val_corr  += (preds == labels).sum().item()
                val_total += B

                all_seq_labels.extend(labels.cpu().numpy())
                all_seq_preds.extend(preds.cpu().numpy())
                all_seq_probs.extend(prob.cpu().numpy())

        val_loss /= val_total
        val_acc  = 100.0 * val_corr / val_total
        labels_arr = np.array(all_seq_labels)
        preds_arr  = np.array(all_seq_preds)
        probs_arr  = np.array(all_seq_probs)

        val_recall    = recall_score(labels_arr, preds_arr, average='binary', pos_label=1, zero_division=0)
        val_precision = precision_score(labels_arr, preds_arr, average='binary', pos_label=1, zero_division=0)
        val_f1        = f1_score(labels_arr, preds_arr, average='binary', pos_label=1, zero_division=0)
        try:
            fpr, tpr, _ = roc_curve(labels_arr, probs_arr[:, 1], pos_label=1)
            val_auc = auc(fpr, tpr)
        except Exception as e:
            my_logger.error("Error computing AUC: %s", e)
            val_auc = 0.0

        my_logger.info(
            f"Epoch {epoch} Validation:"
            f" Loss={val_loss:.6f}, Acc={val_acc:.2f}%"
            f" Recall={val_recall:.2f}, Prec={val_precision:.2f}, F1={val_f1:.2f}, AUC={val_auc:.2f}"
        )
        my_logger.info(f"Val label distribution: {Counter(labels_arr.tolist())}")
        my_logger.info(f"Val predicted classes:    {sorted(set(all_seq_preds))}")

        mlflow.log_metrics({
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_recall": val_recall,
            "val_precision": val_precision,
            "val_f1_score": val_f1,
            "val_auc": val_auc
        }, step=epoch)

        # ------------------
        # SAVE BEST BY RECALL
        # ------------------
        prefix     = f"vgg_{epoch}ep_{learning_rate:.5f}lr_{val_recall:.3f}rec"
        os.makedirs("outputs", exist_ok=True)
        model_path = f"outputs/{prefix}.pth"

        if val_recall > best_recall:
            best_recall   = val_recall
            best_epoch    = epoch
            best_loss     = val_loss
            best_acc      = val_acc
            best_precision= val_precision
            best_f1       = val_f1
            best_auc      = val_auc
            best_model_path = model_path

            torch.save(model.state_dict(), model_path)
            my_logger.info(f"New best model saved: {model_path} (recall={val_recall:.3f})")

            cm = confusion_matrix(labels_arr, preds_arr)
            disp = ConfusionMatrixDisplay(cm)
            disp.plot(cmap=plt.cm.Blues)
            cm_path = f"outputs/{prefix}_confmat.png"
            plt.savefig(cm_path); plt.close()
            best_cm_path = cm_path
            my_logger.info(f"Confusion matrix saved: {cm_path}")

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        torch.cuda.empty_cache()
        if epochs_without_improvement >= early_stopping_patience:
            my_logger.info("Early stopping.")
            break

    my_logger.info(f"Training done in {time.time() - start_time:.2f}s")

    final_metrics = {
        k: v for k, v in {
            "best_model_epoch":    best_epoch,
            "best_model_loss":     best_loss,
            "best_model_acc":      best_acc,
            "best_model_recall":   best_recall,
            "best_model_precision":best_precision,
            "best_model_f1":       best_f1,
            "best_model_auc":      best_auc
        }.items() if v is not None
    }
    if final_metrics:
        mlflow.log_metrics(final_metrics, step=best_epoch)
    else:
        my_logger.warning("No final best-model metrics to log (all were None).")

    my_logger.info("Best Model Metrics:")
    for k, v in final_metrics.items():
        my_logger.info(f"  {k} = {v}")

    if best_model_path:
        my_logger.info(f"Best model file: {best_model_path}")
        my_logger.info(f"Best CM file   : {best_cm_path}")


def main():
    my_logger.info(f"Torch version: {torch.__version__}")

    parser = argparse.ArgumentParser(description="Train VGG binary model")
    parser.add_argument("--train_dir",      type=str, required=True)
    parser.add_argument("--val_dir",        type=str, required=True)
    parser.add_argument("--num_epochs",     type=int, default=20)
    parser.add_argument("--batch_size",     type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate",  type=float, default=0.0005,
                        help="Learning rate (default: 0.0005)")
    parser.add_argument("--use_sampler",    action="store_true",
                        help="If set, use weighted sampler on training set")
    args = parser.parse_args()
    my_logger.info(f"Args: {args}")

    load_dotenv()
    sa   = os.getenv("AZURE_STORAGE_ACCOUNT")
    sk   = os.getenv("AZURE_STORAGE_KEY")
    cont = os.getenv("BLOB_CONTAINER")
    my_logger.info("Downloading from blob…")
    download_from_blob(sa, sk, cont, args.train_dir)
    download_from_blob(sa, sk, cont, args.val_dir)

    mlflow.start_run()

    # load datasets
    train_ds = Dataset2DBinary(args.train_dir)
    val_ds   = DatasetSequence2DBinary(dataset_folder=args.val_dir, sequence_length=30)

    # compute train label distribution
    train_labels = train_ds.labels
    cnt = Counter(train_labels)
    my_logger.info(f"Train label distribution: {cnt}")

    # set up sampler or weighting
    if args.use_sampler:
        weights = [1.0 / cnt[int(lbl)] for lbl in train_labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
        class_weights = None
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        # only apply class weights if both classes are present
        if len(cnt) == 2:
            total = sum(cnt.values())
            class_weights = [total / cnt[i] for i in range(2)]
            my_logger.info(f"Using class weights: {class_weights}")
        else:
            class_weights = None
            my_logger.warning(
                f"Found {len(cnt)} class(es) in training data; skipping class weighting."
            )

    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False)

    my_logger.info(f"Validation sequences: {len(val_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(
        train_loader,
        val_loader,
        device=device,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        class_weights=class_weights
    )

    mlflow.end_run()
    my_logger.info("MLflow run ended")


if __name__ == "__main__":
    main()
