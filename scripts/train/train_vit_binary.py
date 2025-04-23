#!/usr/bin/env python
"""
Train Vision Transformer binary model

This script trains a binary Vision Transformer (ViT) model using pretrained weights from timm. 
It handles single-channel input images by replicating them to 3 channels, resizing them, 
and performing binary classification (positive class labeled as 0).
"""

import argparse
import os
import random
import time
import traceback

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import timm

from torch.utils.data import DataLoader, Subset
from dotenv import load_dotenv
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

from datasets.raster_dataset import Dataset2DBinary
from utils.download import download_from_blob
from utils.log_config import get_custom_logger

my_logger = get_custom_logger('train_vit_binary')

class ViTModel(nn.Module):
    """
    Vision Transformer wrapper for binary classification.
    Handles single-channel input by replicating to 3 channels.
    """
    def __init__(self, num_classes=2, model_name='vit_base_patch16_224'):
        super(ViTModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)  # From (1,512,512) to (3,512,512)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.model(x)

def train_model(train_loader, val_loader, num_epochs, learning_rate):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_logger.info(f"Starting ViT binary training on device: {device}")

    model = ViTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    early_stopping_patience = 3
    epochs_without_improvement = 0
    best_recall = 0.0

    try:
        for epoch in range(num_epochs):
            my_logger.info(f"=== Epoch {epoch+1}/{num_epochs} ===")
            
            # Training
            model.train()
            total_loss, correct, total = 0, 0, 0
            for i, (inputs, _, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

                if i % 50 == 0:
                    batch_acc = (outputs.argmax(dim=1) == labels).float().mean().item()
                    my_logger.info(f"[Train] Batch {i}/{len(train_loader)}: Loss={loss.item():.4f}, Acc={batch_acc:.4f}")

            train_loss = total_loss / total
            train_acc = 100.0 * correct / total
            mlflow.log_metrics({"train_loss": train_loss, "train_accuracy": train_acc}, step=epoch)
            my_logger.info(f"Epoch {epoch+1} Train Loss={train_loss:.4f}, Acc={train_acc:.2f}%")

            # Validation
            model.eval()
            val_loss, correct, total = 0, 0, 0
            all_labels, all_probs = [], []
            with torch.no_grad():
                for inputs, _, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item() * labels.size(0)
                    probs = torch.softmax(outputs, dim=1)
                    preds = probs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            val_loss /= total
            val_acc = 100.0 * correct / total
            val_preds = np.argmax(all_probs, axis=1)
            val_recall = recall_score(all_labels, val_preds, average='binary', zero_division=0)
            val_precision = precision_score(all_labels, val_preds, average='binary', zero_division=0)
            val_f1 = f1_score(all_labels, val_preds, average='binary', zero_division=0)
            fpr, tpr, _ = roc_curve(all_labels, np.array(all_probs)[:, 0], pos_label=0)
            val_auc = auc(fpr, tpr)

            mlflow.log_metrics({
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_recall": val_recall,
                "val_precision": val_precision,
                "val_f1_score": val_f1,
                "val_auc": val_auc
            }, step=epoch)

            my_logger.info(
                f"Epoch {epoch+1} Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%, "
                f"Recall={val_recall:.2f}, Precision={val_precision:.2f}, F1={val_f1:.2f}, AUC={val_auc:.2f}"
            )

            if val_recall > best_recall:
                best_recall = val_recall
                epochs_without_improvement = 0
                os.makedirs("outputs", exist_ok=True)
                file_prefix = f"vit_binary_{epoch+1}epoch_{learning_rate:.5f}lr_{val_recall:.3f}rec"
                model_path = f"outputs/{file_prefix}.pth"
                torch.save(model.state_dict(), model_path)

                cm = confusion_matrix(all_labels, val_preds)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap=plt.cm.Blues)
                plt.savefig(f"outputs/{file_prefix}_confmat.png")
                plt.close()
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                my_logger.info("Early stopping triggered.")
                break

        my_logger.info(f"Training completed in {(time.time() - start_time):.2f}s")

    except Exception as e:
        my_logger.error(f"Error during training: {e}")
        my_logger.error(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(description="Train ViT binary model")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--i", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="ccccii")
    parser.add_argument("--run_cloud", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    mlflow.start_run()
    dataset_folder = f"data/{args.dataset}" if not args.run_cloud else args.dataset
    if args.run_cloud:
        load_dotenv()
        download_from_blob(os.getenv("AZURE_STORAGE_ACCOUNT"), os.getenv("AZURE_STORAGE_KEY"), os.getenv("BLOB_CONTAINER"), dataset_folder)

    dataset = Dataset2DBinary(dataset_folder, max_samples=args.max_samples)

    patient_ids = [pid.item() for pid in dataset.patient_ids]
    labels = [lbl.item() for lbl in dataset.labels]

    patient_to_label = dict(zip(patient_ids, labels))
    unique_patients = list(patient_to_label.keys())
    patient_labels = list(patient_to_label.values())

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)
    train_idx_pat, val_idx_pat = list(skf.split(unique_patients, patient_labels))[args.i]
    train_patients = set(np.array(unique_patients)[train_idx_pat])

    train_idx = [i for i, pid in enumerate(patient_ids) if pid in train_patients]
    val_idx = [i for i in range(len(dataset)) if i not in train_idx]

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False)

    train_model(train_loader, val_loader, args.num_epochs, args.learning_rate)
    mlflow.end_run()

if __name__ == "__main__":
    main()
