#!/usr/bin/env python
"""
Train CLIP-based binary classification model using pretrained HuggingFace CLIP vision encoder.
Handles grayscale images by replicating to RGB, resized to (224,224).
Ensures patient-level stratification for training and validation splits.
"""

import argparse
import os
import time
import traceback
import random

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from datasets.raster_dataset import Dataset2DBinary
from utils.download import download_from_blob
from utils.log_config import get_custom_logger
from transformers import CLIPModel

my_logger = get_custom_logger('train_clip_binary')

class CLIPBinaryModel(nn.Module):
    def __init__(self, num_classes=2, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        hidden_size = self.clip.config.vision_config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        vision_outputs = self.clip.vision_model(pixel_values=x)
        pooled_output = vision_outputs.pooler_output
        return self.classifier(pooled_output)

def train_model(train_loader, val_loader, epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_logger.info(f"Training on device: {device}")

    model = CLIPBinaryModel(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_recall, patience, epochs_no_improve = 0.0, 3, 0
    start_time = time.time()

    try:
        for epoch in range(epochs):
            model.train()
            train_loss, correct, total = 0, 0, 0
            my_logger.info(f"Epoch {epoch+1}/{epochs}")

            for inputs, _, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * labels.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

            train_loss /= total
            train_acc = correct / total
            mlflow.log_metrics({"train_loss": train_loss, "train_accuracy": train_acc}, step=epoch)
            my_logger.info(f"Train loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

            model.eval()
            val_loss, correct, total, all_labels, all_probs = 0, 0, 0, [], []

            with torch.no_grad():
                for inputs, _, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item() * labels.size(0)
                    probs = torch.softmax(outputs, dim=1)
                    preds = probs.argmax(1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            val_loss /= total
            val_acc = correct / total
            val_preds = np.argmax(all_probs, axis=1)
            val_recall = recall_score(all_labels, val_preds, zero_division=0)
            val_precision = precision_score(all_labels, val_preds, zero_division=0)
            val_f1 = f1_score(all_labels, val_preds, zero_division=0)
            val_auc = auc(*roc_curve(all_labels, np.array(all_probs)[:,0], pos_label=0)[:2])

            mlflow.log_metrics({
                "val_loss": val_loss, "val_accuracy": val_acc,
                "val_recall": val_recall, "val_precision": val_precision,
                "val_f1": val_f1, "val_auc": val_auc
            }, step=epoch)

            if val_recall > best_recall:
                best_recall, epochs_no_improve = val_recall, 0
                prefix = f"outputs/clip_binary_{epoch+1}_{lr:.5f}_{val_recall:.3f}"
                os.makedirs("outputs", exist_ok=True)
                torch.save(model.state_dict(), f"{prefix}.pth")
                cm = confusion_matrix(all_labels, val_preds)
                ConfusionMatrixDisplay(cm).plot(cmap='Blues')
                plt.savefig(f"{prefix}_cm.png")
                plt.close()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    my_logger.info("Early stopping triggered.")
                    break

        my_logger.info(f"Training completed in {time.time()-start_time:.2f}s")

    except Exception as e:
        my_logger.error(f"Training error: {e}")
        my_logger.error(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--i", type=int, default=0)
    parser.add_argument('--dataset', default='ccccii')
    parser.add_argument('--run_cloud', action='store_true')
    parser.add_argument('--max_samples', type=int, default=0)
    args = parser.parse_args()

    mlflow.start_run()
    dataset_folder = f"data/{args.dataset}"
    if args.run_cloud:
        load_dotenv()
        download_from_blob(
            os.getenv('AZURE_STORAGE_ACCOUNT'),
            os.getenv('AZURE_STORAGE_KEY'),
            os.getenv('BLOB_CONTAINER'),
            dataset_folder
        )

    dataset = Dataset2DBinary(dataset_folder, max_samples=args.max_samples)

    patient_ids = [int(sample[1]) for sample in dataset]
    labels = [int(sample[2]) for sample in dataset]

    patient_label_dict = {}
    for pid, lbl in zip(patient_ids, labels):
        patient_label_dict[pid] = lbl  # Assumes one label per patient

    unique_patients = list(patient_label_dict.keys())
    patient_labels = [patient_label_dict[pid] for pid in unique_patients]

    if args.i < 0 or args.i >= args.k:
        raise ValueError(f"Fold 'i' must be between 0 and {args.k - 1}.")

    skf = StratifiedKFold(args.k, shuffle=True, random_state=42)
    train_patient_idx, val_patient_idx = list(skf.split(unique_patients, patient_labels))[args.i]

    train_patients = {unique_patients[idx] for idx in train_patient_idx}
    val_patients = {unique_patients[idx] for idx in val_patient_idx}

    train_idx = [i for i, pid in enumerate(patient_ids) if pid in train_patients]
    val_idx = [i for i, pid in enumerate(patient_ids) if pid in val_patients]

    my_logger.info(f"Training samples: {len(train_idx)}, Validation samples: {len(val_idx)}")

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False)

    train_model(train_loader, val_loader, args.num_epochs, args.learning_rate)
    mlflow.end_run()

if __name__ == "__main__":
    main()
