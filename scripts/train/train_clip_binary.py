#!/usr/bin/env python
"""
Train CLIP-based binary model

This script uses a pretrained CLIP model (openai/clip-vit-base-patch32) from Hugging Face Transformers
for binary classification. It handles single–channel input by replicating it to 3 channels,
resizes the image to (224,224), and performs training using the binary dataset.
The positive class is assumed to be labeled as 0.
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

from torch.utils.data import DataLoader, Subset
from dotenv import load_dotenv
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

from datasets.ccccii_dataset import CCCCIIDataset2DBinary
from utils.download import download_from_blob
from utils.log_config import get_custom_logger

from transformers import CLIPModel

my_logger = get_custom_logger('train_clip_binary')

class CLIPBinaryModel(nn.Module):
    """
    A binary classifier using the pretrained CLIP image encoder.
    It replicates single-channel input to 3 channels, resizes the image to (224,224),
    passes it through the CLIP vision model, and then applies a classification head.
    """
    def __init__(self, num_classes=2, model_name="openai/clip-vit-base-patch32"):
        super(CLIPBinaryModel, self).__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        # Optionally, freeze the CLIP vision encoder for faster training:
        # for param in self.clip.vision_model.parameters():
        #     param.requires_grad = False
        hidden_size = self.clip.config.vision_hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: [batch_size, 1, 512, 512] => replicate to 3 channels
        x = x.repeat(1, 3, 1, 1)
        # Resize to (224,224)
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # Pass through CLIP's vision model; note that the vision model expects a key "pixel_values"
        vision_outputs = self.clip.vision_model(pixel_values=x)
        pooled_output = vision_outputs.pooler_output  # shape: (batch, hidden_size)
        logits = self.classifier(pooled_output)
        return logits

def train_model(train_loader, val_loader, num_epochs, learning_rate):
    start_time = time.time()
    my_logger.info("Starting CLIP binary training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_logger.info(f"Using device: {device}")

    try:
        model = CLIPBinaryModel(num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        early_stopping_patience = 3
        epochs_without_improvement = 0
        best_recall = 0.0

        for epoch in range(num_epochs):
            my_logger.info(f"=== Epoch {epoch+1}/{num_epochs} ===")
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
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                if i % 50 == 0:
                    batch_acc = (preds == labels).float().mean().item()
                    my_logger.info(f"[Train] Batch {i}/{len(train_loader)}: Loss={loss.item():.4f}, Acc={batch_acc:.4f}")
            
            train_loss = total_loss / total
            train_acc = 100.0 * correct / total

            my_logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%")
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)

            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            all_labels, all_probs = [], []

            with torch.no_grad():
                for j, (inputs, _, labels) in enumerate(val_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * labels.size(0)
                    probs = torch.softmax(outputs, dim=1)
                    preds = probs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            val_loss /= val_total
            val_acc = 100.0 * val_correct / val_total
            all_probs = np.array(all_probs)
            val_preds = np.argmax(all_probs, axis=1)

            val_recall = recall_score(all_labels, val_preds, average='binary', zero_division=0)
            val_precision = precision_score(all_labels, val_preds, average='binary', zero_division=0)
            val_f1 = f1_score(all_labels, val_preds, average='binary', zero_division=0)
            try:
                positive_probs = all_probs[:, 0]
                fpr, tpr, _ = roc_curve(all_labels, positive_probs, pos_label=0)
                val_auc = auc(fpr, tpr)
                my_logger.info("AUC computed using roc_curve with pos_label=0.")
            except ValueError as e:
                my_logger.error("Error computing AUC: %s", e)
                val_auc = 0.0

            my_logger.info(
                f"Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%, Recall={val_recall:.2f}, "
                f"Precision={val_precision:.2f}, F1={val_f1:.2f}, AUC={val_auc:.2f}"
            )

            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            mlflow.log_metric("val_recall", val_recall, step=epoch)
            mlflow.log_metric("val_precision", val_precision, step=epoch)
            mlflow.log_metric("val_f1_score", val_f1, step=epoch)
            mlflow.log_metric("val_auc", val_auc, step=epoch)

            file_prefix = f"clip_binary_{epoch+1}epoch_{learning_rate:.5f}lr_{val_recall:.3f}rec"
            os.makedirs("outputs", exist_ok=True)
            model_path = f"outputs/{file_prefix}.pth"

            if val_recall > best_recall:
                best_recall = val_recall
                epochs_without_improvement = 0
                torch.save(model.state_dict(), model_path)
                my_logger.info(f"New best model saved as {model_path} with recall={val_recall:.3f}")

                cm = confusion_matrix(all_labels, val_preds)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                cm_path = f"outputs/{file_prefix}_confmat.png"
                disp.plot(cmap=plt.cm.Blues)
                plt.savefig(cm_path)
                plt.close()
                my_logger.info(f"Confusion matrix saved as {cm_path}")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                my_logger.info("Early stopping triggered.")
                break

        my_logger.info("Finished Training CLIP binary model.")
        my_logger.info(f"Total training time: {time.time() - start_time:.2f} seconds")
    except Exception as e:
        my_logger.error("Error during training: %s", str(e))
        my_logger.error(traceback.format_exc())

def main():
    my_logger.info(f"Torch version: {torch.__version__}")

    parser = argparse.ArgumentParser(description="Train CLIP binary model")
    parser.add_argument("--num_epochs", type=int, default=20, help="number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="learning rate")
    parser.add_argument("--k", type=int, default=5, help="number of folds for cross-validation")
    parser.add_argument("--i", type=int, default=0, help="current fold index (0-based)")
    parser.add_argument("--dataset", type=str, default="ccccii", help="Dataset name")
    parser.add_argument("--run_cloud", action="store_true", help="Flag to indicate cloud mode")
    parser.add_argument("--max_samples", type=int, default=0, help="Maximum number of samples to use")
    args = parser.parse_args()
    my_logger.info(f"Args: {args}")
    my_logger.info(f"Current Working Directory: {os.getcwd()}")

    mlflow.start_run()

    if not args.run_cloud:
        dataset_folder = f"data/{args.dataset}"
        my_logger.info("Local mode. dataset_folder=%s", dataset_folder)
    else:
        dataset_folder = args.dataset
        load_dotenv()
        storage_account = os.getenv("AZURE_STORAGE_ACCOUNT")
        storage_key = os.getenv("AZURE_STORAGE_KEY")
        container_name = os.getenv("BLOB_CONTAINER")
        my_logger.info("Cloud mode. Downloading data from blob.")
        download_from_blob(storage_account, storage_key, container_name, dataset_folder)

    my_logger.info("Loading dataset")
    my_dataset = CCCCIIDataset2DBinary(dataset_folder, max_samples=args.max_samples)
    my_logger.info(f"Dataset loaded with a maximum of {args.max_samples} samples")

    my_logger.info("Extracting patient IDs and labels")
    patient_ids = [sample[1] for sample in my_dataset]
    labels = [sample[2] for sample in my_dataset]

    random.seed(42)
    combined = list(zip(patient_ids, labels))
    random.shuffle(combined)
    patient_ids, labels = zip(*combined)
    patient_ids = list(patient_ids)
    labels = list(labels)

    if args.i < 0 or args.i >= args.k:
        my_logger.error(f"Invalid fold index 'i': {args.i}. It must be between 0 and {args.k - 1}.")
        raise ValueError(f"Fold index 'i' must be between 0 and {args.k - 1}, but got {args.i}.")

    my_logger.info(f"Performing Stratified K-Fold with {args.k} splits")
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)
    splits = list(skf.split(np.zeros(len(my_dataset)), labels))

    train_idx, val_idx = splits[args.i]
    my_logger.info(f"Train index: {train_idx[:10]}... ({len(train_idx)} samples)")
    my_logger.info(f"Val index: {val_idx[:10]}... ({len(val_idx)} samples)")

    my_logger.info("Creating data loaders")
    train_dataset = Subset(my_dataset, train_idx)
    val_dataset = Subset(my_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    my_logger.info("Data loaders created")

    my_logger.info("Starting model training")
    train_model(train_loader, val_loader, args.num_epochs, args.learning_rate)
    mlflow.end_run()
    my_logger.info("MLflow run ended")

if __name__ == "__main__":
    main()
