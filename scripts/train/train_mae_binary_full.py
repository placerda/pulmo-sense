#!/usr/bin/env python
"""
Train MAE-based binary model on full dataset (no validation).

This script downloads the entire training dataset, then trains a pretrained MAE
from timm for binary classification over all data, optionally using a weighted sampler.
"""

import argparse
import os
import time
from collections import Counter

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from dotenv import load_dotenv

from datasets.raster_dataset import Dataset2DBinary
from utils.download import download_from_blob
from utils.log_config import get_custom_logger

# setup logger
my_logger = get_custom_logger('train_mae_binary_no_val')


class MAEModel(nn.Module):
    """
    MAE wrapper for binary classification.
    """
    def __init__(self, model_name: str, num_classes: int = 2):
        super().__init__()
        # this name must match one of timm’s pretrained MAE checkpoints
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    def forward(self, x):
        # replicate single-channel to RGB and resize to 224×224
        x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(
            x, size=(224, 224), mode='bilinear', align_corners=False
        )
        return self.model(x)


def train_model(train_loader, device, num_epochs, learning_rate, model_name):
    start_time = time.time()
    my_logger.info(f"Starting full-dataset MAE training for {model_name} on {device}")

    model = MAEModel(model_name=model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scaler = GradScaler()

    for epoch in range(1, num_epochs + 1):
        my_logger.info(f"=== Epoch {epoch}/{num_epochs} ===")
        model.train()

        running_loss = total_correct = total_samples = 0

        for i, (inputs, _, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * labels.size(0)
            total_correct += preds.eq(labels).sum().item()
            total_samples += labels.size(0)

            if i % 50 == 0:
                batch_acc = preds.eq(labels).float().mean().item()
                my_logger.info(
                    f"[Train] Batch {i}/{len(train_loader)} – "
                    f"Loss={loss.item():.4f}, Acc={batch_acc:.4f}"
                )

        epoch_loss = running_loss / total_samples
        epoch_acc  = 100.0 * total_correct / total_samples
        my_logger.info(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%")

        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        mlflow.log_metric("train_accuracy", epoch_acc, step=epoch)

    elapsed = time.time() - start_time
    my_logger.info(f"Training completed in {elapsed:.2f}s")

    # save final model
    os.makedirs("outputs", exist_ok=True)
    model_path = f"outputs/mae_{model_name}_full_{num_epochs}epochs.pth"
    torch.save(model.state_dict(), model_path)
    my_logger.info(f"Model saved: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train MAE binary model (no validation)")
    parser.add_argument("--train_dir",     type=str, required=True,
                        help="Path to download and read full training dataset")
    parser.add_argument("--num_epochs",    type=int, default=20,
                        help="Number of epochs to train")
    parser.add_argument("--batch_size",    type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0005,
                        help="Learning rate (default: 0.0005)")
    parser.add_argument(
        "--model_name",    type=str,
        default="vit_base_patch16_224.mae",
        help="timm MAE model name (e.g. vit_base_patch16_224.mae)"
    )
    parser.add_argument("--use_sampler",   action="store_true",
                        help="Use weighted sampler based on class frequency")
    args = parser.parse_args()
    my_logger.info(f"Args: {args}")

    # load environment and download data
    load_dotenv()
    account   = os.getenv("AZURE_STORAGE_ACCOUNT")
    key       = os.getenv("AZURE_STORAGE_KEY")
    container = os.getenv("BLOB_CONTAINER")
    my_logger.info("Downloading full training dataset from blob storage…")
    download_from_blob(account, key, container, args.train_dir)

    mlflow.start_run()
    my_logger.info("Loading training dataset…")
    train_ds = Dataset2DBinary(args.train_dir)

    # compute and log class distribution
    labels = train_ds.labels
    cnt = Counter(labels)
    my_logger.info(f"Train label distribution: {cnt}")

    # sampler or shuffle
    if args.use_sampler:
        weights = [1.0 / cnt[int(lbl)] for lbl in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    my_logger.info(f"Training samples: {len(train_ds)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run training
    train_model(
        train_loader,
        device=device,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        model_name=args.model_name
    )

    mlflow.end_run()
    my_logger.info("MLflow run ended")


if __name__ == "__main__":
    main()