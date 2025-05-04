#!/usr/bin/env python
"""
Train VGG Binary Model on full dataset without validation or early stopping.
"""

import argparse
import os
import time
from collections import Counter

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from dotenv import load_dotenv

from datasets.raster_dataset import Dataset2DBinary
from utils.log_config import get_custom_logger
from utils.download import download_from_blob

# setup logger
my_logger = get_custom_logger('train_vgg_binary_no_val')


class VGG_Net(nn.Module):
    """
    Wrapper around pretrained VGG16-BN for binary classification.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        base = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        in_features = base.classifier[6].in_features
        base.classifier[6] = nn.Linear(in_features, num_classes)
        self.model = base

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        return self.model(x)


def train_model(train_loader, device, num_epochs, learning_rate, class_weights=None):
    start_time = time.time()
    my_logger.info("Starting training on full dataset (no validation)")
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

    for epoch in range(1, num_epochs + 1):
        my_logger.info(f"=== Epoch {epoch}/{num_epochs} ===")
        model.train()
        running_loss = running_correct = running_total = 0

        for i, (inputs, _, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
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
                    f"[Train] Batch {i}/{len(train_loader)} – Loss={loss.item():.6f}, Acc={batch_acc:.4f}"
                )

        train_loss = running_loss / running_total
        train_acc  = 100.0 * running_correct / running_total
        my_logger.info(f"Epoch {epoch}: Loss={train_loss:.6f}, Acc={train_acc:.2f}%")
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)

    elapsed = time.time() - start_time
    my_logger.info(f"Training completed in {elapsed:.2f}s")

    # save final model
    os.makedirs("outputs", exist_ok=True)
    model_path = f"outputs/vgg_full_{num_epochs}epochs.pth"
    torch.save(model.state_dict(), model_path)
    my_logger.info(f"Model saved: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train VGG binary model (no validation)")
    parser.add_argument("--train_dir",      type=str, required=True,
                        help="Folder with training images (full dataset)")
    parser.add_argument("--num_epochs",     type=int, default=20,
                        help="Number of epochs to train")
    parser.add_argument("--batch_size",     type=int, default=16)
    parser.add_argument("--learning_rate",  type=float, default=0.0005,
                        help="Learning rate (default: 0.0005)")
    parser.add_argument("--use_sampler",    action="store_true",
                        help="Use weighted sampler on training set")
    args = parser.parse_args()

    load_dotenv()
    sa   = os.getenv("AZURE_STORAGE_ACCOUNT")
    sk   = os.getenv("AZURE_STORAGE_KEY")
    cont = os.getenv("BLOB_CONTAINER")
    my_logger.info("Downloading full dataset from blob…")
    download_from_blob(sa, sk, cont, args.train_dir)

    mlflow.start_run()

    # load dataset
    train_ds = Dataset2DBinary(args.train_dir)
    train_labels = train_ds.labels
    cnt = Counter(train_labels)
    my_logger.info(f"Train label distribution: {cnt}")

    # sampler or shuffle
    if args.use_sampler:
        weights = [1.0 / cnt[int(lbl)] for lbl in train_labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
        class_weights = None
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        if len(cnt) == 2:
            total = sum(cnt.values())
            class_weights = [total / cnt[i] for i in range(2)]
            my_logger.info(f"Using class weights: {class_weights}")
        else:
            class_weights = None

    my_logger.info(f"Training samples: {len(train_ds)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(
        train_loader,
        device=device,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        class_weights=class_weights
    )

    mlflow.end_run()
    my_logger.info("MLflow run ended")


if __name__ == "__main__":
    main()
