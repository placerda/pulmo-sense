#!/usr/bin/env python3
"""
Train TimeSformer Binary Model on full dataset without validation or early stopping.
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from dotenv import load_dotenv
from transformers import (
    TimesformerConfig,
    TimesformerForVideoClassification,
    AutoImageProcessor
)
from PIL import Image

from datasets.raster_dataset import DatasetSequence2DBinary
from utils.download import download_from_blob
from utils.log_config import get_custom_logger

logger = get_custom_logger('train_timesformer_binary_no_val')


def train_model(train_loader, device, args, processor, model, optimizer, criterion):
    scaler = GradScaler()
    start_time = time.time()
    logger.info("Starting TimeSformer binary training (no validation)")
    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"=== Epoch {epoch}/{args.num_epochs} ===")
        model.train()
        running_loss = running_correct = running_total = 0

        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            B, L, C, H, W = seq.shape

            # Build uint8 RGB frames
            frames = (
                seq.cpu()
                   .repeat(1, 1, 3, 1, 1)
                   .permute(0, 1, 3, 4, 2)
                   .numpy()
                   .clip(0, 255)
                   .astype(np.uint8)
            )
            videos = [
                [Image.fromarray(frames[b, i]) for i in range(L)]
                for b in range(B)
            ]
            inputs = processor(videos, return_tensors="pt").to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            logits = outputs.logits
            running_loss += loss.item() * B
            preds = logits.argmax(dim=-1)
            running_correct += (preds == labels).sum().item()
            running_total += B

        avg_loss = running_loss / running_total
        acc = 100.0 * running_correct / running_total
        logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")
        mlflow.log_metric('train_loss', avg_loss, step=epoch)
        mlflow.log_metric('train_accuracy', acc, step=epoch)

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.1f}s")

    # Save final model + processor
    output_dir = "outputs/timesformer_full"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info(f"Model and processor saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train TimeSformer binary model (no validation)"
    )
    parser.add_argument('--train_dir',         required=True)
    parser.add_argument('--sequence_length',   type=int,   default=30)
    parser.add_argument('--batch_size',        type=int,   default=1)
    parser.add_argument('--num_epochs',        type=int,   default=20)
    parser.add_argument('--learning_rate',     type=float, default=5e-5)
    parser.add_argument('--use_sampler',       action='store_true')
    parser.add_argument(
        '--model_name_or_path',
        default='facebook/timesformer-base-finetuned-k400'
    )
    args = parser.parse_args()

    load_dotenv()
    sa, sk, cont = (
        os.getenv('AZURE_STORAGE_ACCOUNT'),
        os.getenv('AZURE_STORAGE_KEY'),
        os.getenv('BLOB_CONTAINER')
    )

    logger.info("Downloading training data from Azure Blob Storage…")
    download_from_blob(sa, sk, cont, args.train_dir)

    # Build dataset and loader
    train_ds = DatasetSequence2DBinary(
        dataset_folder=args.train_dir,
        sequence_length=args.sequence_length
    )

    labels_list = getattr(train_ds, 'labels', None)
    if args.use_sampler and labels_list is not None and len(set(labels_list)) == 2:
        cnt = Counter(labels_list)
        weights = [1.0 / cnt[int(lbl)] for lbl in labels_list]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=0
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model + processor
    config = TimesformerConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=2,
        num_frames=args.sequence_length
    )
    config.attention_type = "joint_space_time"
    model = TimesformerForVideoClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True
    ).to(device)

    # Freeze backbone, train only classifier head
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    model.gradient_checkpointing_enable()

    processor = AutoImageProcessor.from_pretrained(args.model_name_or_path)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-5
    )
    criterion = nn.CrossEntropyLoss()

    mlflow.start_run()
    train_model(
        train_loader,
        device,
        args,
        processor,
        model,
        optimizer,
        criterion
    )
    mlflow.end_run()


if __name__ == "__main__":
    main()
