#!/usr/bin/env python
"""
Train LSTM+Attention VGG binary model on full dataset without validation or early stopping.
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
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from dotenv import load_dotenv

from datasets import DatasetSequence2DBinary
from utils.download import download_from_blob, download_from_blob_with_access_key
from utils.log_config import get_custom_logger

# setup logger
logger = get_custom_logger('train_lstm_attention_vgg_full')


class VGGFeatureExtractor(nn.Module):
    """VGG16-BN backbone, classifier head removed."""
    def __init__(self, weights_path: str):
        super().__init__()
        backbone = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        emb_dim = backbone.classifier[6].in_features
        backbone.classifier[6] = nn.Identity()
        state = torch.load(weights_path, map_location='cpu')
        backbone.load_state_dict(state, strict=False)
        self.backbone = backbone
        self.embedding_dim = emb_dim

    def forward(self, x):
        # x: [B*seq_len, 1, H, W]
        x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(x, size=(224, 224),
                                      mode='bilinear', align_corners=False)
        return self.backbone(x)


class LSTMAttentionClassifier(nn.Module):
    """LSTM + temporal attention sequence classifier."""
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 1, num_classes: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [B, seq_len, input_dim]
        outputs, _ = self.lstm(x)
        # compute attention weights
        weights = torch.softmax(self.attn(outputs).squeeze(-1), dim=1)  # [B, seq_len]
        context = (outputs * weights.unsqueeze(-1)).sum(dim=1)          # [B, hidden_dim]
        return self.fc(self.dropout(context))


def train_full(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on full dataset, device={device}")

    # download data
    load_dotenv()
    acct = os.getenv('AZURE_STORAGE_ACCOUNT')
    key  = os.getenv('AZURE_STORAGE_KEY')
    cont = os.getenv('BLOB_CONTAINER')
    logger.info("Downloading full training set...")
    download_from_blob(acct, key, cont, args.train_dir)

    # download pretrained VGG features
    vgg_path = 'models/vgg_backbone.pth'
    os.makedirs(os.path.dirname(vgg_path), exist_ok=True)
    logger.info("Downloading VGG weights...")
    download_from_blob_with_access_key(args.vgg_model_uri, key, vgg_path)

    # prepare dataset + loader
    train_ds = DatasetSequence2DBinary(args.train_dir, args.sequence_length)
    labels = train_ds.labels
    cnt = Counter(labels)
    logger.info(f"Train label distribution: {cnt}")

    if args.use_sampler:
        weights = [1.0 / cnt[int(lbl)] for lbl in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                        replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=sampler)
        class_weights = None
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True)
        if len(cnt) == 2:
            total = sum(cnt.values())
            class_weights = [total / cnt[i] for i in range(2)]
            logger.info(f"Using class weights={class_weights}")
        else:
            class_weights = None

    # build models
    feat_ext = VGGFeatureExtractor(vgg_path).to(device)
    feat_ext.eval()
    for p in feat_ext.parameters():
        p.requires_grad = False

    clf = LSTMAttentionClassifier(input_dim=feat_ext.embedding_dim).to(device)
    if class_weights:
        cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(clf.parameters(),
                           lr=args.learning_rate, weight_decay=1e-5)
    scaler = GradScaler()

    # start MLflow
    mlflow.start_run()
    start_time = time.time()

    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"=== Epoch {epoch}/{args.num_epochs} ===")
        clf.train()
        running_loss = running_correct = running_total = 0

        for imgs, labels in train_loader:
            B, S, C, H, W = imgs.shape
            imgs, labels = imgs.to(device), labels.to(device)
            flat = imgs.view(B * S, C, H, W)

            with torch.no_grad():
                feats = feat_ext(flat).view(B, S, -1)

            optimizer.zero_grad()
            with autocast():
                logits = clf(feats)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = logits.argmax(dim=1)
            running_loss    += loss.item() * B
            running_correct += (preds == labels).sum().item()
            running_total   += B

            if running_total % (args.batch_size * 10) == 0:
                logger.info(f"[Train] Batch ~{running_total} samples")

        train_loss = running_loss / running_total
        train_acc  = running_correct / running_total
        logger.info(f"Epoch {epoch}: Loss={train_loss:.6f}, Acc={train_acc:.4f}")

        mlflow.log_metric('train_loss', train_loss, step=epoch)
        mlflow.log_metric('train_acc',   train_acc,   step=epoch)

    elapsed = time.time() - start_time
    logger.info(f"Training finished in {elapsed:.2f}s")

    # save final model
    os.makedirs('outputs', exist_ok=True)
    model_path = f"outputs/lstm_attn_vgg_full_{args.num_epochs}ep.pth"
    torch.save(clf.state_dict(), model_path)
    logger.info(f"Model saved: {model_path}")

    mlflow.end_run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train LSTM+Attention VGG on full dataset (no validation)"
    )
    parser.add_argument('--train_dir',       type=str, required=True,
                        help="Directory for full training sequences")
    parser.add_argument('--vgg_model_uri',   type=str, required=True,
                        help="Blob URI of pretrained VGG weights")
    parser.add_argument('--sequence_length', type=int, required=True,
                        help="Number of slices per CT sequence")
    parser.add_argument('--num_epochs',      type=int, default=20,
                        help="Total epochs to train")
    parser.add_argument('--batch_size',      type=int, default=16)
    parser.add_argument('--learning_rate',   type=float, default=5e-4)
    parser.add_argument('--use_sampler',     action='store_true',
                        help="Whether to use a weighted sampler")
    args = parser.parse_args()

    train_full(args)
