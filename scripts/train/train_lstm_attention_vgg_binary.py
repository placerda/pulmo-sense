#!/usr/bin/env python
"""
Train LSTM + Attention model with VGG features for binary classification (NCP vs Normal).

This script:
  - Downloads training and validation data from Azure Blob Storage.
  - Extracts per-slice embeddings via a pretrained VGG16-BN (classifier head removed).
  - Trains an LSTM with temporal attention on fixed-length CT sequences.
  - Logs train/validation metrics (loss, accuracy, recall, precision, F1, AUC) to MLflow.
  - Saves the best model and confusion matrix based on validation recall.
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
import mlflow
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
)

from datasets import DatasetSequence2DBinary
from utils.download import download_from_blob, download_from_blob_with_access_key
from utils.log_config import get_custom_logger

logger = get_custom_logger('train_lstm_attention_vgg_binary')

class VGGFeatureExtractor(nn.Module):
    """VGG16-BN backbone, classifier head removed."""
    def __init__(self, weights_path: str):
        super().__init__()
        backbone = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        embedding_dim = backbone.classifier[6].in_features
        backbone.classifier[6] = nn.Identity()
        state = torch.load(weights_path, map_location='cpu')
        backbone.load_state_dict(state, strict=False)
        self.backbone = backbone
        self.embedding_dim = embedding_dim

    def forward(self, x):
        # x: [B*seq_len, 1, H, W]
        x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
        return self.backbone(x)

class TemporalAttention(nn.Module):
    """Attention over time dimension."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, outputs: torch.Tensor):
        # outputs: [B, seq_len, hidden_dim]
        energies = self.score(outputs).squeeze(-1)      # [B, seq_len]
        weights = torch.softmax(energies, dim=1)        # [B, seq_len]
        context = (outputs * weights.unsqueeze(-1)).sum(dim=1)  # [B, hidden_dim]
        return context

class LSTMAttentionClassifier(nn.Module):
    """LSTM + temporal attention sequence classifier."""
    def __init__(self, input_dim: int, hidden_dim: int=128, num_layers: int=1,
                 num_classes: int=2, dropout: float=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attn = TemporalAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [B, seq_len, input_dim]
        outputs, _ = self.lstm(x)
        context = self.attn(outputs)                  # [B, hidden_dim]
        out = self.fc(self.dropout(context))          # [B, num_classes]
        return out


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Download data
    load_dotenv()
    acct, key, cont = os.getenv('AZURE_STORAGE_ACCOUNT'), os.getenv('AZURE_STORAGE_KEY'), os.getenv('BLOB_CONTAINER')
    logger.info("Downloading datasets...")
    download_from_blob(acct, key, cont, args.train_dir)
    download_from_blob(acct, key, cont, args.val_dir)

    vgg_path = 'models/vgg_binary.pth'
    download_from_blob_with_access_key(args.vgg_model_uri, key, vgg_path)

    # Datasets and loaders
    train_ds = DatasetSequence2DBinary(args.train_dir, args.sequence_length)
    val_ds   = DatasetSequence2DBinary(args.val_dir,   args.sequence_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    logger.info(f"Loaded train seqs={len(train_ds)}, val seqs={len(val_ds)}")

    # Models
    feat_ext = VGGFeatureExtractor(vgg_path).to(device)
    feat_ext.eval()
    for p in feat_ext.parameters(): p.requires_grad=False

    clf = LSTMAttentionClassifier(input_dim=feat_ext.embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clf.parameters(), lr=args.lr, weight_decay=1e-5)

    best_recall, best_epoch = 0, 0
    best_metrics = {}
    best_model_path = best_cm_path = None
    no_improve = 0

    mlflow.start_run()
    start = time.time()
    for epoch in range(1, args.epochs+1):
        logger.info(f"Epoch {epoch}/{args.epochs}")
        # Training
        clf.train()
        train_loss = train_corr = total = 0
        for imgs, labels in train_loader:
            B,S,C,H,W = imgs.shape
            imgs, labels = imgs.to(device), labels.to(device)
            flat = imgs.view(B*S, C, H, W)
            with torch.no_grad(): feats = feat_ext(flat).view(B, S, -1)
            logits = clf(feats)
            loss = criterion(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            train_loss += loss.item()*B
            preds = logits.argmax(1)
            train_corr += preds.eq(labels).sum().item()
            total += B
        train_loss /= total
        train_acc = train_corr/total
        mlflow.log_metric('train_loss', train_loss, step=epoch)
        mlflow.log_metric('train_acc', train_acc, step=epoch)

        # Validation
        clf.eval()
        val_loss = val_corr = val_total = 0
        all_lbl, all_pred, all_prob = [], [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                B,S,C,H,W = imgs.shape
                imgs, labels = imgs.to(device), labels.to(device)
                flat = imgs.view(B*S, C, H, W)
                feats = feat_ext(flat).view(B, S, -1)
                logits = clf(feats)
                loss = criterion(logits, labels)
                val_loss += loss.item()*B
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(1)
                val_corr += preds.eq(labels).sum().item()
                val_total += B
                all_lbl.extend(labels.cpu().numpy())
                all_pred.extend(preds.cpu().numpy())
                all_prob.extend(probs.cpu().numpy())
        val_loss /= val_total
        val_acc = val_corr/val_total
        all_lbl = np.array(all_lbl); all_pred=np.array(all_pred); all_prob=np.vstack(all_prob)
        val_recall = recall_score(all_lbl, all_pred, average='binary', pos_label=1, zero_division=0)
        val_precision = precision_score(all_lbl, all_pred, average='binary', pos_label=1, zero_division=0)
        val_f1 = f1_score(all_lbl, all_pred, average='binary', pos_label=1, zero_division=0)
        try:
            fpr,tpr,_ = roc_curve(all_lbl, all_prob[:,1], pos_label=1); val_auc=auc(fpr,tpr)
        except: val_auc=0.0
        for name,val in [('val_loss',val_loss),('val_acc',val_acc),('val_recall',val_recall),
                         ('val_precision',val_precision),('val_f1',val_f1),('val_auc',val_auc)]:
            mlflow.log_metric(name, val, step=epoch)
        logger.info(f"Val loss={val_loss:.4f}, acc={val_acc:.4f}, recall={val_recall:.2f}, "
                    f"prec={val_precision:.2f}, f1={val_f1:.2f}, auc={val_auc:.2f}")

        # Early stopping & save best
        if val_recall>best_recall:
            best_recall, best_epoch = val_recall, epoch
            best_metrics = {'loss':val_loss,'acc':val_acc,'precision':val_precision,'f1':val_f1,'auc':val_auc}
            prefix=f"attn_lstm_{epoch}ep_{val_recall:.3f}rec"
            os.makedirs('outputs',exist_ok=True)
            best_model_path = f"outputs/{prefix}.pth"; torch.save(clf.state_dict(),best_model_path)
            cm=confusion_matrix(all_lbl,all_pred)
            disp=ConfusionMatrixDisplay(cm); disp.plot(cmap=plt.cm.Blues)
            best_cm_path=f"outputs/{prefix}_cm.png"; plt.savefig(best_cm_path); plt.close()
            no_improve=0
        else:
            no_improve+=1
            if no_improve>=args.patience: break

    elapsed=time.time()-start
    logger.info(f"Finished in {elapsed:.2f}s, best recall={best_recall:.3f} at epoch {best_epoch}")
    mlflow.log_metrics({'best_recall':best_recall,'best_epoch':best_epoch})
    mlflow.end_run()


def main():
    parser=argparse.ArgumentParser("Train LSTM+Attention VGG")
    parser.add_argument('--train_dir',type=str, required=True)
    parser.add_argument('--val_dir',  type=str, required=True)
    parser.add_argument('--vgg_model_uri', type=str, required=True)
    parser.add_argument('--sequence_length', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=3)
    args=parser.parse_args()
    train(args)

if __name__=='__main__': main()
