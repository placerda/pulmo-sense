#!/usr/bin/env python
"""
Test pretrained LSTM+Attention VGG binary sequence classifier.

This script:
  - Downloads test data from Azure Blob Storage.
  - Downloads pretrained VGG backbone weights and the trained attention‐LSTM classifier.
  - Builds a VGG‐based feature extractor and an LSTM+Attention classifier.
  - Runs evaluation on the test dataset.
  - Calculates accuracy, recall, precision, F1, AUC.
  - Saves a confusion matrix plot to outputs/.
"""

import argparse
import os

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from dotenv import load_dotenv
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

from datasets import DatasetSequence2DBinary
from utils.download import download_from_blob, download_from_blob_with_access_key
from utils.log_config import get_custom_logger

logger = get_custom_logger('test_lstm_attn_vgg_binary')


class VGGFeatureExtractor(nn.Module):
    """VGG16-BN backbone feature extractor (classifier head removed)."""
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
        # x: [B*seq_len, 1, H, W] → to 3-channel & resize
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
        weights = torch.softmax(self.attn(outputs).squeeze(-1), dim=1)  # [B, seq_len]
        context = (outputs * weights.unsqueeze(-1)).sum(dim=1)          # [B, hidden_dim]
        return self.fc(self.dropout(context))


def test_model(feat_ext, clf, loader, criterion, device, outputs_dir):
    logger.info("Starting testing phase…")
    feat_ext.eval()
    clf.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            B, S, C, H, W = imgs.shape
            imgs, labels = imgs.to(device), labels.to(device)

            flat = imgs.view(B * S, C, H, W)
            feats_chunks = []
            chunk_size = 64
            for i in range(0, flat.size(0), chunk_size):
                with autocast():
                    feats_chunks.append(feat_ext(flat[i:i+chunk_size]))
            feats = torch.cat(feats_chunks, dim=0).view(B, S, -1)

            with autocast():
                logits = clf(feats)
                loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * B
            correct += (preds == labels).sum().item()
            total += B

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    test_loss = total_loss / total
    test_acc  = 100.0 * correct / total
    labels_arr = np.array(all_labels)
    preds_arr  = np.array(all_preds)
    probs_arr  = np.array(all_probs)

    recall    = recall_score(labels_arr, preds_arr, average='binary', pos_label=1, zero_division=0)
    precision = precision_score(labels_arr, preds_arr, average='binary', pos_label=1, zero_division=0)
    f1        = f1_score(labels_arr, preds_arr, average='binary', pos_label=1, zero_division=0)
    try:
        fpr, tpr, _ = roc_curve(labels_arr, probs_arr[:,1], pos_label=1)
        auc_score    = auc(fpr, tpr)
    except:
        auc_score = 0.0

    logger.info(
        f"Test Results: Loss={test_loss:.6f}, Acc={test_acc:.2f}% | "
        f"Rec={recall:.2f}, Prec={precision:.2f}, F1={f1:.2f}, AUC={auc_score:.2f}"
    )
    mlflow.log_metrics({
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_recall": recall,
        "test_precision": precision,
        "test_f1_score": f1,
        "test_auc": auc_score
    })

    cm = confusion_matrix(labels_arr, preds_arr)
    disp = ConfusionMatrixDisplay(cm)
    os.makedirs(outputs_dir, exist_ok=True)
    fig = disp.plot()
    path = os.path.join(outputs_dir, "confusion_matrix.png")
    fig.figure_.savefig(path)
    plt.close(fig.figure_)
    logger.info(f"Confusion matrix saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test pretrained LSTM+Attention VGG sequence model"
    )
    parser.add_argument("--model_uri",       type=str, required=True,
                        help="Blob URI to the trained LSTM+Attention .pth file")
    parser.add_argument("--vgg_model_uri",   type=str, required=True,
                        help="Blob URI to the pretrained VGG backbone .pth file")
    parser.add_argument("--test_dir",        type=str, required=True,
                        help="Local folder for test data (will be downloaded)")
    parser.add_argument("--sequence_length", type=int, default=30,
                        help="Number of slices per CT sequence")
    parser.add_argument("--batch_size",      type=int, default=16,
                        help="Batch size for testing")
    args = parser.parse_args()

    load_dotenv()
    acct = os.getenv("AZURE_STORAGE_ACCOUNT")
    key  = os.getenv("AZURE_STORAGE_KEY")
    cont = os.getenv("BLOB_CONTAINER")

    logger.info("Downloading test dataset from Azure Blob Storage…")
    download_from_blob(acct, key, cont, args.test_dir)

    # Download model artifacts
    vgg_path  = "models/vgg_backbone.pth"
    lstm_path = "models/lstm_attn_vgg_classifier.pth"
    os.makedirs(os.path.dirname(vgg_path),  exist_ok=True)
    os.makedirs(os.path.dirname(lstm_path), exist_ok=True)

    logger.info("Downloading VGG backbone weights…")
    download_from_blob_with_access_key(args.vgg_model_uri, key, vgg_path)

    logger.info("Downloading LSTM+Attention classifier…")
    download_from_blob_with_access_key(args.model_uri, key, lstm_path)

    mlflow.start_run()

    test_ds = DatasetSequence2DBinary(
        dataset_folder=args.test_dir,
        sequence_length=args.sequence_length
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    feat_ext = VGGFeatureExtractor(vgg_path).to(device)
    for p in feat_ext.parameters():
        p.requires_grad = False

    clf = LSTMAttentionClassifier(input_dim=feat_ext.embedding_dim).to(device)
    logger.info(f"Loading classifier state from {lstm_path}")
    clf.load_state_dict(torch.load(lstm_path, map_location=device))

    criterion = nn.CrossEntropyLoss()

    test_model(
        feat_ext,
        clf,
        test_loader,
        criterion,
        device,
        outputs_dir="outputs"
    )

    mlflow.end_run()
    logger.info("Testing run finished.")


if __name__ == "__main__":
    main()
