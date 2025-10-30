#!/usr/bin/env python
"""
Test pretrained LSTM-VGG binary sequence classifier.

This script:
  - Downloads test data from Azure Blob Storage.
  - Downloads pretrained VGG backbone weights and the trained LSTM classifier.
  - Builds a VGG-based feature extractor and an LSTM classifier.
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

logger = get_custom_logger('test_lstm_vgg_binary')


class VGGFeatureExtractor(nn.Module):
    """VGG16-BN backbone feature extractor (classifier head removed)."""
    def __init__(self, weights_path: str):
        super().__init__()
        backbone = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        self.embedding_dim = backbone.classifier[6].in_features
        backbone.classifier[6] = nn.Identity()
        state = torch.load(weights_path, map_location='cpu')
        backbone.load_state_dict(state, strict=False)
        self.backbone = backbone

    def forward(self, x):
        # x: [B*seq_len, 1, H, W] → convert to 3-ch and resize to 224×224
        x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(x, size=(224, 224),
                                      mode='bilinear', align_corners=False)
        return self.backbone(x)


class LSTMClassifier(nn.Module):
    """LSTM sequence classifier."""
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 1, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [B, S, input_dim]
        outputs, _ = self.lstm(x)
        last = outputs[:, -1, :]             # take last time step
        return self.fc(self.dropout(last))


def test_model(feat_ext, clf, test_loader, criterion, device, outputs_dir):
    logger.info("Starting testing phase…")
    feat_ext.eval()
    clf.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            B, S, C, H, W = imgs.shape
            imgs, labels = imgs.to(device), labels.to(device)

            # flatten to [B*S, C, H, W] to run through VGG
            flat = imgs.view(B * S, C, H, W)

            # extract features in chunks to save memory
            chunk_size = 64
            feats_chunks = []
            for i in range(0, flat.size(0), chunk_size):
                with autocast():
                    feats_chunks.append(feat_ext(flat[i:i + chunk_size]))
            feats = torch.cat(feats_chunks, dim=0).view(B, S, -1)  # [B, S, embed_dim]

            # run classifier
            with autocast():
                logits = clf(feats)
                loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            test_loss += loss.item() * B
            test_correct += (preds == labels).sum().item()
            test_total += B

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # finalize metrics
    test_loss /= test_total
    test_acc = 100.0 * test_correct / test_total

    labels_arr = np.array(all_labels)
    preds_arr = np.array(all_preds)
    probs_arr = np.array(all_probs)

    test_recall = recall_score(labels_arr, preds_arr, average='binary', pos_label=1, zero_division=0)
    test_precision = precision_score(labels_arr, preds_arr, average='binary', pos_label=1, zero_division=0)
    test_f1 = f1_score(labels_arr, preds_arr, average='binary', pos_label=1, zero_division=0)
    try:
        fpr, tpr, _ = roc_curve(labels_arr, probs_arr[:, 1], pos_label=1)
        test_auc = auc(fpr, tpr)
    except Exception as e:
        logger.error("Error computing AUC: %s", e)
        test_auc = 0.0

    # log results
    logger.info(
        f"Test Results: Loss={test_loss:.6f}, Acc={test_acc:.2f}% | "
        f"Rec={test_recall:.2f}, Prec={test_precision:.2f}, "
        f"F1={test_f1:.2f}, AUC={test_auc:.2f}"
    )
    mlflow.log_metrics({
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_recall": test_recall,
        "test_precision": test_precision,
        "test_f1_score": test_f1,
        "test_auc": test_auc
    })

    # confusion matrix
    cm = confusion_matrix(labels_arr, preds_arr)
    disp = ConfusionMatrixDisplay(cm)
    os.makedirs(outputs_dir, exist_ok=True)
    fig = disp.plot()
    cm_path = os.path.join(outputs_dir, "confusion_matrix.png")
    fig.figure_.savefig(cm_path)
    plt.close(fig.figure_)
    logger.info(f"Confusion matrix saved to {cm_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test pretrained LSTM-VGG binary sequence model"
    )
    parser.add_argument(
        "--model_uri", type=str, required=True,
        help="Blob URI to the trained LSTM classifier .pth file"
    )
    parser.add_argument(
        "--vgg_model_uri", type=str, required=True,
        help="Blob URI to the pretrained VGG backbone .pth file"
    )
    parser.add_argument(
        "--test_dir", type=str, required=True,
        help="Local folder for test data (will be downloaded)"
    )
    parser.add_argument(
        "--sequence_length", type=int, default=30,
        help="Number of slices per CT sequence"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for testing"
    )
    args = parser.parse_args()

    load_dotenv()
    acct = os.getenv("AZURE_STORAGE_ACCOUNT")
    key  = os.getenv("AZURE_STORAGE_KEY")
    cont = os.getenv("BLOB_CONTAINER")

    logger.info("Downloading test dataset from Azure Blob Storage…")
    download_from_blob(acct, key, cont, args.test_dir)

    # download VGG weights and LSTM classifier
    vgg_path = "models/vgg_backbone.pth"
    os.makedirs(os.path.dirname(vgg_path), exist_ok=True)
    logger.info("Downloading VGG backbone weights…")
    download_from_blob_with_access_key(args.vgg_model_uri, key, vgg_path)

    lstm_path = "models/lstm_vgg_classifier.pth"
    os.makedirs(os.path.dirname(lstm_path), exist_ok=True)
    logger.info("Downloading trained LSTM classifier…")
    download_from_blob_with_access_key(args.model_uri, key, lstm_path)

    mlflow.start_run()

    # prepare data
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

    # build models
    feat_ext = VGGFeatureExtractor(vgg_path).to(device)
    for p in feat_ext.parameters():
        p.requires_grad = False

    clf = LSTMClassifier(input_dim=feat_ext.embedding_dim).to(device)
    logger.info(f"Loading LSTM classifier state from {lstm_path}")
    clf.load_state_dict(torch.load(lstm_path, map_location=device))

    criterion = nn.CrossEntropyLoss()

    # run evaluation
    test_model(
        feat_ext, clf, test_loader,
        criterion, device,
        outputs_dir="outputs"
    )

    mlflow.end_run()
    logger.info("Testing run finished.")


if __name__ == "__main__":
    main()
