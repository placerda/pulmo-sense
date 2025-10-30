#!/usr/bin/env python
"""
Test Vision Transformer binary sequence model with slice-wise majority voting.

This script:
  - Downloads the test dataset from Azure Blob Storage.
  - Downloads the trained ViT model checkpoint from Azure Blob Storage.
  - Runs sequence-level inference on 30-slice sequences using slice-wise majority voting.
  - Computes accuracy, recall, precision, F1, AUC.
  - Logs metrics to MLflow and saves a confusion matrix plot to outputs/.
"""

import argparse
import os
import mlflow
import torch
import torch.nn as nn
import timm
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from datasets import DatasetSequence2DBinary
from utils.download import download_from_blob, download_from_blob_with_access_key
from utils.log_config import get_custom_logger

# setup logger
my_logger = get_custom_logger('test_vit_binary_seq')


def test_model(model, test_loader, device, outputs_dir):
    my_logger.info("Starting testing phase with slice-wise voting…")
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for seq_slices, seq_labels in test_loader:
            B, L, C, H, W = seq_slices.shape
            seq_slices = seq_slices.to(device)
            labels = seq_labels.to(device)

            # chunked inference to save memory
            chunk_size = 10
            slice_logits_chunks = []
            for start in range(0, L, chunk_size):
                end = min(start + chunk_size, L)
                flat = seq_slices[:, start:end].reshape(-1, C, H, W)
                # replicate to 3-channel and resize
                flat = flat.repeat(1, 3, 1, 1)
                flat = nn.functional.interpolate(
                    flat, size=(224, 224), mode='bilinear', align_corners=False
                )
                with autocast():
                    out = model(flat)
                slice_logits_chunks.append(out.view(B, -1, 2))

            slice_logits = torch.cat(slice_logits_chunks, dim=1)  # (B, L, 2)

                    # ----- Hard Voting -----
            # slice_preds: (B, L) array of per-slice class predictions
            slice_preds = slice_logits.argmax(dim=2)
            # votes: number of positive-slice votes per sequence
            votes = slice_preds.sum(dim=1)
            # preds: final sequence labels via majority vote (hard voting)
            preds = (votes > (L // 2)).long()

            # ----- Soft Voting -----
            # prob_pos: proportion of positive votes as probability for positive class
            prob_pos = votes.float() / L
            probs = torch.stack([1 - prob_pos, prob_pos], dim=1)  # (B, 2)
            slice_preds = slice_logits.argmax(dim=2)            # (B, L)
            votes = slice_preds.sum(dim=1)                      # (B,)
            preds = (votes > (L // 2)).long()                   # (B,)
            prob_pos = votes.float() / L                        # (B,)
            probs = torch.stack([1 - prob_pos, prob_pos], dim=1) # (B, 2)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    labels_arr = np.array(all_labels)
    preds_arr = np.array(all_preds)
    probs_arr = np.array(all_probs)

    # compute metrics
    acc = (preds_arr == labels_arr).mean()
    recall = recall_score(labels_arr, preds_arr, average='binary', pos_label=1, zero_division=0)
    precision = precision_score(labels_arr, preds_arr, average='binary', pos_label=1, zero_division=0)
    f1 = f1_score(labels_arr, preds_arr, average='binary', pos_label=1, zero_division=0)
    try:
        fpr, tpr, _ = roc_curve(labels_arr, probs_arr[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
    except Exception as e:
        my_logger.error(f"AUC computation error: {e}")
        roc_auc = 0.0

    my_logger.info(
        f"Test Results (voting): Acc={acc:.4f}, Recall={recall:.4f}, "
        f"Precision={precision:.4f}, F1={f1:.4f}, AUC={roc_auc:.4f}"
    )

    # log metrics to MLflow
    mlflow.log_metrics({
        "test_accuracy": acc,
        "test_recall": recall,
        "test_precision": precision,
        "test_f1": f1,
        "test_auc": roc_auc
    })

    # confusion matrix
    cm = confusion_matrix(labels_arr, preds_arr)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    os.makedirs(outputs_dir, exist_ok=True)
    cm_path = os.path.join(outputs_dir, "confusion_matrix_voting.png")
    plt.savefig(cm_path)
    plt.close()
    my_logger.info(f"Confusion matrix saved to {cm_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test pretrained ViT binary sequence model with voting"
    )
    parser.add_argument("--model_uri", type=str, required=True,
                        help="URI to the pretrained .pth model file in Blob Storage")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Local folder for test data (will be downloaded)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for testing")
    args = parser.parse_args()

    load_dotenv()
    sa = os.getenv("AZURE_STORAGE_ACCOUNT")
    sk = os.getenv("AZURE_STORAGE_KEY")
    cont = os.getenv("BLOB_CONTAINER")

    my_logger.info("Downloading test dataset from Azure Blob Storage...")
    download_from_blob(sa, sk, cont, args.test_dir)

    # download model artifact
    model_path = 'models/vit_binary.pth'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    my_logger.info("Downloading trained model artifact...")
    download_from_blob_with_access_key(args.model_uri, sk, model_path)

    mlflow.start_run()
    # sequence dataset
    test_ds = DatasetSequence2DBinary(dataset_folder=args.test_dir, sequence_length=30)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    my_logger.info(f"Test sequences: {len(test_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_logger.info(f"Using device: {device}")

    # load model
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
    my_logger.info(f"Loading pretrained model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # test
    test_model(model, test_loader, device, outputs_dir="outputs")

    mlflow.end_run()
    my_logger.info("Testing run finished.")


if __name__ == "__main__":
    main()
