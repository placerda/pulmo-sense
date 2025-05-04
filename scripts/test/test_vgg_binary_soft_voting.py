#!/usr/bin/env python
"""
Test pretrained VGG Binary Sequence Classifier with slice-wise majority voting.

This script:
  - Downloads test data from Azure Blob Storage.
  - Loads a pretrained VGG model (.pth file).
  - Runs evaluation on the test dataset using slice-wise voting.
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
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
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

my_logger = get_custom_logger('test_vgg_binary')

class VGG_Net(nn.Module):
    """
    Wrapper around pretrained VGG16-BN for binary classification.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # if input is single-channel, repeat to 3-channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.model(x)

def test_model(model, test_loader, criterion, device, outputs_dir):
    my_logger.info("Starting testing phase with slice-wise voting…")
    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0

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
                with autocast():
                    out = model(flat)
                # reshape back to (B, chunk_len, num_classes)
                slice_logits_chunks.append(out.view(B, -1, 2))
            slice_logits = torch.cat(slice_logits_chunks, dim=1)  # (B, L, 2)

            # compute a loss on the *mean* logits if you want to log loss
            seq_logits = slice_logits.mean(dim=1)                # (B, 2)
            loss = criterion(seq_logits, labels)
            test_loss += loss.item() * B

            # SLICE-WISE MAJORITY VOTING
            #  1) preds per slice
            slice_preds = slice_logits.argmax(dim=2)             # (B, L)
            #  2) count “positive” votes per sequence
            votes = slice_preds.sum(dim=1)                       # (B,)
            #  3) final sequence prediction
            preds = (votes > (L // 2)).long()                    # (B,)
            #  4) use vote proportion as positive-class “score” for AUC
            prob_pos = votes.float() / L                         # (B,)
            probs = torch.stack([1 - prob_pos, prob_pos], dim=1) # (B, 2)

            # accumulate metrics
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
        my_logger.error("Error computing AUC: %s", e)
        test_auc = 0.0

    # log results
    my_logger.info(
        f"Test Results (voting): Loss={test_loss:.6f}, Acc={test_acc:.2f}% | "
        f"Recall={test_recall:.2f}, Precision={test_precision:.2f}, "
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
    disp.plot(cmap=plt.cm.Blues)
    os.makedirs(outputs_dir, exist_ok=True)
    cm_path = os.path.join(outputs_dir, "confusion_matrix_voting.png")
    plt.savefig(cm_path)
    plt.close()
    my_logger.info(f"Confusion matrix saved to {cm_path}")

def main():
    parser = argparse.ArgumentParser(description="Test pretrained VGG binary sequence model with voting")
    parser.add_argument("--model_uri", type=str, required=True,
                        help="URI to the pretrained .pth model file")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Local folder for test data (will be downloaded)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for testing")
    args = parser.parse_args()

    load_dotenv()
    sa   = os.getenv("AZURE_STORAGE_ACCOUNT")
    sk   = os.getenv("AZURE_STORAGE_KEY")
    cont = os.getenv("BLOB_CONTAINER")

    my_logger.info("Downloading test dataset from Azure Blob Storage...")
    download_from_blob(sa, sk, cont, args.test_dir)

    # download model artifact
    vgg_path = 'models/vgg_binary.pth'
    download_from_blob_with_access_key(args.model_uri, sk, vgg_path)

    mlflow.start_run()
    test_ds = DatasetSequence2DBinary(dataset_folder=args.test_dir, sequence_length=30)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_logger.info(f"Using device: {device}")

    model = VGG_Net(num_classes=2).to(device)
    my_logger.info(f"Loading pretrained model from {vgg_path}")
    model.load_state_dict(torch.load(vgg_path, map_location=device))

    criterion = nn.CrossEntropyLoss()
    test_model(model, test_loader, criterion, device, outputs_dir="outputs")

    mlflow.end_run()
    my_logger.info("Testing run finished.")

if __name__ == "__main__":
    main()
