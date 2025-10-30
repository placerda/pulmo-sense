#!/usr/bin/env python
"""
Test VGG binary sequence model with soft voting - LOCAL VERSION
Adaptado para rodar localmente com dataset covid19ctpng_processed
"""

import argparse
import os
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

from datasets import DatasetSequence2DBinary
from utils.log_config import get_custom_logger

logger = get_custom_logger('test_vgg_binary_soft_voting')


def test_model(model, test_loader, device, outputs_dir):
    logger.info("Starting testing phase with soft voting…")
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for seq_slices, seq_labels in test_loader:
            B, L, C, H, W = seq_slices.shape
            seq_slices = seq_slices.to(device)
            labels = seq_labels.to(device)

            # Chunked inference to save memory
            chunk_size = 10
            slice_logits_chunks = []
            for start in range(0, L, chunk_size):
                end = min(start + chunk_size, L)
                flat = seq_slices[:, start:end].reshape(-1, C, H, W)
                # Replicate to 3-channel and resize
                flat = flat.repeat(1, 3, 1, 1)
                flat = nn.functional.interpolate(
                    flat, size=(224, 224), mode='bilinear', align_corners=False
                )
                with autocast():
                    out = model(flat)
                slice_logits_chunks.append(out.view(B, -1, 2))

            slice_logits = torch.cat(slice_logits_chunks, dim=1)  # (B, L, 2)

            # Soft Voting
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

    # Compute metrics
    acc = (preds_arr == labels_arr).mean()
    recall = recall_score(labels_arr, preds_arr, average='binary', pos_label=1, zero_division=0)
    precision = precision_score(labels_arr, preds_arr, average='binary', pos_label=1, zero_division=0)
    f1 = f1_score(labels_arr, preds_arr, average='binary', pos_label=1, zero_division=0)
    try:
        fpr, tpr, _ = roc_curve(labels_arr, probs_arr[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
    except Exception as e:
        logger.error(f"AUC computation error: {e}")
        roc_auc = 0.0

    logger.info(
        f"Test Results (soft voting): Acc={acc:.4f}, Recall={recall:.4f}, "
        f"Precision={precision:.4f}, F1={f1:.4f}, AUC={roc_auc:.4f}"
    )

    # Confusion matrix
    cm = confusion_matrix(labels_arr, preds_arr)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    os.makedirs(outputs_dir, exist_ok=True)
    cm_path = os.path.join(outputs_dir, "confusion_matrix_vgg_soft_voting.png")
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {cm_path}")
    
    return {
        "accuracy": acc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "auc": roc_auc
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test VGG binary sequence model with soft voting"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pretrained .pth model file")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Path to test data directory")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for testing")
    parser.add_argument("--sequence_length", type=int, default=30,
                        help="Number of slices per sequence")
    parser.add_argument("--outputs_dir", type=str, default="outputs",
                        help="Directory to save outputs")
    args = parser.parse_args()

    # Setup dataset
    logger.info(f"Loading test dataset from {args.test_dir}")
    test_ds = DatasetSequence2DBinary(
        dataset_folder=args.test_dir, 
        sequence_length=args.sequence_length
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0
    )
    logger.info(f"Test sequences: {len(test_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load VGG model
    logger.info("Creating VGG16-BN model...")
    model = models.vgg16_bn(weights=None)
    # Modify classifier for binary classification
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 2)
    
    logger.info(f"Loading pretrained weights from {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # Test
    results = test_model(model, test_loader, device, outputs_dir=args.outputs_dir)
    
    logger.info("="*60)
    logger.info("Testing completed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
