#!/usr/bin/env python
"""
Test LSTM+Attention VGG binary sequence classifier - LOCAL VERSION
Adaptado para rodar localmente com dataset covid19ctpng_processed
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

from datasets import DatasetSequence2DBinary
from utils.log_config import get_custom_logger

logger = get_custom_logger('test_lstm_attn_vgg_binary')


class VGGFeatureExtractor(nn.Module):
    """VGG16-BN backbone feature extractor (classifier head removed)."""
    def __init__(self, weights_path: str):
        super().__init__()
        backbone = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        emb_dim = backbone.classifier[6].in_features
        backbone.classifier[6] = nn.Identity()
        
        # Load pretrained weights
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
        f"Rec={recall:.4f}, Prec={precision:.4f}, F1={f1:.4f}, AUC={auc_score:.4f}"
    )

    # Confusion matrix
    cm = confusion_matrix(labels_arr, preds_arr)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    os.makedirs(outputs_dir, exist_ok=True)
    cm_path = os.path.join(outputs_dir, "confusion_matrix_lstm_attn_vgg.png")
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {cm_path}")
    
    return {
        "loss": test_loss,
        "accuracy": test_acc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "auc": auc_score
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test LSTM+Attention VGG binary sequence classifier"
    )
    parser.add_argument("--vgg_model_path", type=str, required=True,
                        help="Path to VGG backbone .pth file")
    parser.add_argument("--lstm_model_path", type=str, required=True,
                        help="Path to LSTM+Attention classifier .pth file")
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

    # Load VGG feature extractor
    logger.info(f"Loading VGG feature extractor from {args.vgg_model_path}")
    feat_ext = VGGFeatureExtractor(args.vgg_model_path)
    feat_ext.to(device)

    # Load LSTM+Attention classifier
    logger.info(f"Loading LSTM+Attention classifier from {args.lstm_model_path}")
    clf = LSTMAttentionClassifier(
        input_dim=feat_ext.embedding_dim,
        hidden_dim=128,
        num_layers=1,
        num_classes=2,
        dropout=0.5
    )
    clf.load_state_dict(torch.load(args.lstm_model_path, map_location=device))
    clf.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Test
    results = test_model(feat_ext, clf, test_loader, criterion, device, 
                        outputs_dir=args.outputs_dir)
    
    logger.info("="*60)
    logger.info("Testing completed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
