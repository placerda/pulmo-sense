#!/usr/bin/env python
"""
Train LSTM + Attention model with VGG features for binary classification.

This script uses a pretrained VGG model as a frozen feature extractor.
Then it trains an LSTM with temporal attention for binary classification (NCP vs Normal).
"""

import argparse
import os
import time
import traceback

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from dotenv import load_dotenv

from datasets import DatasetSequence2DBinary
from utils.download import download_from_blob, download_from_blob_with_access_key
from utils.log_config import get_custom_logger

my_logger = get_custom_logger('train_lstm_attention_vgg_binary')

class VGGFeatureExtractor(nn.Module):
    def __init__(self, vgg_weights_path):
        super().__init__()
        self.vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        in_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Identity()
        state_dict = torch.load(vgg_weights_path, map_location='cpu')
        self.vgg.load_state_dict(state_dict, strict=False)
        self.embedding_dim = in_features

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.vgg(x)

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs):
        energies = self.attn_score(lstm_outputs).squeeze(-1)
        attn_weights = torch.softmax(energies, dim=1)
        context = (lstm_outputs * attn_weights.unsqueeze(-1)).sum(dim=1)
        return context, attn_weights

class LSTM_AttnNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = TemporalAttention(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        return self.fc(self.dropout(context)), attn_weights

def train_model(train_loader, val_loader, epochs, lr, vgg_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_logger.info(f"Training on {device}")

    feature_extractor = VGGFeatureExtractor(vgg_path).to(device).eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False

    model = LSTM_AttnNet(
        input_size=feature_extractor.embedding_dim,
        hidden_size=128,
        num_layers=1,
        num_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_recall, patience, epochs_no_improve = 0.0, 3, 0
    start_time = time.time()

    try:
        for epoch in range(epochs):
            model.train()
            train_loss, correct, total = 0, 0, 0
            my_logger.info(f"Epoch {epoch+1}/{epochs}")

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                bsz, seq_len, ch, h, w = inputs.shape
                inputs_flat = inputs.view(bsz * seq_len, ch, h, w)
                with torch.no_grad():
                    feats = feature_extractor(inputs_flat).view(bsz, seq_len, -1)

                outputs, _ = model(feats)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * bsz
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

            train_loss /= total
            train_acc = correct / total
            mlflow.log_metrics({"train_loss": train_loss, "train_accuracy": train_acc}, step=epoch)
            my_logger.info(f"Train loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

            model.eval()
            val_loss, correct, total, all_labels, all_probs = 0, 0, 0, [], []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    bsz, seq_len, ch, h, w = inputs.shape
                    feats = feature_extractor(inputs.view(bsz * seq_len, ch, h, w)).view(bsz, seq_len, -1)
                    outputs, _ = model(feats)
                    val_loss += criterion(outputs, labels).item() * bsz
                    probs = torch.softmax(outputs, dim=1)
                    correct += (probs.argmax(1) == labels).sum().item()
                    total += labels.size(0)
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            val_loss /= total
            val_acc = correct / total
            val_preds = np.argmax(all_probs, axis=1)
            val_recall = recall_score(all_labels, val_preds, zero_division=0)
            val_precision = precision_score(all_labels, val_preds, zero_division=0)
            val_f1 = f1_score(all_labels, val_preds, zero_division=0)
            val_auc = auc(*roc_curve(all_labels, np.array(all_probs)[:,0], pos_label=0)[:2])

            mlflow.log_metrics({
                "val_loss": val_loss, "val_accuracy": val_acc,
                "val_recall": val_recall, "val_precision": val_precision,
                "val_f1": val_f1, "val_auc": val_auc
            }, step=epoch)

            if val_recall > best_recall:
                best_recall, epochs_no_improve = val_recall, 0
                prefix = f"outputs/attn_lstm_{epoch+1}_{lr:.5f}_{val_recall:.3f}"
                os.makedirs("outputs", exist_ok=True)
                torch.save(model.state_dict(), f"{prefix}.pth")
                cm = confusion_matrix(all_labels, val_preds)
                ConfusionMatrixDisplay(cm).plot(cmap='Blues')
                plt.savefig(f"{prefix}_cm.png")
                plt.close()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    my_logger.info("Early stopping triggered.")
                    break

        my_logger.info(f"Training completed in {time.time()-start_time:.2f}s")

    except Exception as e:
        my_logger.error(f"Training error: {e}")
        my_logger_
