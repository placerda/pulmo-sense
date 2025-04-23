#!/usr/bin/env python
"""
Train LSTM with VGG features for binary classification (NCP vs Normal).

This script extracts features from CT slices using a pretrained VGG model, 
and trains an LSTM classifier on sequences of these features.
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
from utils.log_config import get_custom_logger
from utils.download import download_from_blob, download_from_blob_with_access_key

my_logger = get_custom_logger('train_lstm_vgg_binary')

class VGGFeatureExtractor(nn.Module):
    """
    VGG feature extractor without classification head.
    """
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

class LSTM_Net(nn.Module):
    """
    Simple LSTM model for sequence classification.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

def train_model(train_loader, val_loader, epochs, lr, vgg_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_logger.info(f"Training LSTM-VGG binary model on {device}")

    feature_extractor = VGGFeatureExtractor(vgg_weights_path=vgg_path).to(device)
    feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False

    lstm_model = LSTM_Net(
        input_size=feature_extractor.embedding_dim,
        hidden_size=128,
        num_layers=1,
        num_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=lr, weight_decay=1e-5)

    best_recall = 0.0
    patience = 3
    epochs_no_improve = 0

    try:
        for epoch in range(epochs):
            my_logger.info(f"Epoch {epoch+1}/{epochs}")
            lstm_model.train()
            train_loss, correct, total = 0, 0, 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                bsz, seq_len, ch, h, w = inputs.size()
                inputs = inputs.view(bsz * seq_len, ch, h, w)
                with torch.no_grad():
                    feats = feature_extractor(inputs).view(bsz, seq_len, -1)

                outputs = lstm_model(feats)
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
            my_logger.info(f"Train loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

            lstm_model.eval()
            val_loss, correct, total = 0, 0, 0
            all_labels, all_probs = [], []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    bsz, seq_len, ch, h, w = inputs.size()
                    inputs = inputs.view(bsz * seq_len, ch, h, w)
                    feats = feature_extractor(inputs).view(bsz, seq_len, -1)
                    outputs = lstm_model(feats)
                    val_loss += criterion(outputs, labels).item() * bsz
                    probs = torch.softmax(outputs, dim=1)
                    preds = probs.argmax(1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            val_loss /= total
            val_acc = correct / total
            val_preds = np.argmax(all_probs, axis=1)
            val_recall = recall_score(all_labels, val_preds, zero_division=0)
            val_precision = precision_score(all_labels, val_preds, zero_division=0)
            val_f1 = f1_score(all_labels, val_preds, zero_division=0)
            fpr, tpr, _ = roc_curve(all_labels, np.array(all_probs)[:,0], pos_label=0)
            val_auc = auc(fpr, tpr)

            mlflow.log_metrics({
                "val_loss": val_loss, "val_accuracy": val_acc, "val_recall": val_recall,
                "val_precision": val_precision, "val_f1": val_f1, "val_auc": val_auc
            }, step=epoch)

            if val_recall > best_recall:
                best_recall = val_recall
                epochs_no_improve = 0
                os.makedirs("outputs", exist_ok=True)
                file_prefix = f"lstm_vgg_epoch{epoch+1}_{lr:.5f}lr_{val_recall:.3f}rec"
                torch.save(lstm_model.state_dict(), f"outputs/{file_prefix}.pth")
                cm = confusion_matrix(all_labels, val_preds)
                ConfusionMatrixDisplay(cm).plot(cmap=plt.cm.Blues)
                plt.savefig(f"outputs/{file_prefix}_confmat.png")
                plt.close()
                my_logger.info(f"Saved new best model at epoch {epoch+1}")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                my_logger.info("Early stopping triggered.")
                break

        my_logger.info("Training completed successfully.")

    except Exception as e:
        my_logger.error(f"Error during training: {e}")
        my_logger.error(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--i", type=int, default=0)
    parser.add_argument('--dataset', default='ccccii')
    parser.add_argument('--run_cloud', action='store_true')
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument('--vgg_model_path', default='models/vgg_binary_best.pth')
    parser.add_argument('--sequence_length', type=int, default=30)
    args = parser.parse_args()

    mlflow.start_run()

    dataset_folder = f"data/{args.dataset}"
    if args.run_cloud:
        load_dotenv()
        download_from_blob(os.getenv('AZURE_STORAGE_ACCOUNT'), os.getenv('AZURE_STORAGE_KEY'), os.getenv('BLOB_CONTAINER'), dataset_folder)
        if not os.path.exists(args.vgg_model_path):
            download_from_blob_with_access_key(os.getenv('PRETRAINED_BINARY_VGG_MODEL_URI'), os.getenv('AZURE_STORAGE_KEY'), args.vgg_model_path)

    dataset = DatasetSequence2DBinary(dataset_folder, args.sequence_length, args.max_samples)

    patient_ids = [pid.item() for pid in dataset.patient_ids]
    labels = [lbl.item() for lbl in dataset.labels]

    skf = StratifiedKFold(args.k, shuffle=True, random_state=42)
    splits = list(skf.split(patient_ids, labels))[args.i]

    train_loader = DataLoader(Subset(dataset, splits[0]), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, splits[1]), batch_size=args.batch_size, shuffle=False)

    train_model(train_loader, val_loader, args.num_epochs, args.learning_rate, args.vgg_model_path)

    mlflow.end_run()

if __name__ == "__main__":
    main()
