#!/usr/bin/env python
"""
Train LSTM+Attention with VGG-based Feature Extraction for Binary Classification

This script loads a pretrained VGG model (used as a feature extractor),
then trains an LSTM with temporal attention over sequences of VGG features.
It is adapted from your multiclass version by setting num_classes=2.
The path to the pretrained VGG weights is provided as an argument.
"""

import argparse
import os
import time
import traceback
import random

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Subset
from dotenv import load_dotenv

from datasets import CCCCIIDatasetSequence2DBinary
from utils.download import download_from_blob, download_from_blob_with_access_key
from utils.log_config import get_custom_logger

my_logger = get_custom_logger('train_lstm_attention_vgg_binary')

# ----------------------------------------------------------------------
# 1) VGG Feature Extractor – loads a fine‐tuned VGG and replaces its classifier
# ----------------------------------------------------------------------
class VGGFeatureExtractor(nn.Module):
    def __init__(self, vgg_weights_path):
        super(VGGFeatureExtractor, self).__init__()
        self.vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        in_features = self.vgg.classifier[6].in_features
        # Replace final FC with identity so that we get penultimate features
        self.vgg.classifier[6] = nn.Identity()
        state_dict = torch.load(vgg_weights_path, map_location='cpu')
        self.vgg.load_state_dict(state_dict, strict=False)
        self.embedding_dim = in_features

    def forward(self, x):
        # x: [batch, 1, H, W] -> repeat to 3 channels and resize to 224x224
        x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        features = self.vgg(x)  # expected shape: [batch, embedding_dim]
        return features

# ----------------------------------------------------------------------
# 2) Temporal Attention Module
# ----------------------------------------------------------------------
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attn_score = nn.Linear(hidden_dim, 1, bias=False)
    def forward(self, lstm_outputs):
        # lstm_outputs: [batch, seq, hidden_dim]
        energies = self.attn_score(lstm_outputs).squeeze(-1)  # [batch, seq]
        attn_weights = torch.softmax(energies, dim=1)
        context = torch.sum(lstm_outputs * attn_weights.unsqueeze(-1), dim=1)
        return context, attn_weights

# ----------------------------------------------------------------------
# 3) LSTM with Attention for Classification
# ----------------------------------------------------------------------
class LSTM_AttnNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
        super(LSTM_AttnNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = TemporalAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        # x: [batch, seq, input_size]
        lstm_outputs, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_outputs)
        context = self.dropout(context)
        logits = self.fc(context)
        return logits, attn_weights

# ----------------------------------------------------------------------
# 4) Training Function
# ----------------------------------------------------------------------
def train_model(train_loader, val_loader, num_epochs, learning_rate, vgg_model_path):
    start_time = time.time()
    my_logger.info('Starting LSTM+Attention training (binary)')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_logger.info(f'Using device: {device}')

    try:
        # Load pretrained VGG feature extractor (frozen)
        feature_extractor = VGGFeatureExtractor(vgg_weights_path=vgg_model_path).to(device)
        feature_extractor.eval()
        for param in feature_extractor.parameters():
            param.requires_grad = False

        # Set binary classification (num_classes=2)
        num_classes = 2
        input_size = feature_extractor.embedding_dim  # e.g., 4096 for VGG16_BN
        hidden_size = 128
        num_layers = 1
        dropout_rate = 0.5

        attn_lstm_model = LSTM_AttnNet(input_size, hidden_size, num_layers, num_classes, dropout_rate=dropout_rate).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(attn_lstm_model.parameters(), lr=learning_rate, weight_decay=1e-5)

        best_recall = 0.0
        early_stopping_patience = 3
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            my_logger.info(f'Starting epoch {epoch+1}/{num_epochs}')
            attn_lstm_model.train()
            total_loss = 0
            correct_predictions = 0
            total_samples = 0

            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                bsz, seq_len, ch, h, w = inputs.size()
                # Flatten to process each slice through VGG independently
                inputs = inputs.view(bsz * seq_len, ch, h, w)
                with torch.no_grad():
                    vgg_features = feature_extractor(inputs)  # [bsz*seq_len, embedding_dim]
                vgg_features = vgg_features.view(bsz, seq_len, -1)
                outputs, attn_weights = attn_lstm_model(vgg_features)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * bsz
                preds = outputs.argmax(dim=1)
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)

                if i % 5 == 0:
                    batch_acc = (preds == labels).float().mean().item()
                    my_logger.info(f'Epoch {epoch+1}, Batch [{i+1}/{len(train_loader)}]: Loss={loss.item():.4f}, Acc={batch_acc:.4f}')
                    mlflow.log_metrics({'running_train_loss': loss.item(), 'running_train_accuracy': batch_acc},
                                       step=epoch * len(train_loader) + i)

            train_loss = total_loss / total_samples
            train_accuracy = correct_predictions / total_samples
            mlflow.log_metrics({'train_loss': train_loss, 'train_accuracy': train_accuracy}, step=epoch)
            my_logger.info(f'Epoch {epoch+1}: Training Loss={train_loss:.4f}, Accuracy={train_accuracy:.4f}')

            # --- Validation ---
            attn_lstm_model.eval()
            val_loss = 0
            correct = 0
            total = 0
            all_labels = []
            all_probs = []
            with torch.no_grad():
                for j, (inputs, labels) in enumerate(val_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    bsz, seq_len, ch, h, w = inputs.size()
                    inputs = inputs.view(bsz * seq_len, ch, h, w)
                    vgg_features = feature_extractor(inputs)
                    vgg_features = vgg_features.view(bsz, seq_len, -1)
                    outputs, attn_weights = attn_lstm_model(vgg_features)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * bsz
                    probabilities = torch.softmax(outputs, dim=1)
                    preds = probabilities.argmax(dim=1)
                    total += labels.size(0)
                    correct += (preds == labels).sum().item()
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probabilities.cpu().numpy())
            val_loss /= total
            val_accuracy = 100.0 * correct / total

            # IMPORTANT: For binary classification the positive class is the one with label 0.
            # Extract probability for class 0 and compute ROC AUC with pos_label=0.
            all_probs = np.array(all_probs)
            positive_probs = all_probs[:, 0]
            try:
                val_auc = roc_auc_score(all_labels, positive_probs)
                my_logger.info("AUC computed using positive class probabilities (index 0).")
            except ValueError as e:
                my_logger.error("Error computing AUC: %s", e)
                val_auc = 0.0

            # For recall, precision, f1 we can use binary averaging
            val_preds = np.argmax(all_probs, axis=1)
            val_recall = recall_score(all_labels, val_preds, average='binary', zero_division=0)
            val_precision = precision_score(all_labels, val_preds, average='binary', zero_division=0)
            val_f1 = f1_score(all_labels, val_preds, average='binary', zero_division=0)

            my_logger.info(
                f'Epoch {epoch+1} Validation: Loss={val_loss:.4f}, Acc={val_accuracy:.2f}%, '
                f'Recall={val_recall:.2f}, Precision={val_precision:.2f}, F1={val_f1:.2f}, AUC={val_auc:.2f}'
            )
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
            mlflow.log_metric("val_recall", val_recall, step=epoch)
            mlflow.log_metric("val_precision", val_precision, step=epoch)
            mlflow.log_metric("val_f1_score", val_f1, step=epoch)
            mlflow.log_metric("val_auc", val_auc, step=epoch)

            if val_recall > best_recall:
                best_recall = val_recall
                epochs_without_improvement = 0
                output_dir = './outputs'
                os.makedirs(output_dir, exist_ok=True)
                file_prefix = f'attn_lstm_binary_{total_samples}smps_{epoch+1:03}ep_{learning_rate:.5f}lr_{val_recall:.3f}rec'
                file_path = os.path.join(output_dir, f'{file_prefix}.pth')
                torch.save(attn_lstm_model.state_dict(), file_path)
                my_logger.info(f'New best model saved with Recall: {val_recall:.3f}')
                cm = confusion_matrix(all_labels, np.argmax(all_probs, axis=1))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap=plt.cm.Blues)
                cm_file = os.path.join(output_dir, f'{file_prefix}_confmat.png')
                plt.savefig(cm_file)
                plt.close()
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                my_logger.info(f'Early stopping triggered at epoch {epoch+1}')
                break

        my_logger.info('Finished Training with LSTM Attention VGG (binary)')
        my_logger.info(f'Total training time: {time.time()-start_time:.2f} seconds')

    except Exception as e:
        my_logger.error("Error during training: %s", str(e))
        my_logger.error(traceback.format_exc())

def main():
    mlflow.start_run()
    my_logger.info("MLflow run started")
    parser = argparse.ArgumentParser(description='Train LSTM Attention VGG Binary Model')
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--i", type=int, default=0)
    parser.add_argument('--dataset', type=str, default='ccccii', help='Dataset name')
    parser.add_argument('--run_cloud', action='store_true')
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument('--cnn_model_path', type=str, default='models/pretrained_vgg_binary.pth',
                        help="Path to pretrained VGG model weights for binary classification")
    parser.add_argument('--sequence_length', type=int, default=30)
    args = parser.parse_args()
    my_logger.info(f"Arguments parsed: {args}")
    dataset = args.dataset
    if not args.run_cloud:
        dataset_folder = f"data/{dataset}"
    else:
        dataset_folder = args.dataset
        load_dotenv()
        storage_account = os.getenv('AZURE_STORAGE_ACCOUNT')
        storage_account_key = os.getenv('AZURE_STORAGE_KEY')
        container_name = os.getenv('BLOB_CONTAINER')
        my_logger.info(f"Downloading dataset from blob: storage_account={storage_account}, container_name={container_name}")
        download_from_blob(storage_account, storage_account_key, container_name, dataset_folder)
        try:
            if not os.path.exists(args.cnn_model_path):
                model_uri = os.getenv('PRETRAINED_VGG_MODEL_URI', '')
                my_logger.info(f"Downloading pretrained VGG model from blob: {model_uri}")
                os.makedirs(os.path.dirname(args.cnn_model_path), exist_ok=True)
                download_from_blob_with_access_key(model_uri, storage_account_key, args.cnn_model_path)
                my_logger.info(f"Pretrained VGG model downloaded to {args.cnn_model_path}")
        except Exception as download_err:
            my_logger.error(f"Failed to download pretrained VGG model: {str(download_err)}")
            exit(-1)
    my_logger.info("Loading dataset")
    sequence_length = args.sequence_length
    my_dataset = CCCCIIDatasetSequence2DBinary(dataset_folder, sequence_length=sequence_length, max_samples=args.max_samples)
    my_logger.info(f"Dataset loaded with max_samples={args.max_samples}, sequence_length={sequence_length}")
    labels = my_dataset.labels
    if args.i < 0 or args.i >= args.k:
        raise ValueError(f"Fold index 'i' must be between 0 and {args.k - 1}, but got {args.i}.")
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)
    splits = list(skf.split(np.zeros(len(my_dataset)), labels))
    train_idx, val_idx = splits[args.i]
    my_logger.info(f"Train index length: {len(train_idx)}, Val index length: {len(val_idx)}")
    train_dataset = Subset(my_dataset, train_idx)
    val_dataset = Subset(my_dataset, val_idx)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    my_logger.info("Data loaders created")
    my_logger.info("Starting model training")
    train_model(train_loader, val_loader, args.num_epochs, args.learning_rate, args.cnn_model_path)
    mlflow.end_run()
    my_logger.info("MLflow run ended")

if __name__ == "__main__":
    main()
