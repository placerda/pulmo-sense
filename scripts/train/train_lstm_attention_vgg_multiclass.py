# File: /scripts/train/train_lstm_attention_vgg_multiclass.py

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

# local imports
from datasets import DatasetSequence2D
from utils.download import download_from_blob, download_from_blob_with_access_key
from utils.log_config import get_custom_logger

my_logger = get_custom_logger('train_lstm_attention_vgg_multiclass')


# ----------------------------------------------------------------------
# 1) VGG Feature Extractor (similar to train_lstm_vgg_multiclass.py)
# ----------------------------------------------------------------------
class VGGFeatureExtractor(nn.Module):
    """
    Loads a VGG16 (with BatchNorm) model; replaces the final classification layer
    with Identity, so we can extract a high-level embedding (e.g., size 4096).
    We assume you trained/fine-tuned it previously (as in train_vgg_multiclass.py).
    """

    def __init__(self, vgg_weights_path):
        super(VGGFeatureExtractor, self).__init__()
        # Load a pretrained VGG16 with BatchNorm
        self.vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)

        # Replace the final classifier layer with an identity to get the penultimate features
        in_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Identity()

        # Load your custom fine-tuned weights
        state_dict = torch.load(vgg_weights_path, map_location='cpu')
        # strict=False in case classifier layer shape changed
        self.vgg.load_state_dict(state_dict, strict=False)

        # Store the embedding dimension (often 4096 for VGG16_BN)
        self.embedding_dim = in_features

    def forward(self, x):
        """
        x: [batch_size, 1, H, W], single-channel
        Returns a feature vector of shape [batch_size, 4096] (for VGG16_BN).
        """
        # Repeat the single channel to 3 channels
        x = x.repeat(1, 3, 1, 1)
        # Resize to (224,224) if not already. This is essential for VGG.
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # Pass through the VGG feature extractor (up to the penultimate layer)
        features = self.vgg(x)  # shape: [batch_size, 4096]
        return features


# ----------------------------------------------------------------------
# 2) Temporal Attention Module (same as in train_lstm_attention_multiclass.py)
# ----------------------------------------------------------------------
class TemporalAttention(nn.Module):
    """
    A simple additive attention mechanism over LSTM outputs.
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attn_score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs):
        """
        lstm_outputs: [batch_size, seq_len, hidden_dim]
        Returns:
          context: [batch_size, hidden_dim] (weighted sum across time steps)
          attn_weights: [batch_size, seq_len] (the attention for each time step)
        """
        # energies: [batch_size, seq_len, 1]
        energies = self.attn_score(lstm_outputs)
        energies = energies.squeeze(-1)  # [batch_size, seq_len]

        # attention weights
        attn_weights = torch.softmax(energies, dim=1)  # [batch_size, seq_len]

        # Weighted sum of the LSTM outputs across time
        attn_weights_expanded = attn_weights.unsqueeze(-1)  # [batch_size, seq_len, 1]
        context = torch.sum(lstm_outputs * attn_weights_expanded, dim=1)  # [batch_size, hidden_dim]

        return context, attn_weights


# ----------------------------------------------------------------------
# 3) LSTM with Attention for Classification
# ----------------------------------------------------------------------
class LSTM_AttnNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
        super(LSTM_AttnNet, self).__init__()
        # LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = TemporalAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x: [batch_size, seq_len, input_size]
        Returns:
          logits: [batch_size, num_classes]
          attn_weights: [batch_size, seq_len]
        """
        # LSTM => [batch_size, seq_len, hidden_size]
        lstm_outputs, _ = self.lstm(x)

        # Apply temporal attention
        context_vector, attn_weights = self.attention(lstm_outputs)

        # Classification
        context_vector = self.dropout(context_vector)
        logits = self.fc(context_vector)  # [batch_size, num_classes]

        return logits, attn_weights


# ----------------------------------------------------------------------
# 4) Training Function
# ----------------------------------------------------------------------
def train_model(train_loader, val_loader, num_epochs, learning_rate, vgg_model_path):
    start_time = time.time()
    my_logger.info('Starting LSTM+Attention with VGG-based feature extraction')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_logger.info(f'Using device: {device}')

    try:
        # --- 1) Load your pretrained VGG for feature extraction ---
        feature_extractor = VGGFeatureExtractor(vgg_weights_path=vgg_model_path).to(device)
        feature_extractor.eval()  # Freeze VGG
        for param in feature_extractor.parameters():
            param.requires_grad = False

        # --- 2) Create LSTM+Attention model ---
        num_classes = 3
        input_size = feature_extractor.embedding_dim  # likely 4096
        hidden_size = 128
        num_layers = 1
        dropout_rate = 0.5

        attn_lstm_model = LSTM_AttnNet(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(attn_lstm_model.parameters(), lr=learning_rate, weight_decay=1e-5)

        early_stopping_patience = 3
        epochs_without_improvement = 0
        best_recall = 0.0

        # --- 3) Training Loop ---
        for epoch in range(num_epochs):
            my_logger.info(f'Starting epoch {epoch + 1}/{num_epochs}')
            attn_lstm_model.train()

            total_loss = 0
            correct_predictions = 0
            total_samples = 0

            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                batch_size, seq_len, channels, height, width = inputs.size()

                # Flatten => [batch_size * seq_len, channels, height, width]
                inputs = inputs.view(batch_size * seq_len, channels, height, width)

                # Extract VGG features => [batch_size * seq_len, 4096]
                with torch.no_grad():
                    vgg_features = feature_extractor(inputs)

                # Reshape => [batch_size, seq_len, 4096]
                vgg_features = vgg_features.view(batch_size, seq_len, -1)

                # Forward pass through LSTM+Attention
                outputs, attn_weights = attn_lstm_model(vgg_features)

                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_size
                predictions = outputs.argmax(dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)

                if i % 5 == 0:
                    batch_accuracy = (predictions == labels).float().mean().item()
                    my_logger.info(f'Epoch {epoch+1}, Batch [{i+1}/{len(train_loader)}], '
                                   f'Loss: {loss.item():.4f}, Accuracy: {batch_accuracy:.4f}')
                    mlflow.log_metrics(
                        {'running_train_loss': loss.item(),
                         'running_train_accuracy': batch_accuracy},
                        step=epoch * len(train_loader) + i
                    )

            train_loss = total_loss / total_samples
            train_accuracy = correct_predictions / total_samples
            mlflow.log_metrics({'train_loss': train_loss, 'train_accuracy': train_accuracy}, step=epoch)

            my_logger.info(f'Epoch {epoch+1}: Training Loss={train_loss:.4f}, Acc={train_accuracy:.4f}')

            # --- 4) Validation Loop ---
            attn_lstm_model.eval()
            val_loss = 0
            correct = 0
            total = 0
            all_labels = []
            all_probabilities = []

            with torch.no_grad():
                for j, (inputs, labels) in enumerate(val_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    batch_size, seq_len, channels, height, width = inputs.size()

                    # Flatten
                    inputs = inputs.view(batch_size * seq_len, channels, height, width)

                    # Extract VGG features
                    vgg_features = feature_extractor(inputs)
                    # Reshape => [batch_size, seq_len, 4096]
                    vgg_features = vgg_features.view(batch_size, seq_len, -1)

                    # LSTM + Attention forward
                    outputs, attn_weights = attn_lstm_model(vgg_features)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * batch_size

                    probabilities = torch.softmax(outputs, dim=1)
                    predicted = probabilities.argmax(dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())

            val_loss /= total
            val_accuracy = 100.0 * correct / total

            val_recall = recall_score(all_labels, np.argmax(all_probabilities, axis=1),
                                      average='macro', zero_division=0)
            val_precision = precision_score(all_labels, np.argmax(all_probabilities, axis=1),
                                            average='macro', zero_division=0)
            val_f1 = f1_score(all_labels, np.argmax(all_probabilities, axis=1),
                              average='macro', zero_division=0)
            try:
                val_auc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr')
            except ValueError:
                val_auc = 0.0

            my_logger.info(f'Validation: Loss={val_loss:.4f}, Acc={val_accuracy:.2f}%, '
                           f'Recall={val_recall:.2f}, Precision={val_precision:.2f}, '
                           f'F1={val_f1:.2f}, AUC={val_auc:.2f}')

            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
            mlflow.log_metric("val_recall", val_recall, step=epoch)
            mlflow.log_metric("val_precision", val_precision, step=epoch)
            mlflow.log_metric("val_f1_score", val_f1, step=epoch)
            mlflow.log_metric("val_auc", val_auc, step=epoch)

            # --- 5) Check for best model & Early stopping ---
            if val_recall > best_recall:
                best_recall = val_recall
                epochs_without_improvement = 0

                output_dir = './outputs'
                os.makedirs(output_dir, exist_ok=True)
                file_prefix = f'lstm_attention_vgg_{total}smps_{epoch+1:03}epoch_{learning_rate:.5f}lr_{val_recall:.3f}rec'
                file_name = f'{file_prefix}.pth'
                torch.save(attn_lstm_model.state_dict(), os.path.join(output_dir, file_name))
                my_logger.info(f'New best model saved with recall={val_recall:.3f}')

                # Save confusion matrix
                cm = confusion_matrix(all_labels, np.argmax(all_probabilities, axis=1))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap=plt.cm.Blues)
                confusion_matrix_file = f'{output_dir}/{file_prefix}_confusion_matrix.png'
                plt.savefig(confusion_matrix_file)
                plt.close()
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                my_logger.info(f'Early stopping triggered at epoch {epoch+1}')
                break

        my_logger.info('Finished Training (LSTM+Attention+VGG)')
        training_time = time.time() - start_time
        my_logger.info(f'Total training time: {training_time:.2f} seconds')

    except Exception as e:
        my_logger.error("Error during training: %s", str(e))
        my_logger.error("Detailed traceback:")
        my_logger.error(traceback.format_exc())


# ----------------------------------------------------------------------
# 5) Main
# ----------------------------------------------------------------------
def main():
    my_logger.info(f"Torch version: {torch.__version__}")
    mlflow.start_run()
    my_logger.info("MLflow run started")

    parser = argparse.ArgumentParser(description='Train LSTM+Attention model using VGG feature extraction (multiclass).')
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--k", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--i", type=int, default=0, help="Current fold index (0-based)")
    parser.add_argument("--dataset", type=str, default='ccccii', help="Dataset name")
    parser.add_argument("--run_cloud", action='store_true', help="Flag to indicate whether to run in cloud mode")
    parser.add_argument("--max_samples", type=int, default=0, help="Maximum number of samples to use")
    parser.add_argument("--vgg_model_path", type=str, default='models/vgg_multiclass_best.pth',
                        help="Path to the pretrained VGG model weights")
    parser.add_argument("--sequence_length", type=int, default=30, help="Sequence length for LSTM input")

    args = parser.parse_args()
    my_logger.info(f"Arguments parsed: {args}")
    my_logger.info(f"Current Working Directory: {os.getcwd()}")

    if not args.run_cloud:
        my_logger.info("Running in local mode")
        dataset_folder = f"data/{args.dataset}"
    else:
        my_logger.info("Running in cloud mode, attempting to download dataset and pretrained model from blob storage")
        load_dotenv()
        storage_account = os.getenv("AZURE_STORAGE_ACCOUNT")
        storage_account_key = os.getenv("AZURE_STORAGE_KEY")
        container_name = os.getenv("BLOB_CONTAINER")

        # Download pretrained VGG if needed
        if not os.path.exists(args.vgg_model_path):
            model_uri = os.getenv('PRETRAINED_VGG_MODEL_URI', '')
            my_logger.info(f"Downloading pretrained VGG model from: {model_uri}")
            os.makedirs(os.path.dirname(args.vgg_model_path), exist_ok=True)
            download_from_blob_with_access_key(model_uri, storage_account_key, args.vgg_model_path)
            my_logger.info(f"VGG model downloaded to {args.vgg_model_path}")

        # Download dataset
        dataset_folder = args.dataset
        my_logger.info(f"Downloading dataset from blob storage: {dataset_folder}")
        download_from_blob(storage_account, storage_account_key, container_name, dataset_folder)

    # Load the dataset
    my_logger.info("Loading sequence dataset")
    sequence_length = args.sequence_length
    my_dataset = DatasetSequence2D(dataset_folder, sequence_length=sequence_length, max_samples=args.max_samples)
    my_logger.info(f"Dataset loaded with max_samples={args.max_samples}, sequence_length={sequence_length}")

    # Extract labels
    labels = my_dataset.labels
    if args.i < 0 or args.i >= args.k:
        raise ValueError(f"Fold index 'i' must be between 0 and {args.k - 1}, got {args.i}")

    # Stratified K-Fold
    my_logger.info(f"Performing Stratified K-Fold with k={args.k}")
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)
    splits = list(skf.split(np.zeros(len(my_dataset)), labels))
    train_idx, val_idx = splits[args.i]

    train_dataset = Subset(my_dataset, train_idx)
    val_dataset = Subset(my_dataset, val_idx)
    my_logger.info(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Start training
    my_logger.info("Starting training with LSTM+Attention and VGG features")
    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        vgg_model_path=args.vgg_model_path
    )

    my_logger.info("Training completed")
    mlflow.end_run()
    my_logger.info("MLflow run ended")


if __name__ == "__main__":
    main()
