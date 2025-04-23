import argparse
import os
import time
import traceback
from dotenv import load_dotenv
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import random

from datasets import DatasetSequence2D
from utils.log_config import get_custom_logger
from utils.download import download_from_blob, download_from_blob_with_access_key

my_logger = get_custom_logger('train_lstm_attention_multiclass')

# -----------------------------
# Simple CNN for feature extraction (same as in the original code)
# -----------------------------
class CNN_Net(nn.Module):
    def __init__(self, num_classes, input_height=None, input_width=None, dropout_rate=0.5):
        super(CNN_Net, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(dropout_rate),
        )
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x, return_embedding=False):
        """
        x shape: [batch_size, channels, height, width]
        """
        if len(x.size()) != 4:
            raise ValueError(f"Expected 4D input (batch_size, channels, height, width), got {x.size()}")
        batch_size = x.size(0)
        x = x.float()
        embedding = self.cnn(x)                      # [batch_size, 128, 1, 1]
        embedding = embedding.view(batch_size, -1)   # [batch_size, 128]
        
        if return_embedding:
            return embedding

        out = self.fc(embedding)
        return out


# -----------------------------
# Attention Module
# -----------------------------
class TemporalAttention(nn.Module):
    """
    A simple additive attention mechanism that produces 
    a context vector by weighting each time step of the LSTM.
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        # We map each LSTM hidden state (size = hidden_dim) to a scalar score
        self.attn_score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs):
        """
        Args:
            lstm_outputs: [batch_size, seq_len, hidden_dim]
        Returns:
            context: [batch_size, hidden_dim] (weighted sum over time steps)
            attn_weights: [batch_size, seq_len] (attention weights per time step)
        """
        # Compute attention energies: shape = [batch_size, seq_len, 1]
        energies = self.attn_score(lstm_outputs)

        # Squeeze last dim: [batch_size, seq_len]
        energies = energies.squeeze(-1)

        # Compute normalized attention weights
        attn_weights = torch.softmax(energies, dim=1)  # [batch_size, seq_len]

        # Apply attention weights: 
        # Weighted sum over the seq_len dimension => context has shape [batch_size, hidden_dim]
        attn_weights_expanded = attn_weights.unsqueeze(-1)  # [batch_size, seq_len, 1]
        context = torch.sum(lstm_outputs * attn_weights_expanded, dim=1)

        return context, attn_weights


# -----------------------------
# LSTM with Attention
# -----------------------------
class LSTM_AttnNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
        """
        Args:
          input_size: dimension of features (e.g., 128 from CNN embeddings)
          hidden_size: dimension of LSTM hidden states
          num_layers: number of LSTM layers
          num_classes: number of output classes
        """
        super(LSTM_AttnNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = TemporalAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x shape: [batch_size, seq_len, input_size]
        Returns:
          logits: [batch_size, num_classes]
          attn_weights: [batch_size, seq_len]
        """
        # LSTM forward pass => outputs shape = [batch_size, seq_len, hidden_size]
        lstm_outputs, _ = self.lstm(x)

        # Apply temporal attention
        context_vector, attn_weights = self.attention(lstm_outputs)

        # Dropout, then classification
        context_vector = self.dropout(context_vector)
        logits = self.fc(context_vector)  # shape = [batch_size, num_classes]

        return logits, attn_weights


# -----------------------------
# TRAINING FUNCTION
# -----------------------------
def train_model(train_loader, val_loader, num_epochs, learning_rate, cnn_model_path):
    start_time = time.time()
    my_logger.info('Starting Training with Temporal Attention LSTM')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_logger.info(f'Using device: {device}')

    try:

        # Check if the model file exists
        if not os.path.exists(cnn_model_path):
            raise FileNotFoundError(f"Model file '{cnn_model_path}' does not exist.")
        
        # Check if the model file is not empty
        if os.path.getsize(cnn_model_path) == 0:
            raise ValueError(f"Model file '{cnn_model_path}' is empty.")
        
        # Optional: Log the size of the model file
        file_size = os.path.getsize(cnn_model_path)
        my_logger.info(f"Loading CNN model from '{cnn_model_path}' (Size: {file_size} bytes)")

        # -------------------------
        # 1) Load pretrained CNN
        # -------------------------
        num_classes = 3
        input_height = 512
        input_width = 512
        dropout_rate = 0.5
        cnn_model = CNN_Net(num_classes, input_height, input_width, dropout_rate=dropout_rate).to(device)    
        cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
        cnn_model.eval()
        for param in cnn_model.parameters():
            param.requires_grad = False

        # -------------------------
        # 2) Initialize LSTM + Attention
        # -------------------------
        input_size = 128   # Because our CNN outputs a 128-dim feature vector
        hidden_size = 128
        num_layers = 1
        attn_lstm_model = LSTM_AttnNet(input_size, hidden_size, num_layers, num_classes, dropout_rate=dropout_rate).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(attn_lstm_model.parameters(), lr=learning_rate, weight_decay=1e-5)

        early_stopping_patience = 3
        epochs_without_improvement = 0
        best_recall = 0.0

        # -------------------------
        # 3) Training Loop
        # -------------------------
        for epoch in range(num_epochs):
            my_logger.info(f'Starting epoch {epoch + 1}/{num_epochs}')
            attn_lstm_model.train()

            total_loss = 0
            correct_predictions = 0
            total_samples = 0

            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                batch_size, seq_len, channels, height, width = inputs.size()
                
                # Flatten to pass each slice to CNN => [batch_size * seq_len, channels, height, width]
                inputs = inputs.view(batch_size * seq_len, channels, height, width)

                with torch.no_grad():
                    # Extract CNN embeddings => [batch_size * seq_len, 128]
                    features = cnn_model(inputs, return_embedding=True)

                # Reshape back to [batch_size, seq_len, 128] for LSTM
                features = features.view(batch_size, seq_len, -1)

                # Forward pass through LSTM with attention
                outputs, attn_weights = attn_lstm_model(features)

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
                    my_logger.info(f'Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {batch_accuracy:.4f}')
                    mlflow.log_metrics({'running_train_loss': loss.item(),
                                        'running_train_accuracy': batch_accuracy},
                                       step=epoch * len(train_loader) + i)

            train_loss = total_loss / total_samples
            train_accuracy = correct_predictions / total_samples
            mlflow.log_metrics({'train_loss': train_loss, 'train_accuracy': train_accuracy}, step=epoch)

            my_logger.info(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')

            # -------------------------
            # 4) Validation Loop
            # -------------------------
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

                    # Flatten to pass each slice to CNN
                    inputs = inputs.view(batch_size * seq_len, channels, height, width)
                    features = cnn_model(inputs, return_embedding=True)
                    features = features.view(batch_size, seq_len, -1)

                    outputs, attn_weights = attn_lstm_model(features)
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

            val_recall = recall_score(all_labels, np.argmax(all_probabilities, axis=1), average='macro', zero_division=0)
            val_precision = precision_score(all_labels, np.argmax(all_probabilities, axis=1), average='macro', zero_division=0)
            val_f1 = f1_score(all_labels, np.argmax(all_probabilities, axis=1), average='macro', zero_division=0)
            try:
                val_auc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr')
            except ValueError:
                val_auc = 0.0

            my_logger.info(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%, '
                           f'Recall: {val_recall:.2f}, Precision: {val_precision:.2f}, F1: {val_f1:.2f}, AUC: {val_auc:.2f}')

            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
            mlflow.log_metric("val_recall", val_recall, step=epoch)
            mlflow.log_metric("val_precision", val_precision, step=epoch)
            mlflow.log_metric("val_f1_score", val_f1, step=epoch)
            mlflow.log_metric("val_auc", val_auc, step=epoch)

            # Check if this is the best model so far
            if val_recall > best_recall:
                best_recall = val_recall
                epochs_without_improvement = 0
                output_dir = './outputs'
                os.makedirs(output_dir, exist_ok=True)
                file_prefix = f'attn_lstm_{total}smps_{epoch + 1:03}epoch_{learning_rate:.5f}lr_{val_recall:.3f}rec'
                file_name = f'{file_prefix}.pth'
                torch.save(attn_lstm_model.state_dict(), f'{output_dir}/{file_name}')
                my_logger.info(f'New best model saved with recall: {val_recall:.3f}')

                # Confusion matrix
                cm = confusion_matrix(all_labels, np.array(all_probabilities).argmax(axis=1))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap=plt.cm.Blues)
                confusion_matrix_file = f'{output_dir}/{file_prefix}_confusion_matrix.png'
                plt.savefig(confusion_matrix_file)
                plt.close()
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                my_logger.info(f'Early stopping triggered. Last Recall: {val_recall}, Last Precision: {val_precision}, Last Accuracy: {val_accuracy:.2f}%')
                break

        my_logger.info('Finished Training with Attention LSTM')
        training_time = time.time() - start_time
        my_logger.info(f'Training completed in {training_time:.2f} seconds')

    except Exception as e:
        my_logger.error("Error during training: %s", str(e))
        my_logger.error("Detailed traceback:")
        my_logger.error(traceback.format_exc())


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def main():
    my_logger.info(f"Torch version: {torch.__version__}")
    mlflow.start_run()
    my_logger.info("MLflow run started")

    parser = argparse.ArgumentParser(description='Train an LSTM model with Temporal Attention for multiclass classification')
    parser.add_argument("--num_epochs", type=int, default=20, help="number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="learning rate")
    parser.add_argument("--k", type=int, default=5, help="number of folds for cross-validation")
    parser.add_argument("--i", type=int, default=0, help="current fold index (0-based)")
    parser.add_argument('--dataset', type=str, default='ccccii', help='Dataset name')
    parser.add_argument('--run_cloud', action='store_true', help='Flag to indicate whether to run in cloud mode')
    parser.add_argument('--max_samples', type=int, default=0, help='Maximum number of samples to use')
    parser.add_argument('--cnn_model_path', type=str, default='models/pretrained_cnn.pth', help='Path to pretrained CNN model weights')
    parser.add_argument('--sequence_length', type=int, default=30, help='Sequence length for LSTM input')

    args = parser.parse_args()
    my_logger.info(f"Arguments parsed: {args}")
    my_logger.info(f"Current Working Directory: {os.getcwd()}")

    dataset = args.dataset
    if not args.run_cloud:
        my_logger.info("Running in local mode")
        dataset_folder = f"data/{dataset}"
    else:
        my_logger.info("Running in cloud mode, downloading dataset and pre-trained model from blob storage")
        load_dotenv()
        storage_account = os.getenv('AZURE_STORAGE_ACCOUNT')
        storage_account_key = os.getenv('AZURE_STORAGE_KEY')

        try:
            model_uri= os.getenv('PRETRAINED_CNN_MODEL_URI')        
            my_logger.info(f"Downloading pre-trained model from blob: storage_account={storage_account}, pre-trained model={model_uri}")
            os.makedirs(os.path.dirname(args.cnn_model_path), exist_ok=True)
            download_from_blob_with_access_key(model_uri, storage_account_key, args.cnn_model_path)
            my_logger.info(f"Model downloaded from blob to {args.cnn_model_path}")
        except Exception as download_err:
            my_logger.error(f"Failed to download model from storage account: {str(download_err)}")
            exit(-1)

        try:
            dataset_folder = dataset
            container_name = os.getenv('BLOB_CONTAINER')
            my_logger.info(f"Downloading dataset from blob: storage_account={storage_account}, container_name={container_name}")
            download_from_blob(storage_account, storage_account_key, container_name, dataset_folder)
        except Exception as download_err:
            my_logger.error(f"Failed to download dataset from storage account: {str(download_err)}")
            exit(-1)




    my_logger.info("Loading dataset")
    sequence_length = args.sequence_length
    my_dataset = DatasetSequence2D(dataset_folder, sequence_length=sequence_length, max_samples=args.max_samples)
    my_logger.info(f"Dataset loaded with max_samples={args.max_samples}, sequence_length={sequence_length}")

    labels = my_dataset.labels
    if args.i < 0 or args.i >= args.k:
        raise ValueError(f"Fold index 'i' must be between 0 and {args.k - 1}, but got {args.i}.")

    my_logger.info(f"Performing Stratified K-Fold with {args.k} splits")
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)
    splits = list(skf.split(np.zeros(len(my_dataset)), labels))
    train_idx, val_idx = splits[args.i]

    my_logger.info(f"Train index length: {len(train_idx)}, Val index length: {len(val_idx)}")
    train_dataset = Subset(my_dataset, train_idx)
    val_dataset = Subset(my_dataset, val_idx)

    train_dataset_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    my_logger.info("Starting model training with Attention-based LSTM")
    train_model(
        train_loader=train_dataset_loader,
        val_loader=val_dataset_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        cnn_model_path=args.cnn_model_path
    )
    my_logger.info("Model training completed")

    mlflow.end_run()
    my_logger.info("MLflow run ended")


if __name__ == "__main__":
    main()
