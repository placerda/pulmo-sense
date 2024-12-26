import argparse
import os
import random
import time
import traceback

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold

from pathlib import Path

from datasets.ccccii_dataset import CCCCIIDataset2D
from utils.download import download_from_blob
from utils.log_config import get_custom_logger
my_logger = get_custom_logger('train_lstm_multiclass')

class CNN_Net(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(CNN_Net, self).__init__()
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
        if len(x.size()) != 4:
            raise ValueError(f"Expected 4D input (batch_size, channels, height, width), got {x.size()}")
        batch_size = x.size(0)
        x = x.float()
        embedding = self.cnn(x)
        embedding = embedding.view(batch_size, -1)
        
        if return_embedding:
            return embedding

        x = self.fc(embedding)
        return x

class LSTM_Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
        super(LSTM_Net, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Get last time step
        out = self.dropout(out)
        out = self.fc(out)
        return out

class CCCCIIDatasetSequence2D(Dataset):
    def __init__(self, dataset_folder, sequence_length=30, max_samples=None):
        self.sequence_length = sequence_length
        self.dataset_folder = dataset_folder
        self.max_samples = max_samples

        dataset = CCCCIIDataset2D(dataset_folder, max_samples)
        data = []
        for i in range(len(dataset)):
            slice_image, patient_id, label = dataset[i]
            data.append((slice_image, patient_id, label))
        
        patient_slices = {}
        patient_labels = {}
        for slice_image, patient_id, label in data:
            if patient_id not in patient_slices:
                patient_slices[patient_id] = []
                patient_labels[patient_id] = []
            patient_slices[patient_id].append(slice_image)
            patient_labels[patient_id].append(label)
        
        self.sequences = []
        self.labels = []
        for patient_id in patient_slices:
            slices = patient_slices[patient_id]
            labels = patient_labels[patient_id]
            num_slices = len(slices)
            num_sequences = num_slices // self.sequence_length
            for i in range(num_sequences):
                start_idx = i * self.sequence_length
                end_idx = start_idx + self.sequence_length
                seq_slices = slices[start_idx:end_idx]
                seq_labels = labels[start_idx:end_idx]
                label = max(set(seq_labels), key=seq_labels.count)
                self.sequences.append(seq_slices)
                self.labels.append(label)
            if num_slices % self.sequence_length != 0:
                seq_slices = slices[-self.sequence_length:]
                seq_labels = labels[-self.sequence_length:]
                label = max(set(seq_labels), key=seq_labels.count)
                self.sequences.append(seq_slices)
                self.labels.append(label)
        
        if self.max_samples is not None:
            self.sequences = self.sequences[:self.max_samples]
            self.labels = self.labels[:self.max_samples]
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_slices = self.sequences[idx]
        label = self.labels[idx]
        seq_slices = torch.stack([torch.from_numpy(slice_image) for slice_image in seq_slices])
        return seq_slices, label

def train_model(train_dataset_loader, val_dataset_loader, num_epochs, learning_rate, cnn_model_path):
    start_time = time.time()
    my_logger.info('Starting Training')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_logger.info(f'Using device: {device}')

    try:
        num_classes = 3
        dropout_rate = 0.5
        cnn_model = CNN_Net(num_classes=num_classes, dropout_rate=dropout_rate)
        cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
        cnn_model.eval()
        cnn_model.to(device)
        for param in cnn_model.parameters():
            param.requires_grad = False

        input_size = 128
        hidden_size = 128
        num_layers = 1
        lstm_model = LSTM_Net(input_size, hidden_size, num_layers, num_classes, dropout_rate=dropout_rate)
        lstm_model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate, weight_decay=1e-5)

        early_stopping_patience = 3
        epochs_without_improvement = 0

        best_recall = 0.0

        for epoch in range(num_epochs):
            my_logger.info(f'Starting epoch {epoch + 1}')

            lstm_model.train()
            total_loss = 0
            correct_predictions = 0
            total_samples = 0

            for i, (inputs, labels) in enumerate(train_dataset_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                batch_size, seq_len, channels, height, width = inputs.size()
                inputs = inputs.view(batch_size * seq_len, channels, height, width)
                with torch.no_grad():
                    features = cnn_model(inputs, return_embedding=True)
                features = features.view(batch_size, seq_len, -1)

                outputs = lstm_model(features)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_size
                predictions = outputs.argmax(dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)

                if i % 100 == 0:
                    batch_accuracy = (predictions == labels).float().mean().item()
                    my_logger.info(f'Train batch [{i+1}/{len(train_dataset_loader)}], Loss: {loss.item()}, Accuracy: {batch_accuracy}')
                    mlflow.log_metrics({'running_train_loss': loss.item(), 'running_train_accuracy': batch_accuracy}, step=epoch * len(train_dataset_loader) + i)

            train_loss = total_loss / total_samples
            train_accuracy = correct_predictions / total_samples
            mlflow.log_metrics({'train_loss': train_loss, 'train_accuracy': train_accuracy}, step=epoch)
            my_logger.info(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss}, Training Accuracy: {train_accuracy}')

            my_logger.info(f'Starting validation phase for epoch {epoch + 1}')
            lstm_model.eval()
            with torch.no_grad():
                val_loss = 0
                correct = 0
                total = 0
                all_labels = []
                all_probabilities = []

                for j, (inputs, labels) in enumerate(val_dataset_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    batch_size, seq_len, channels, height, width = inputs.size()
                    inputs = inputs.view(batch_size * seq_len, channels, height, width)
                    features = cnn_model(inputs, return_embedding=True)
                    features = features.view(batch_size, seq_len, -1)
                    outputs = lstm_model(features)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * batch_size

                    probabilities = torch.softmax(outputs, dim=1)
                    predicted = probabilities.argmax(dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    if j % 100 == 0:
                        batch_accuracy = (predicted == labels).float().mean().item()
                        my_logger.info(f'Val batch [{j+1}/{len(val_dataset_loader)}], Loss: {loss.item()}, Accuracy: {batch_accuracy}')

                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())

                val_loss /= total
                val_accuracy = 100 * correct / total

                val_recall = recall_score(all_labels, np.array(all_probabilities).argmax(axis=1), average='macro', zero_division=0)
                val_precision = precision_score(all_labels, np.array(all_probabilities).argmax(axis=1), average='macro', zero_division=0)
                f1 = f1_score(all_labels, np.array(all_probabilities).argmax(axis=1), average='macro', zero_division=0)
                try:
                    auc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr')
                except ValueError as e:
                    my_logger.error(f'Error calculating AUC: {e}')
                    auc = 0

                my_logger.info(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, '
                               f'Recall: {val_recall:.2f}, Precision: {val_precision:.2f}, F1 Score: {f1:.2f}, AUC: {auc:.2f}')

                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
                mlflow.log_metric("val_recall", val_recall, step=epoch)
                mlflow.log_metric("val_precision", val_precision, step=epoch)
                mlflow.log_metric("val_f1_score", f1, step=epoch)
                mlflow.log_metric("val_auc", auc, step=epoch)

                if val_recall > best_recall:
                    best_recall = val_recall
                    epochs_without_improvement = 0
                    output_dir = './outputs'
                    os.makedirs(output_dir, exist_ok=True)
                    file_name = f'lstm_multiclass_{total_samples}smps_{epoch + 1}epoch_{learning_rate:.5f}lr_{val_recall:.3f}rec.pth'
                    torch.save(lstm_model.state_dict(), f'{output_dir}/{file_name}')
                    my_logger.info(f'New best model saved with recall: {val_recall:.3f}')

                    cm = confusion_matrix(all_labels, np.array(all_probabilities).argmax(axis=1))
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                    disp.plot(cmap=plt.cm.Blues)
                    confusion_matrix_file = f'{output_dir}/confusion_matrix_{epoch+1}.png'
                    plt.savefig(confusion_matrix_file)

                else:
                    epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                my_logger.info(f'Early stopping triggered. Last Recall: {val_recall}, Last Precision: {val_precision}, Last Accuracy: {val_accuracy}')
                break

        my_logger.info('Finished Training')
        training_time = time.time() - start_time
        my_logger.info(f'Training completed in {training_time:.2f} seconds')

    except Exception as e:
        my_logger.error("Error during training: %s", str(e))
        my_logger.error("Detailed traceback:")
        my_logger.error(traceback.format_exc())

def main():
    my_logger.info(f"Torch version: {torch.__version__}")
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument("--num_epochs", type=int, help="number of epochs to train")
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--learning_rate", type=float, help="learning rate")
    parser.add_argument("--k", type=int, help="number of folds for cross-validation")
    parser.add_argument("--i", type=int, help="current fold index (0-based)")
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--run_cloud', action='store_true', help='Flag to indicate whether to run in cloud mode')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use')
    parser.add_argument('--cnn_model_path', type=str, help='Path to pretrained CNN model weights')
    parser.add_argument('--sequence_length', type=int, default=30, help='Sequence length for LSTM input')

    args = parser.parse_args()
    my_logger.info(f"Arguments parsed: {args}")

    my_logger.info("Setting dataset variables")
    dataset = args.dataset

    if not args.run_cloud:
        my_logger.info(f"Running in local mode, setting dataset folder to 'data/raw/{dataset}'")
        dataset_folder = f"data/raw/{dataset}"
    else:
        my_logger.info("Running in cloud mode, downloading dataset from blob storage")
        dataset_folder = dataset
        load_dotenv()
        storage_account = os.getenv('AZURE_STORAGE_ACCOUNT')
        storage_account_key = os.getenv('AZURE_STORAGE_KEY')
        container_name = os.getenv('BLOB_CONTAINER')
        my_logger.info(f"Downloading dataset from blob: storage_account={storage_account}, container_name={container_name}")
        download_from_blob(storage_account, storage_account_key, container_name, dataset_folder)

    my_logger.info("Loading dataset")
    sequence_length = args.sequence_length
    my_dataset = CCCCIIDatasetSequence2D(dataset_folder, sequence_length=sequence_length, max_samples=args.max_samples)
    my_logger.info(f"Dataset loaded with a maximum of {args.max_samples} samples and sequence length {sequence_length}")

    labels = my_dataset.labels

    if args.i < 0 or args.i >= args.k:
        my_logger.error(f"Invalid fold index 'i': {args.i}. It must be between 0 and {args.k - 1}.")
        raise ValueError(f"Fold index 'i' must be between 0 and {args.k - 1}, but got {args.i}.")

    my_logger.info(f"Performing Stratified K-Fold with {args.k} splits")
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)
    splits = list(skf.split(np.zeros(len(my_dataset)), labels))

    train_idx, val_idx = splits[args.i]
    my_logger.info(f"Train index: {train_idx[:10]}... ({len(train_idx)} samples)")
    my_logger.info(f"Val index: {val_idx[:10]}... ({len(val_idx)} samples)")

    my_logger.info("Creating data loaders")
    train_dataset = Subset(my_dataset, train_idx)
    val_dataset = Subset(my_dataset, val_idx)

    train_dataset_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    my_logger.info("Data loaders created")

    my_logger.info("Starting model training")
    train_model(
        train_dataset_loader=train_dataset_loader,
        val_dataset_loader=val_dataset_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        cnn_model_path=args.cnn_model_path
    )
    my_logger.info("Model training completed")

    mlflow.end_run()
    my_logger.info("MLflow run ended")

if __name__ == "__main__":
    main()
