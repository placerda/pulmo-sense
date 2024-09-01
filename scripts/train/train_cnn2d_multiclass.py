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
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path

from datasets.ccccii_dataset import CCCCIIDataset2D
from utils.download import download_from_blob
from utils.log_config import get_custom_logger

my_logger = get_custom_logger('train_cnn2d_multiclass')

class CNN_Net(nn.Module):
    def __init__(self, num_classes, input_height, input_width, dropout_rate=0.5):
        super(CNN_Net, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),  # Dropout after pooling
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),  # Dropout after pooling
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),  # Dropout after pooling
        )
        self.fc = nn.Linear(128 * (input_height // 8) * (input_width // 8), num_classes)
        
    def forward(self, x):
        if len(x.size()) != 4:
            raise ValueError(f"Expected 4D input (batch_size, channels, height, width), got {x.size()}")

        batch_size, channels, height, width = x.size()
        x = x.float()
        x = self.cnn(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


def train_model(train_dataset_loader, val_dataset_loader, num_epochs, learning_rate):
    start_time = time.time()  # Start time of training
    my_logger.info('Starting Training')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_logger.info(f'Using device: {device}')

    try:
        num_classes = 3
        input_height = 512
        input_width = 512
        model = CNN_Net(num_classes, input_height, input_width).to(device)        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        early_stopping_patience = 3
        epochs_without_improvement = 0

        best_recall = 0.0

        for epoch in range(num_epochs):
            my_logger.info(f'Starting epoch {epoch + 1}')

            # Training phase
            model.train()
            total_loss = 0
            correct_predictions = 0
            total_samples = 0

            for i, (inputs, _, labels) in enumerate(train_dataset_loader):
                my_logger.info(f'Batch {i} shape: {inputs.size()}')
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)

                if i % 5 == 0:
                    batch_accuracy = (predictions == labels).float().mean().item()
                    my_logger.info(f'Batch [{i+1}/{len(train_dataset_loader)}], Loss: {loss.item()}, Accuracy: {batch_accuracy}')
                    mlflow.log_metrics({'running_train_loss': loss.item(), 'running_train_accuracy': batch_accuracy}, step=epoch * len(train_dataset_loader) + i)

            train_loss = total_loss / total_samples
            train_accuracy = correct_predictions / total_samples
            mlflow.log_metrics({'train_loss': train_loss, 'train_accuracy': train_accuracy}, step=epoch)
            my_logger.info(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss}, Training Accuracy: {train_accuracy}')

            # Validation phase
            model.eval()
            with torch.no_grad():
                val_loss = 0
                correct = 0
                total = 0
                all_labels = []
                all_probabilities = []
                
                for inputs, _, labels in val_dataset_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    probabilities = torch.softmax(outputs, dim=1)
                    predicted = probabilities.argmax(dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())

                val_loss /= len(val_dataset_loader)
                val_accuracy = 100 * correct / total

                # Adjust metrics calculations to handle missing classes
                recall = recall_score(all_labels, np.array(all_probabilities).argmax(axis=1), average='macro', zero_division=0)
                precision = precision_score(all_labels, np.array(all_probabilities).argmax(axis=1), average='macro', zero_division=0)
                f1 = f1_score(all_labels, np.array(all_probabilities).argmax(axis=1), average='macro', zero_division=0)
                
                # Handle missing classes in AUC calculation
                try:
                    auc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr')
                except ValueError as e:
                    my_logger.error(f'Error calculating AUC: {e}')
                    auc = 0  # or handle as appropriate

                my_logger.info(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, '
                               f'Recall: {recall:.2f}, Precision: {precision:.2f}, F1 Score: {f1:.2f}, AUC: {auc:.2f}')

                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
                mlflow.log_metric("val_recall", recall, step=epoch)
                mlflow.log_metric("val_precision", precision, step=epoch)
                mlflow.log_metric("val_f1_score", f1, step=epoch)
                mlflow.log_metric("val_auc", auc, step=epoch)

                if recall > best_recall:
                    best_recall = recall
                    epochs_without_improvement = 0
                    torch.save(model.state_dict(), f'./outputs/cnn_lstm_model.pth')
                    my_logger.info(f'New best model saved with recall: {recall:.2f}')
                else:
                    epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                my_logger.info('Early stopping triggered')
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

    # get command-line arguments
    my_logger.info("Parsing command-line arguments")
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument("--num_epochs", type=int, help="number of epochs to train")
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--learning_rate", type=float, help="learning rate")
    parser.add_argument("--k", type=int, help="number of folds for cross-validation")
    parser.add_argument("--i", type=int, help="current fold index (0-based)")
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--run_cloud', action='store_true', help='Flag to indicate whether to run in cloud mode')

    args = parser.parse_args()
    my_logger.info(f"Arguments parsed: {args}")

    my_logger.info("Setting dataset variables")
    dataset = args.dataset

    # Determine dataset folder and whether to download from blob
    if not args.run_cloud:
        my_logger.info(f"Running in local mode, setting dataset folder to 'data/raw/{dataset}'")
        dataset_folder = f"data/raw/{dataset}"
    else:
        my_logger.info("Running in cloud mode, downloading dataset from blob storage")
        dataset_folder = dataset
        # get storage parameters
        load_dotenv()
        storage_account = os.getenv('AZURE_STORAGE_ACCOUNT')
        storage_account_key = os.getenv('AZURE_STORAGE_KEY')
        container_name = os.getenv('BLOB_CONTAINER')
        my_logger.info(f"Downloading dataset from blob: storage_account={storage_account}, container_name={container_name}")
        download_from_blob(storage_account, storage_account_key, container_name, dataset_folder)

    my_logger.info("Loading dataset")
    my_dataset = CCCCIIDataset2D(dataset_folder)
    my_logger.info("Dataset loaded")

    # Group by patient
    my_logger.info("Extracting patient IDs and labels")
    patient_ids = [sample[1] for sample in my_dataset]
    labels = [sample[2] for sample in my_dataset]  # Extract labels

    if args.i < 0 or args.i >= args.k:
        my_logger.error(f"Invalid fold index 'i': {args.i}. It must be between 0 and {args.k - 1}.")
        raise ValueError(f"Fold index 'i' must be between 0 and {args.k - 1}, but got {args.i}.")

    my_logger.info(f"Performing Stratified Group K-Fold with {args.k} splits")
    sgkf = StratifiedGroupKFold(n_splits=args.k, shuffle=True, random_state=42)
    splits = list(sgkf.split(np.zeros(len(my_dataset)), labels, groups=patient_ids))

    train_idx, val_idx = splits[args.i]
    my_logger.info(f"Train index: {train_idx[:5]}... ({len(train_idx)} samples), Val index: {val_idx[:5]}... ({len(val_idx)} samples)")

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
        learning_rate=args.learning_rate
    )
    my_logger.info("Model training completed")

    mlflow.end_run()
    my_logger.info("MLflow run ended")


if __name__ == "__main__":
    main()
