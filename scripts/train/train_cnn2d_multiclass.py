import argparse
import os
import random
import time
import traceback
from dotenv import load_dotenv
import matplotlib.pyplot as plt

import mlflow
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from datasets import Dataset2D
from utils.download import download_from_blob
from utils.log_config import get_custom_logger

my_logger = get_custom_logger('train_cnn2d_multiclass')


class CNN_Net(nn.Module):
    def __init__(self, num_classes, input_height=None, input_width=None, dropout_rate=0.5):
        super(CNN_Net, self).__init__()
        self.input_height = input_height
        self.input_width = input_width        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(dropout_rate),
        )
        self.embedding_dim = 128
        self.fc = nn.Linear(self.embedding_dim, num_classes)
        
    def forward(self, x, return_embedding=False):
        x = x.float()
        embedding = self.cnn(x)
        embedding = embedding.view(embedding.size(0), -1)
        
        if return_embedding:
            return embedding

        x = self.fc(embedding)
        return x


def train_model(train_dataset_loader, val_dataset_loader, num_epochs, learning_rate):
    start_time = time.time()
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

                if i % 100 == 0:
                    batch_accuracy = (predictions == labels).float().mean().item()
                    my_logger.info(f'Train batch [{i+1}/{len(train_dataset_loader)}], Loss: {loss.item()}, Accuracy: {batch_accuracy}')
                    mlflow.log_metrics({'running_train_loss': loss.item(), 'running_train_accuracy': batch_accuracy}, step=epoch * len(train_dataset_loader) + i)

            train_loss = total_loss / total_samples
            train_accuracy = correct_predictions / total_samples
            mlflow.log_metrics({'train_loss': train_loss, 'train_accuracy': train_accuracy}, step=epoch)
            my_logger.info(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss}, Training Accuracy: {train_accuracy}')

            # Validation phase
            my_logger.info(f'Starting validation phase for epoch {epoch + 1}')            
            model.eval()
            with torch.no_grad():
                val_loss = 0
                correct = 0
                total = 0
                all_labels = []
                all_probabilities = []
                
                for j, (inputs, _, labels) in enumerate(val_dataset_loader):           
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    probabilities = torch.softmax(outputs, dim=1)
                    predicted = probabilities.argmax(dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    if j % 100 == 0:
                        batch_accuracy = (predicted == labels).float().mean().item()
                        my_logger.info(f'Val batch [{j+1}/{len(val_dataset_loader)}], Loss: {loss.item()}, Accuracy: {batch_accuracy}')

                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())

                val_loss /= len(val_dataset_loader)
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
                file_prefix = f'cnn_multiclass_{total_samples}smps_{epoch + 1:03}epoch_{learning_rate:.5f}lr_{val_recall:.3f}rec'
                file_name = f'{file_prefix}.pth'
                torch.save(model.state_dict(), f'{output_dir}/{file_name}')
                my_logger.info(f'New best model saved with recall: {val_recall:.3f}')

                # Generate and save confusion matrix
                cm = confusion_matrix(all_labels, np.array(all_probabilities).argmax(axis=1))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap=plt.cm.Blues)
                confusion_matrix_file = f'{output_dir}/{file_prefix}_confusion_matrix.png'
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
    # Iniciar uma execução no MLflow
    mlflow.start_run()
    my_logger.info("MLflow run started")

    # obter argumentos de linha de comando
    my_logger.info("Parsing command-line arguments")
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument("--num_epochs", type=int, default=20, help="number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="learning rate")
    parser.add_argument("--k", type=int, default=5, help="number of folds for cross-validation")
    parser.add_argument("--i", type=int, default=0, help="current fold index (0-based)")
    parser.add_argument('--dataset', type=str, default='ccccii', help='Dataset name')
    parser.add_argument('--run_cloud', action='store_true', help='Flag to indicate whether to run in cloud mode')
    parser.add_argument('--max_samples', type=int, default=0, help='Maximum number of samples to use')

    args = parser.parse_args()
    my_logger.info(f"Arguments parsed: {args}")

    my_logger.info(f"Current Working Directory: {os.getcwd()}")

    my_logger.info("Setting dataset variables")
    dataset = args.dataset

    if not args.run_cloud:
        my_logger.info(f"Running in local mode, setting dataset folder to 'data/{dataset}'")
        dataset_folder = f"data/{dataset}"
    else:
        my_logger.info("Running in cloud mode, downloading dataset from blob storage")
        dataset_folder = args.dataset
        load_dotenv()
        storage_account = os.getenv('AZURE_STORAGE_ACCOUNT')
        storage_account_key = os.getenv('AZURE_STORAGE_KEY')
        container_name = os.getenv('BLOB_CONTAINER')
        my_logger.info(f"Downloading dataset from blob: storage_account={storage_account}, container_name={container_name}")
        download_from_blob(storage_account, storage_account_key, container_name, dataset_folder)

    my_logger.info("Loading dataset")
    my_dataset = Dataset2D(dataset_folder, max_samples=args.max_samples)
    my_logger.info(f"Dataset loaded with a maximum of {args.max_samples} samples")

    # Group by patient
    my_logger.info("Extracting patient IDs and labels")
    patient_ids = [sample[1] for sample in my_dataset]
    labels = [sample[2] for sample in my_dataset]  # Extract labels

    # Shuffle patients
    random.seed(42)
    combined = list(zip(patient_ids, labels))
    random.shuffle(combined)
    patient_ids, labels = zip(*combined)
    patient_ids = list(patient_ids)
    labels = list(labels)    

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
        learning_rate=args.learning_rate
    )

    my_logger.info("Model training completed")

    mlflow.end_run()
    my_logger.info("MLflow run ended")


if __name__ == "__main__":
    main()
