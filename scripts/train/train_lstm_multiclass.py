import argparse
import os
import time
import traceback
from dotenv import load_dotenv
import os
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

from datasets import CCCCIIDatasetSequence2D
from utils.log_config import get_custom_logger
from utils.download import download_from_blob, download_from_blob_with_access_key

my_logger = get_custom_logger('train_lstm_multiclass')


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


def train_model(train_loader, val_loader, num_epochs, learning_rate, cnn_model_path):
    start_time = time.time()
    my_logger.info('Starting Training')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_logger.info(f'Using device: {device}')

    try:
        # Inicializar e carregar o modelo CNN pré-treinado
        num_classes = 3
        input_height = 512
        input_width = 512
        dropout_rate = 0.5
        cnn_model = CNN_Net(num_classes, input_height, input_width, dropout_rate=dropout_rate).to(device)    
        cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
        cnn_model.eval()
        cnn_model.to(device)
        for param in cnn_model.parameters():
            param.requires_grad = False

        # Inicializar o modelo LSTM
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

            # Fase de treinamento
            lstm_model.train()

            # Inicializar acumuladores de métricas de treinamento
            total_loss = 0
            correct_predictions = 0
            total_samples = 0

            for i, (inputs, labels) in enumerate(train_loader):
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

                if i % 5 == 0:
                    batch_accuracy = (predictions == labels).float().mean().item()
                    my_logger.info(f'Batch [{i+1}/{len(train_loader)}], Loss: {loss.item()}, Accuracy: {batch_accuracy}')
                    mlflow.log_metrics({'running_train_loss': loss.item(), 'running_train_accuracy': batch_accuracy}, step=epoch * len(train_loader) + i)

            # Calcular e registrar médias de treinamento por época
            train_loss = total_loss / total_samples
            train_accuracy = correct_predictions / total_samples
            mlflow.log_metrics({'train_loss': train_loss, 'train_accuracy': train_accuracy}, step=epoch)
            my_logger.info(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss}, Training Accuracy: {train_accuracy}')

            # Fase de validação
            my_logger.info(f'Starting validation phase for epoch {epoch + 1}')
            lstm_model.eval()
            with torch.no_grad():
                val_loss = 0
                correct = 0
                total = 0
                all_labels = []
                all_probabilities = []

                for j, (inputs, labels) in enumerate(val_loader):
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

                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())

            # Calcular métricas de validação
            val_loss /= total
            val_accuracy = 100.0 * correct / total
            val_recall = recall_score(all_labels, np.argmax(all_probabilities, axis=1), average='macro', zero_division=0)
            val_precision = precision_score(all_labels, np.argmax(all_probabilities, axis=1), average='macro', zero_division=0)
            val_f1 = f1_score(all_labels, np.argmax(all_probabilities, axis=1), average='macro', zero_division=0)
            try:
                val_auc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr')
            except ValueError:
                val_auc = 0.0

            my_logger.info(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, '
                           f'Recall: {val_recall:.2f}, Precision: {val_precision:.2f}, F1 Score: {val_f1:.2f}, AUC: {val_auc:.2f}')

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
                file_prefix = f'lstm_multiclass_{total_samples}smps_{epoch + 1:03}epoch_{learning_rate:.5f}lr_{val_recall:.3f}rec'
                file_name = f'{file_prefix}.pth'
                torch.save(lstm_model.state_dict(), f'{output_dir}/{file_name}')
                my_logger.info(f'New best model saved with recall: {val_recall:.3f}')

                # Generate and save confusion matrix
                cm = confusion_matrix(all_labels, np.array(all_probabilities).argmax(axis=1))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap=plt.cm.Blues)
                confusion_matrix_file = f'{output_dir}/{file_prefix}_confusion_matrix.png'
                plt.savefig(confusion_matrix_file)

            else:
                epochs_without_improvement += 1


            # Early stopping
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

    # Obter argumentos de linha de comando
    parser = argparse.ArgumentParser(description='Train a CNN-LSTM multiclass model')
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

    my_logger.info("Setting dataset variables")
    dataset = args.dataset 

    if not args.run_cloud:        
        my_logger.info(f"Running in local mode, no need to download dataset and pre-trained model")        
        dataset_folder = f'data/{dataset}'   

    else:
        my_logger.info("Running in cloud mode, downloading dataset and pre-trained model from blob storage")
        load_dotenv()

        try:
            dataset_folder = dataset
            storage_account = os.getenv('AZURE_STORAGE_ACCOUNT')
            storage_account_key = os.getenv('AZURE_STORAGE_KEY')
            container_name = os.getenv('BLOB_CONTAINER')
            my_logger.info(f"Downloading dataset from blob: storage_account={storage_account}, container_name={container_name}")
            download_from_blob(storage_account, storage_account_key, container_name, dataset_folder)
        except Exception as download_err:
            my_logger.error(f"Failed to download dataset from storage account: {str(download_err)}")

        try:
            model_uri= os.getenv('PRETRAINED_CNN_MODEL_URI')        
            my_logger.info(f"Downloading pre-trained model from blob: storage_account={storage_account}, pre-trained model={model_uri}")
            download_from_blob_with_access_key(model_uri, storage_account_key, args.cnn_model_path)
            my_logger.info(f"Model downloaded from AzureML to {args.cnn_model_path}")
        except Exception as download_err:
            my_logger.error(f"Failed to download model from storage account: {str(download_err)}")

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