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

from datasets.ccccii_dataset import CCCCIIDataset
from utils.download import download_from_blob
from utils.log_config import get_custom_logger

my_logger = get_custom_logger('train_cnn_lstm_multiclass')

class CNN_LSTM_Net(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(CNN_LSTM_Net, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),  # Dropout after pooling
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),  # Dropout after pooling
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),  # Dropout after pooling
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout before the fully connected layer
        self.fc = nn.Linear(64, 3)
        self.softmax = nn.Softmax(dim=1)  

    def forward(self, x):
        my_logger.info(f'Input shape before reshaping: {x.size()}')
        
        if len(x.size()) != 5:
            raise ValueError(f"Expected 5D input (batch_size, channels, depth, height, width), got {x.size()}")

        batch_size, channels, depth, height, width = x.size()
        x = x.float()
        x = self.cnn(x)
        x = x.view(batch_size, -1, 128)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the output of the last time step
        lstm_out = self.dropout(lstm_out)  # Apply dropout
        x = self.fc(lstm_out)
        x = self.softmax(x)
        return x


def train_model(train_dataset_loader, val_dataset_loader, num_epochs, learning_rate):
    start_time = time.time()  # Start time of training
    my_logger.info('Starting Training')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_logger.info(f'Using device: {device}')

    try:
        model = CNN_LSTM_Net().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Added weight_decay for L2 regularization

        # Define early stopping parameters
        early_stopping_patience = 3  # Number of epochs to wait for improvement before stopping
        epochs_without_improvement = 0  # Counter for epochs without improvement

        best_recall = 0.0

        for epoch in range(num_epochs):
            my_logger.info(f'Starting epoch {epoch + 1}')

            # Training phase
            model.train()

            # initialize training metrics accumulators
            total_loss = 0
            correct_predictions = 0
            total_samples = 0

            for i, (inputs, labels) in enumerate(train_dataset_loader):
                my_logger.info(f'Batch {i} shape: {inputs.size()}')
                inputs, labels = inputs.to(device), labels.to(device)
                # Remove the following line as the channel dimension is already present in the dataset
                # inputs = inputs.unsqueeze(1)  # This line is causing the extra dimension

                # Forward pass: Compute predicted y by passing x to the model
                outputs = model(inputs)


                # Compute and print loss
                loss = criterion(outputs, labels)
                
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Compute training metrics
                total_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)                

                # Log training metrics every 10 mini-batches for monitoring
                if i % 5 == 0:
                    batch_accuracy = (predictions == labels).float().mean().item()
                    my_logger.info(f'Batch [{i+1}/{len(train_dataset_loader)}], Loss: {loss.item()}, Accuracy: {batch_accuracy}')
                    mlflow.log_metrics({'running_train_loss': loss.item(), 'running_train_accuracy': batch_accuracy}, step=epoch * len(train_dataset_loader) + i)

            # Calculate and log epoch-level averages
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
                
                for inputs, labels in val_dataset_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    # inputs = inputs.unsqueeze(1)  

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    probabilities = outputs.softmax(dim=1)  # Get probabilities
                    predicted = probabilities.argmax(dim=1)  # Convert probabilities to predictions
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # Collect all labels and probabilities for metrics calculation
                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())  # Store probabilities

                # Calculate validation metrics
                val_loss /= len(val_dataset_loader)
                val_accuracy = 100 * correct / total
                recall = recall_score(all_labels, np.array(all_probabilities).argmax(axis=1), average='macro')
                precision = precision_score(all_labels, np.array(all_probabilities).argmax(axis=1), average='macro')
                f1 = f1_score(all_labels, np.array(all_probabilities).argmax(axis=1), average='macro')
                auc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr')

                my_logger.info(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, '
                               f'Recall: {recall:.2f}, Precision: {precision:.2f}, F1 Score: {f1:.2f}')

                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
                mlflow.log_metric("val_recall", recall, step=epoch)
                mlflow.log_metric("val_precision", precision, step=epoch)
                mlflow.log_metric("val_f1_score", f1, step=epoch)
                mlflow.log_metric("val_auc", auc, step=epoch)

                # Check for improvement
                if recall > best_recall:
                    best_recall = recall
                    epochs_without_improvement = 0  # Reset counter
                    # Save the model
                    torch.save(model.state_dict(), f'./outputs/cnn_lstm_model.pth')
                    my_logger.info(f'New best model saved with recall: {recall:.2f}')
                else:
                    epochs_without_improvement += 1  # Increment counter

            # Early stopping check
            if epochs_without_improvement >= early_stopping_patience:
                my_logger.info('Early stopping triggered')
                break  # Break out of the training loop

        my_logger.info('Finished Training')
        training_time = time.time() - start_time  # Calculate total training time
        my_logger.info(f'Training completed in {training_time:.2f} seconds')

    except Exception as e:
        my_logger.error("Error during training: %s", str(e))  # Updated logging to properly handle arguments
        my_logger.error("Detailed traceback:")
        my_logger.error(traceback.format_exc())


def main():
    my_logger.info(f"Torch version: {torch.__version__}")

    # get command-line arguments
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument("--num_epochs", type=int, help="number of epochs to train")
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--learning_rate", type=float, help="learning rate")
    parser.add_argument("--k", type=int, help="number of folds for cross-validation")
    parser.add_argument("--i", type=int, help="current fold index (0-based)")
    parser.add_argument('--dataset', type=str, help='Dataset name')
    args = parser.parse_args()


    my_logger.info(
        f'dataset: {args.dataset}, '
        f'k: {args.k}, '
        f'i: {args.i}, '
        f'num_epochs: {args.num_epochs}, '
        f'batch_size: {args.batch_size}, '
        f'learning_rate: {args.learning_rate}'
    )

    dataset = args.dataset

    # Start Run
    mlflow.start_run()

    # get storage parameters
    load_dotenv()
    storage_account = os.getenv('AZURE_STORAGE_ACCOUNT')
    storage_account_key = os.getenv('AZURE_STORAGE_KEY')
    container_name = os.getenv('BLOB_CONTAINER')
    
    dataset_folder = dataset
    download_from_blob(storage_account, storage_account_key, container_name, dataset_folder)

    my_dataset = CCCCIIDataset(args.dataset)

    # Extract labels from the dataset
    labels = [sample[1] for sample in my_dataset]  # Assuming my_dataset[i] returns (input, label)

    # Prepare Stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)
    splits = list(skf.split(np.zeros(len(my_dataset)), labels))  # Pass dummy X (np.zeros) and labels

    train_idx, val_idx = splits[args.i]

    train_dataset = Subset(my_dataset, train_idx)
    val_dataset = Subset(my_dataset, val_idx)

    train_dataset_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    train_model(
        train_dataset_loader=train_dataset_loader,
        val_dataset_loader=val_dataset_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )
    mlflow.end_run()

if __name__ == "__main__":
    main()

# To run this script, use the following command:
# python -m scripts.train.train_cnn_lstm --num_epochs <NUM_EPOCHS> --batch_size <BATCH_SIZE> --learning_rate <LEARNING_RATE> --k <NUM_FOLDS> --i <CURRENT_FOLD> --dataset <DATASET1> ...
