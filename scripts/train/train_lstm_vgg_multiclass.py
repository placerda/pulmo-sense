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
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import random

from datasets import CCCCIIDatasetSequence2D
from utils.log_config import get_custom_logger
from utils.download import download_from_blob, download_from_blob_with_access_key

my_logger = get_custom_logger('train_lstm_vgg_multiclass')


# -----------------------------------------------------------------
# 1) VGG Feature Extractor
# -----------------------------------------------------------------
class VGGFeatureExtractor(nn.Module):
    """
    A class that uses VGG16 (with BatchNorm) to extract embeddings.
    We load the pretrained weights from your train_vgg_multiclass.py
    and replace the final classifier layer with an identity layer,
    so we get the high-level feature vector (e.g., shape 4096).
    """

    def __init__(self, vgg_weights_path):
        super(VGGFeatureExtractor, self).__init__()
        # Load a pretrained VGG16 with batch norm
        # (we assume you trained it starting from the official ImageNet weights)
        self.vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)

        # Replace the last classification layer with Identity so we get "features"
        in_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Identity()

        # Load your fine-tuned weights from train_vgg_multiclass.py
        # (Strict=False is often needed if the shapes changed)
        state_dict = torch.load(vgg_weights_path, map_location='cpu')
        self.vgg.load_state_dict(state_dict, strict=False)

        # We'll store the embedding dimension
        self.embedding_dim = in_features  # Usually 4096 for VGG16_BN

    def forward(self, x):
        """
        Args:
          x: [batch_size, 1, 512, 512]  (single-channel input)
        Returns:
          features: [batch_size, embedding_dim] (e.g., 4096)
        """
        # Repeat the channel dimension to 3, because VGG expects 3-channel RGB
        x = x.repeat(1, 3, 1, 1)
        # VGG expects roughly (3, 224, 224); if your images are 512x512,
        # you can either crop/resize here or rely on the dataset to do so.
        # If you have 512x512, you must do:
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # Pass through VGG up to the penultimate layer
        features = self.vgg(x)  # shape [batch_size, 4096]
        return features


# -----------------------------------------------------------------
# 2) LSTM Model (same as your original or simplified version)
# -----------------------------------------------------------------
class LSTM_Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
        """
        The same LSTM you used in train_lstm_multiclass.py
        """
        super(LSTM_Net, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        """
        x shape: [batch_size, seq_len, input_size]
        We get the final hidden state from the LSTM
        and pass it through a fully connected layer.
        """
        out, _ = self.lstm(x)       # out: [batch_size, seq_len, hidden_size]
        out = out[:, -1, :]         # Take last time-step: [batch_size, hidden_size]
        out = self.dropout(out)
        out = self.fc(out)          # [batch_size, num_classes]
        return out


# -----------------------------------------------------------------
# 3) Training Function
# -----------------------------------------------------------------
def train_model(train_loader, val_loader, num_epochs, learning_rate, vgg_model_path):
    start_time = time.time()
    my_logger.info('Starting Training using pretrained VGG features + LSTM')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_logger.info(f'Using device: {device}')

    try:
        # ------------------------------------
        # a) Initialize the VGG feature extractor
        # ------------------------------------
        vgg_extractor = VGGFeatureExtractor(vgg_weights_path=vgg_model_path).to(device)
        vgg_extractor.eval()  # We'll freeze it so it doesn't train further
        for param in vgg_extractor.parameters():
            param.requires_grad = False

        # ------------------------------------
        # b) Initialize the LSTM
        # ------------------------------------
        num_classes = 3
        input_size = vgg_extractor.embedding_dim  # typically 4096 from VGG16_BN
        hidden_size = 128
        num_layers = 1
        dropout_rate = 0.5

        lstm_model = LSTM_Net(input_size, hidden_size, num_layers, num_classes, dropout_rate).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate, weight_decay=1e-5)

        best_recall = 0.0
        early_stopping_patience = 3
        epochs_without_improvement = 0

        # ------------------------------------
        # c) Main Training Loop
        # ------------------------------------
        for epoch in range(num_epochs):
            my_logger.info(f'Starting epoch {epoch + 1}/{num_epochs}')
            lstm_model.train()

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
                    features = vgg_extractor(inputs)

                # Reshape => [batch_size, seq_len, 4096]
                features = features.view(batch_size, seq_len, -1)

                # Forward pass in LSTM => [batch_size, num_classes]
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
                    my_logger.info(f'Batch [{i+1}/{len(train_loader)}], '
                                   f'Loss: {loss.item():.4f}, Accuracy: {batch_accuracy:.4f}')
                    mlflow.log_metrics({
                        'running_train_loss': loss.item(),
                        'running_train_accuracy': batch_accuracy
                    }, step=epoch * len(train_loader) + i)

            train_loss = total_loss / total_samples
            train_accuracy = correct_predictions / total_samples
            mlflow.log_metrics({'train_loss': train_loss, 'train_accuracy': train_accuracy}, step=epoch)

            my_logger.info(f'Epoch [{epoch+1}/{num_epochs}], '
                           f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')

            # ------------------------------------
            # d) Validation
            # ------------------------------------
            lstm_model.eval()
            val_loss = 0
            correct = 0
            total = 0
            all_labels = []
            all_probabilities = []

            with torch.no_grad():
                for j, (inputs, labels) in enumerate(val_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    batch_size, seq_len, channels, height, width = inputs.size()

                    # Flatten => [batch_size * seq_len, channels, height, width]
                    inputs = inputs.view(batch_size * seq_len, channels, height, width)

                    # Extract VGG features
                    features = vgg_extractor(inputs)
                    # Reshape => [batch_size, seq_len, 4096]
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

            val_loss /= total
            val_accuracy = 100.0 * correct / total
            val_preds = np.argmax(all_probabilities, axis=1)

            val_recall = recall_score(all_labels, val_preds, average='macro', zero_division=0)
            val_precision = precision_score(all_labels, val_preds, average='macro', zero_division=0)
            val_f1 = f1_score(all_labels, val_preds, average='macro', zero_division=0)
            try:
                val_auc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr')
            except ValueError:
                val_auc = 0.0

            my_logger.info(
                f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%, '
                f'Recall: {val_recall:.2f}, Precision: {val_precision:.2f}, '
                f'F1: {val_f1:.2f}, AUC: {val_auc:.2f}'
            )

            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
            mlflow.log_metric("val_recall", val_recall, step=epoch)
            mlflow.log_metric("val_precision", val_precision, step=epoch)
            mlflow.log_metric("val_f1_score", val_f1, step=epoch)
            mlflow.log_metric("val_auc", val_auc, step=epoch)

            # ------------------------------------
            # e) Check for best model
            # ------------------------------------
            if val_recall > best_recall:
                best_recall = val_recall
                epochs_without_improvement = 0
                output_dir = './outputs'
                os.makedirs(output_dir, exist_ok=True)
                file_prefix = f'lstm_vgg_{total}smps_{epoch + 1:03}epoch_{learning_rate:.5f}lr_{val_recall:.3f}rec'
                file_name = f'{file_prefix}.pth'
                torch.save(lstm_model.state_dict(), f'{output_dir}/{file_name}')

                my_logger.info(f'New best model saved with recall: {val_recall:.3f}')

                # Generate confusion matrix
                cm = confusion_matrix(all_labels, val_preds)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap=plt.cm.Blues)
                confusion_matrix_file = f'{output_dir}/{file_prefix}_confusion_matrix.png'
                plt.savefig(confusion_matrix_file)
                plt.close()
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                my_logger.info(
                    f'Early stopping triggered. Last Recall: {val_recall:.2f}, '
                    f'Last Precision: {val_precision:.2f}, Last Accuracy: {val_accuracy:.2f}%'
                )
                break

        my_logger.info('Finished Training LSTM + VGG features')
        training_time = time.time() - start_time
        my_logger.info(f'Training completed in {training_time:.2f} seconds')

    except Exception as e:
        my_logger.error("Error during training: %s", str(e))
        my_logger.error("Detailed traceback:")
        my_logger.error(traceback.format_exc())


# -----------------------------------------------------------------
# 4) Main Function
# -----------------------------------------------------------------
def main():
    my_logger.info(f"Torch version: {torch.__version__}")
    mlflow.start_run()
    my_logger.info("MLflow run started")

    parser = argparse.ArgumentParser(description='Train an LSTM model using VGG features for multiclass classification')
    parser.add_argument("--num_epochs", type=int, default=20, help="number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="learning rate")
    parser.add_argument("--k", type=int, default=5, help="number of folds for cross-validation")
    parser.add_argument("--i", type=int, default=0, help="current fold index (0-based)")
    parser.add_argument('--dataset', type=str, default='ccccii', help='Dataset name')
    parser.add_argument('--run_cloud', action='store_true', help='Flag to indicate whether to run in cloud mode')
    parser.add_argument('--max_samples', type=int, default=0, help='Maximum number of samples to use')
    parser.add_argument('--vgg_model_path', type=str, default='models/vgg_multiclass_best.pth',
                        help='Path to the pretrained VGG model weights')
    parser.add_argument('--sequence_length', type=int, default=30, help='Sequence length for LSTM input')

    args = parser.parse_args()
    my_logger.info(f"Arguments parsed: {args}")
    my_logger.info(f"Current Working Directory: {os.getcwd()}")

    if not args.run_cloud:
        my_logger.info("Running in local mode")
        dataset_folder = f"data/{args.dataset}"
    else:
        my_logger.info("Running in cloud mode, downloading dataset and pre-trained model from blob storage")
        load_dotenv()
        storage_account = os.getenv('AZURE_STORAGE_ACCOUNT')
        storage_account_key = os.getenv('AZURE_STORAGE_KEY')

        try:
            model_uri= os.getenv('PRETRAINED_VGG_MODEL_URI')        
            my_logger.info(f"Downloading pre-trained model from blob: storage_account={storage_account}, pre-trained model={model_uri}")
            os.makedirs(os.path.dirname(args.vgg_model_path), exist_ok=True)
            download_from_blob_with_access_key(model_uri, storage_account_key, args.vgg_model_path)
            my_logger.info(f"Model downloaded from blob to {args.vgg_model_path}")
        except Exception as download_err:
            my_logger.error(f"Failed to download model from storage account: {str(download_err)}")
            exit(-1)

        try:
            dataset_folder = args.dataset
            container_name = os.getenv('BLOB_CONTAINER')
            my_logger.info(f"Downloading dataset from blob: storage_account={storage_account}, container_name={container_name}")
            download_from_blob(storage_account, storage_account_key, container_name, dataset_folder)
        except Exception as download_err:
            my_logger.error(f"Failed to download dataset from storage account: {str(download_err)}")
            exit(-1)

    my_logger.info("Loading dataset")
    sequence_length = args.sequence_length
    my_dataset = CCCCIIDatasetSequence2D(dataset_folder, sequence_length=sequence_length, max_samples=args.max_samples)
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

    my_logger.info("Starting LSTM + VGG training pipeline")
    train_model(
        train_loader=train_dataset_loader,
        val_loader=val_dataset_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        vgg_model_path=args.vgg_model_path
    )
    my_logger.info("Model training completed")

    mlflow.end_run()
    my_logger.info("MLflow run ended")


if __name__ == "__main__":
    main()
