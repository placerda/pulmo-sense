# File: scripts/train/train_vgg_multiclass.py

import argparse
import os
import random
import time
import traceback

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Subset

from dotenv import load_dotenv
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedGroupKFold

import matplotlib.pyplot as plt

from datasets.ccccii_dataset import CCCCIIDataset2D
from utils.download import download_from_blob
from utils.log_config import get_custom_logger

my_logger = get_custom_logger('train_vgg_multiclass')

class VGG_Net(nn.Module):
    """
    A simple wrapper around VGG from torchvision.
    We adapt it to single-channel input by repeating the grayscale channel to 3 channels,
    and then replace the final classifier to match our number of classes = 3.
    """
    def __init__(self, num_classes=3):
        super(VGG_Net, self).__init__()
        # Load a pretrained VGG16, for instance
        self.model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        
        # Adjust the first convolution layer to accept 3 channels 
        # (since we will replicate grayscale to [1 -> 3] below, we keep the default 3).
        
        # Replace the last classifier layer:
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        # x is shape [batch_size, 1, 512, 512].
        # Convert 1-channel to 3-channels by repeating the dimension
        x = x.repeat(1, 3, 1, 1)  # Now shape [batch_size, 3, 512, 512].
        return self.model(x)

def train_model(train_loader, val_loader, num_epochs, learning_rate):
    start_time = time.time()
    my_logger.info('Starting VGG multiclass training')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_logger.info(f'Using device: {device}')

    try:
        model = VGG_Net(num_classes=3).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        early_stopping_patience = 3
        epochs_without_improvement = 0
        best_recall = 0.0

        for epoch in range(num_epochs):
            my_logger.info(f'=== Epoch {epoch+1}/{num_epochs} ===')

            # ========== TRAINING ========== 
            model.train()
            total_loss, correct, total = 0, 0, 0

            for i, (inputs, _, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                if i % 50 == 0:
                    batch_acc = (preds == labels).float().mean().item()
                    my_logger.info(f'[Train] Batch {i}/{len(train_loader)} - Loss: {loss.item():.4f}, Acc: {batch_acc:.4f}')
            
            train_loss = total_loss / total
            train_acc = 100.0 * correct / total

            my_logger.info(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%')
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)

            # ========== VALIDATION ========== 
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            all_labels, all_probs = [], []

            with torch.no_grad():
                for j, (inputs, _, labels) in enumerate(val_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * labels.size(0)
                    probs = torch.softmax(outputs, dim=1)
                    preds = probs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            val_loss /= val_total
            val_acc = 100.0 * val_correct / val_total
            val_preds = np.argmax(all_probs, axis=1)

            val_recall = recall_score(all_labels, val_preds, average='macro', zero_division=0)
            val_precision = precision_score(all_labels, val_preds, average='macro', zero_division=0)
            val_f1 = f1_score(all_labels, val_preds, average='macro', zero_division=0)
            try:
                val_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
            except ValueError:
                val_auc = 0.0

            my_logger.info(
                f'Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%, Recall={val_recall:.2f}, '
                f'Precision={val_precision:.2f}, F1={val_f1:.2f}, AUC={val_auc:.2f}'
            )

            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            mlflow.log_metric("val_recall", val_recall, step=epoch)
            mlflow.log_metric("val_precision", val_precision, step=epoch)
            mlflow.log_metric("val_f1_score", val_f1, step=epoch)
            mlflow.log_metric("val_auc", val_auc, step=epoch)

            # Check for best model
            if val_recall > best_recall:
                best_recall = val_recall
                epochs_without_improvement = 0
                
                # Save model
                os.makedirs("outputs", exist_ok=True)
                model_path = f"outputs/vgg_best_epoch_{epoch+1}_{val_recall:.3f}_rec.pth"
                torch.save(model.state_dict(), model_path)
                my_logger.info(f"New best model saved at epoch {epoch+1} with recall={val_recall:.3f}")

                # Confusion Matrix
                cm = confusion_matrix(all_labels, val_preds)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap=plt.cm.Blues)
                plt.savefig(f"outputs/confusion_matrix_epoch_{epoch+1}.png")
                plt.close()

            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                my_logger.info("Early stopping triggered.")
                break

        my_logger.info("Finished Training VGG model.")
        my_logger.info(f"Total training time: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        my_logger.error("Error during training: %s", str(e))
        my_logger.error(traceback.format_exc())

def main():
    my_logger.info(f"Torch version: {torch.__version__}")

    parser = argparse.ArgumentParser(description="Train VGG multiclass model")
    parser.add_argument("--num_epochs", type=int, help="number of epochs to train")
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--learning_rate", type=float, help="learning rate")
    parser.add_argument("--k", type=int, help="number of folds for cross-validation")
    parser.add_argument("--i", type=int, help="current fold index (0-based)")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--run_cloud", action='store_true', help="Flag to indicate cloud mode")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use")

    args = parser.parse_args()
    my_logger.info(f"Args: {args}")

    # Possibly start an MLflow run here
    mlflow.start_run()

    if not args.run_cloud:
        dataset_folder = f"data/raw/{args.dataset}"
        my_logger.info("Local mode. dataset_folder=%s", dataset_folder)
    else:
        dataset_folder = args.dataset
        load_dotenv()
        storage_account = os.getenv("AZURE_STORAGE_ACCOUNT")
        storage_key = os.getenv("AZURE_STORAGE_KEY")
        container_name = os.getenv("BLOB_CONTAINER")
        my_logger.info("Cloud mode. Downloading data from blob.")
        download_from_blob(storage_account, storage_key, container_name, dataset_folder)

    # Prepare dataset
    my_dataset = CCCCIIDataset2D(dataset_folder, max_samples=args.max_samples)
    patient_ids = [sample[1] for sample in my_dataset]  # patient_id
    labels = [sample[2] for sample in my_dataset]

    # Cross-validation or holdout
    sgkf = StratifiedGroupKFold(n_splits=args.k, shuffle=True, random_state=42)
    splits = list(sgkf.split(np.zeros(len(my_dataset)), labels, groups=patient_ids))

    train_idx, val_idx = splits[args.i]
    train_subset = Subset(my_dataset, train_idx)
    val_subset = Subset(my_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )

    mlflow.end_run()

if __name__ == "__main__":
    main()
