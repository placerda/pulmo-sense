#!/usr/bin/env python
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import logging
import torchvision.models as models
from utils.download import download_from_blob

# Import Mosmed dataset (ensure the path is correct)
from datasets.raster_dataset import Dataset2DBinary

# -------------------------
# Logger Setup
# -------------------------
def get_custom_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = get_custom_logger('train_dynamic_vgg')

# -------------------------
# Dynamic Multi-Kernel Convolution Module
# -------------------------
class DynamicMultiKernelBlock(nn.Module):
    """
    This block applies parallel convolutions with different kernel sizes and uses a gating mechanism 
    to dynamically combine the outputs.
    """
    def __init__(self, channels, kernel_sizes=[3, 5, 7], reduction=16):
        super(DynamicMultiKernelBlock, self).__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            padding = k // 2  # Keeps spatial dimensions the same
            self.branches.append(nn.Conv2d(channels, channels, kernel_size=k, padding=padding))
        # Gating mechanism: perform global pooling and generate weights for each branch
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, len(kernel_sizes)),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        weights = self.gate(x)  # [batch, num_branches]
        out = 0
        for i, branch in enumerate(self.branches):
            branch_out = branch(x)
            weight = weights[:, i].view(x.size(0), 1, 1, 1)
            out = out + weight * branch_out
        return out

# -------------------------
# New Architecture: DynamicKernelVGG
# -------------------------
class DynamicKernelVGG(nn.Module):
    """
    Modifies the standard VGG16 by inserting a DynamicMultiKernelBlock after each pooling layer,
    allowing dynamic adjustment of the receptive field.
    """
    def __init__(self, num_classes=2):
        super(DynamicKernelVGG, self).__init__()
        vgg = models.vgg16_bn(pretrained=True)
        features = list(vgg.features)
        new_features = []
        block = 0
        for layer in features:
            new_features.append(layer)
            if isinstance(layer, nn.MaxPool2d):
                # Insert dynamic convolution block after each pooling layer
                if block == 0:
                    channels = 64
                elif block == 1:
                    channels = 128
                elif block == 2:
                    channels = 256
                elif block == 3:
                    channels = 512
                elif block == 4:
                    channels = 512
                new_features.append(DynamicMultiKernelBlock(channels))
                block += 1
        self.features = nn.Sequential(*new_features)
        self.avgpool = vgg.avgpool
        in_features = vgg.classifier[6].in_features
        classifier = list(vgg.classifier)
        classifier[-1] = nn.Linear(in_features, num_classes)
        self.classifier = nn.Sequential(*classifier)
    
    def forward(self, x):
        # If input is 1 channel, repeat it to simulate 3 channels (RGB)
        x = x.repeat(1, 3, 1, 1)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# -------------------------
# Evaluation Function
# -------------------------
def evaluate(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            # Assume dataset returns (input, _, label)
            inputs, _, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    try:
        positive_probs = [p[0] for p in all_probs]
        fpr, tpr, _ = roc_curve(all_labels, positive_probs, pos_label=0)
        auc_val = auc(fpr, tpr)
    except Exception as e:
        logger.error(f"Error computing AUC: {e}")
        auc_val = 0.0
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    return acc, auc_val, f1, precision, recall, all_labels, all_preds

# -------------------------
# Training Loop
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Train DynamicKernelVGG on the dataset")
    parser.add_argument("--dataset", type=str, default="ccccii", help="Path to the dataset")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model_save_path", type=str, required=True, help="Path to save the best model weights")
    args = parser.parse_args()
    
    load_dotenv()
    dataset_folder = args.dataset
    load_dotenv()
    storage_account = os.getenv("AZURE_STORAGE_ACCOUNT")
    storage_key = os.getenv("AZURE_STORAGE_KEY")
    container_name = os.getenv("BLOB_CONTAINER")
    logger.info("Cloud mode. Downloading data from blob.")
    download_from_blob(storage_account, storage_key, container_name, dataset_folder)

    logger.info("Loading dataset from folder: %s", dataset_folder)
    dataset = Dataset2DBinary(root_dir=dataset_folder)
    
    # Split dataset: 80% training, 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DynamicKernelVGG(num_classes=2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            inputs, _, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        acc, auc_val, f1, precision, recall, _, _ = evaluate(model, val_loader, device)
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f} - Val Acc: {acc*100:.2f}% - AUC: {auc_val:.4f} - F1: {f1:.4f}")
        
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), args.model_save_path)
            logger.info(f"New best model saved at epoch {epoch+1} with accuracy {acc*100:.2f}%")
    
    logger.info("Training complete.")
    # Generate and save the confusion matrix
    _, _, _, _, _, all_labels, all_preds = evaluate(model, val_loader, device)
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", pad=20)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    fig.colorbar(cax)
    plt.savefig("confusion_matrix_dynamic_vgg.png", bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix saved as confusion_matrix_dynamic_vgg.png")
    
if __name__ == "__main__":
    main()
