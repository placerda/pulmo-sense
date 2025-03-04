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
from datasets import CCCCIIDatasetSequence2D
from utils.download import download_from_blob, download_from_blob_with_access_key
from utils.log_config import get_custom_logger

my_logger = get_custom_logger('train_transformer_vgg_multiclass')

# ----------------------------------------------------------------------
# 1) VGG Feature Extractor (same as before)
# ----------------------------------------------------------------------
class VGGFeatureExtractor(nn.Module):
    def __init__(self, vgg_weights_path):
        super(VGGFeatureExtractor, self).__init__()
        self.vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        in_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Identity()  # remove final classifier
        state_dict = torch.load(vgg_weights_path, map_location='cpu')
        self.vgg.load_state_dict(state_dict, strict=False)
        self.embedding_dim = in_features  # typically 4096

    def forward(self, x):
        # x: [batch_size, 1, H, W]
        x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        features = self.vgg(x)  # shape [batch_size, 4096]
        return features

# ----------------------------------------------------------------------
# 2) A Small Transformer Encoder for Sequences
# ----------------------------------------------------------------------
class TransformerEncoderNet(nn.Module):
    """
    Replaces the LSTM with a small TransformerEncoder stack. We then pool
    across the sequence dimension to get a final embedding.
    """
    def __init__(self, input_size, num_classes, n_heads=4, num_layers=2, dim_feedforward=512, dropout=0.5):
        super(TransformerEncoderNet, self).__init__()

        # We embed each slice's 4096-dim vector into model_dim
        # If you want model_dim == input_size, that's fine. But let's keep it flexible.
        self.model_dim = 256  # You can choose e.g. 128, 256, etc.
        
        self.input_proj = nn.Linear(input_size, self.model_dim)

        # Positional encoding (learnable or fixed). For simplicity, we do a small fixed encoding
        self.positional_encoding = PositionalEncoding(self.model_dim, dropout=dropout)

        # Build a stack of TransformerEncoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # since our sequence is in [batch, seq, dim]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # final classification
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.model_dim, num_classes)

    def forward(self, x):
        """
        x shape: [batch_size, seq_len, input_size] e.g. [B, T, 4096]
        Returns:
          logits shape: [batch_size, num_classes]
        """
        # Project input to model_dim
        x = self.input_proj(x)  # [batch_size, seq_len, model_dim]

        # Add positional encodings
        x = self.positional_encoding(x)  # [batch_size, seq_len, model_dim]

        # Pass through Transformer encoders
        encoded = self.transformer_encoder(x)  # [batch_size, seq_len, model_dim]

        # We can simply pool (e.g. average) across seq_len
        # or we can take the CLS token approach or the last element
        # We'll do a global average pool:
        out = encoded.mean(dim=1)  # [batch_size, model_dim]

        out = self.dropout(out)
        logits = self.fc(out)  # [batch_size, num_classes]
        return logits


class PositionalEncoding(nn.Module):
    """
    Classic sine/cosine positional encoding, or you can do a learnable approach.
    For simplicity, let's do a standard sine/cosine (fixed).
    """
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # create a long enough PEx table
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # freq:  10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # shape [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        # add the positional encoding
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

# ----------------------------------------------------------------------
# 3) Training Function
# ----------------------------------------------------------------------
def train_model(train_loader, val_loader, num_epochs, learning_rate, vgg_model_path):
    start_time = time.time()
    my_logger.info('Starting Transformer + VGG-based feature extraction')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_logger.info(f'Using device: {device}')

    try:
        # 1) VGG Feature Extractor
        feature_extractor = VGGFeatureExtractor(vgg_model_path).to(device)
        feature_extractor.eval()
        for param in feature_extractor.parameters():
            param.requires_grad = False

        # 2) Transformer-based sequence model
        num_classes = 3
        input_size = feature_extractor.embedding_dim  # e.g. 4096
        transformer_model = TransformerEncoderNet(
            input_size=input_size,
            num_classes=num_classes,
            n_heads=4,           # or tweak
            num_layers=2,        # or tweak
            dim_feedforward=512, # or tweak
            dropout=0.5
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(transformer_model.parameters(), lr=learning_rate, weight_decay=1e-5)

        early_stopping_patience = 3
        epochs_without_improvement = 0
        best_recall = 0.0

        # 3) Training Loop
        for epoch in range(num_epochs):
            my_logger.info(f'Starting epoch {epoch + 1}/{num_epochs}')
            transformer_model.train()

            total_loss = 0
            correct = 0
            total = 0

            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                bsz, seq_len, channels, height, width = inputs.size()

                # Flatten slices for VGG
                inputs = inputs.view(bsz * seq_len, channels, height, width)
                with torch.no_grad():
                    features = feature_extractor(inputs)  # [bsz*seq_len, 4096]

                # reshape => [bsz, seq_len, 4096]
                features = features.view(bsz, seq_len, -1)

                logits = transformer_model(features)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * bsz
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                if i % 5 == 0:
                    batch_acc = (preds == labels).float().mean().item()
                    my_logger.info(f'Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Loss={loss.item():.4f}, Acc={batch_acc:.4f}')
                    mlflow.log_metric('running_train_loss', loss.item(), step=epoch*len(train_loader)+i)
                    mlflow.log_metric('running_train_accuracy', batch_acc, step=epoch*len(train_loader)+i)

            train_loss = total_loss / total
            train_acc = correct / total
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('train_accuracy', train_acc, step=epoch)
            my_logger.info(f'Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}')

            # Validation
            transformer_model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            all_labels, all_probs = [], []

            with torch.no_grad():
                for j, (inputs, labels) in enumerate(val_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    bsz, seq_len, channels, height, width = inputs.size()

                    inputs = inputs.view(bsz * seq_len, channels, height, width)
                    feats = feature_extractor(inputs)
                    feats = feats.view(bsz, seq_len, -1)

                    logits = transformer_model(feats)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * bsz

                    probs = torch.softmax(logits, dim=1)
                    preds = probs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            val_loss /= val_total
            val_acc = val_correct / val_total
            val_preds = np.argmax(all_probs, axis=1)

            val_recall = recall_score(all_labels, val_preds, average='macro', zero_division=0)
            val_precision = precision_score(all_labels, val_preds, average='macro', zero_division=0)
            val_f1 = f1_score(all_labels, val_preds, average='macro', zero_division=0)
            try:
                val_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
            except ValueError:
                val_auc = 0.0

            my_logger.info(
                f'Epoch {epoch+1} Val: loss={val_loss:.4f}, acc={val_acc:.4f}, '
                f'recall={val_recall:.2f}, precision={val_precision:.2f}, f1={val_f1:.2f}, auc={val_auc:.2f}'
            )
            mlflow.log_metric('val_loss', val_loss, step=epoch)
            mlflow.log_metric('val_accuracy', val_acc, step=epoch)
            mlflow.log_metric('val_recall', val_recall, step=epoch)
            mlflow.log_metric('val_precision', val_precision, step=epoch)
            mlflow.log_metric('val_f1_score', val_f1, step=epoch)
            mlflow.log_metric('val_auc', val_auc, step=epoch)

            # Early Stopping
            if val_recall > best_recall:
                best_recall = val_recall
                epochs_without_improvement = 0

                output_dir = './outputs'
                os.makedirs(output_dir, exist_ok=True)
                file_name = f'transformer_vgg_{val_total}val_{epoch+1:03}ep_{learning_rate:.5f}lr_{val_recall:.3f}rec.pth'
                torch.save(transformer_model.state_dict(), os.path.join(output_dir, file_name))

                my_logger.info(f'New best model saved. val_recall={val_recall:.3f}')
                cm = confusion_matrix(all_labels, val_preds)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap=plt.cm.Blues)
                plt.savefig(os.path.join(output_dir, f'{file_name}_confmat.png'))
                plt.close()
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                my_logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        my_logger.info("Finished Training (Transformer + VGG).")
        training_time = time.time() - start_time
        my_logger.info(f"Total training time: {training_time:.2f}s")

    except Exception as e:
        my_logger.error(f"Error during training: {e}")
        my_logger.error(traceback.format_exc())


# ----------------------------------------------------------------------
# 4) Main
# ----------------------------------------------------------------------
def main():
    my_logger.info(f"Torch version: {torch.__version__}")
    mlflow.start_run()
    my_logger.info("MLflow run started")

    parser = argparse.ArgumentParser(description='Train a small Transformer over VGG features (multiclass).')
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--k", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--i", type=int, default=0, help="Current fold index (0-based)")
    parser.add_argument("--dataset", type=str, default='ccccii', help="Dataset name")
    parser.add_argument("--run_cloud", action='store_true', help="Cloud mode")
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples")
    parser.add_argument("--vgg_model_path", type=str, default='models/vgg_multiclass_best.pth', help="Path to pretrained VGG weights")
    parser.add_argument("--sequence_length", type=int, default=30, help="Sequence length")

    args = parser.parse_args()
    my_logger.info(f"Args: {args}")
    my_logger.info(f"Current Working Dir: {os.getcwd()}")

    if not args.run_cloud:
        dataset_folder = f"data/{args.dataset}"
    else:
        # Download from blob if needed
        load_dotenv()
        storage_account = os.getenv("AZURE_STORAGE_ACCOUNT")
        storage_key = os.getenv("AZURE_STORAGE_KEY")
        container_name = os.getenv("BLOB_CONTAINER")
        
        model_uri= os.getenv('PRETRAINED_VGG_MODEL_URI')        
        my_logger.info(f"Downloading pre-trained model from blob: storage_account={storage_account}, pre-trained model={model_uri}")
        os.makedirs(os.path.dirname(args.vgg_model_path), exist_ok=True)
        download_from_blob_with_access_key(model_uri, storage_key, args.vgg_model_path)
        my_logger.info(f"Model downloaded from blob to {args.vgg_model_path}")
    
        dataset_folder = args.dataset
        my_logger.info(f"Downloading dataset from blob: {dataset_folder}")
        download_from_blob(storage_account, storage_key, container_name, dataset_folder)

    # load dataset
    my_dataset = CCCCIIDatasetSequence2D(dataset_folder, sequence_length=args.sequence_length, max_samples=args.max_samples)
    labels = my_dataset.labels
    if args.i < 0 or args.i >= args.k:
        raise ValueError(f"Fold index i={args.i} invalid for k={args.k} folds.")

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)
    splits = list(skf.split(np.zeros(len(my_dataset)), labels))
    train_idx, val_idx = splits[args.i]
    train_dataset = Subset(my_dataset, train_idx)
    val_dataset = Subset(my_dataset, val_idx)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    my_logger.info(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")

    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        vgg_model_path=args.vgg_model_path
    )

    mlflow.end_run()


if __name__ == "__main__":
    main()
