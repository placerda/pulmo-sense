#!/usr/bin/env python
import argparse
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from utils.log_config import get_custom_logger
import mlflow  # Added for mlflow logging

# Import Mosmed dataset classes
from datasets import MosmedDataset2DBinary, MosmedSequenceDataset2DBinary

my_logger = get_custom_logger('test_mosmed')

# -------------------------
# Evaluation Function
# -------------------------
def evaluate_model(model, loader, device, is_sequence=False, is_lstm_attn=False, feature_extractor=None):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            if is_sequence:
                # For sequence models, batch is (batch, seq, 1, 512, 512) with labels
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                if is_lstm_attn:
                    # Process each slice through the VGG feature extractor
                    b, seq, c, h, w = inputs.size()
                    inputs_flat = inputs.view(b * seq, c, h, w)
                    features = feature_extractor(inputs_flat)
                    features = features.view(b, seq, -1)
                    outputs, _ = model(features)
                else:
                    outputs, _ = model(inputs)
            else:
                # For 2D models: inputs shape [batch, 1, 512, 512]
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
        auc_value = auc(fpr, tpr)
    except Exception as e:
        my_logger.error(f"Error computing AUC: {e}")
        auc_value = 0.0
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    return acc, auc_value, f1, precision, recall, all_labels, all_preds

# -------------------------
# Main Test Script
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Test Best Models on Mosmed Dataset")
    parser.add_argument("--mosmed_dataset", type=str, default="mosmed_png", help="Path or container name for the Mosmed dataset")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_type", type=str, choices=["vgg", "vit", "lstm_attn"], required=True,
                        help="Model type to test")
    parser.add_argument("--model_path", type=str, required=True, help="Path to best model weights")
    parser.add_argument("--vgg_model_path", type=str, default="",  help="Path to pretrained VGG weights (required for lstm_attn)")
    parser.add_argument("--model_uri", type=str, required=True,
                        help="Blob URI to download pretrained model weights")
    parser.add_argument("--vgg_model_uri", type=str, default="",
                        help="Blob URI to download pretrained VGG model weights (required for lstm_attn)")
    args = parser.parse_args()

    # Determine dataset folder based on run mode
    dataset_folder = args.mosmed_dataset
    load_dotenv()
    storage_account = os.getenv("AZURE_STORAGE_ACCOUNT")
    storage_key = os.getenv("AZURE_STORAGE_KEY")
    container_name = os.getenv("BLOB_CONTAINER")
    my_logger.info("Cloud mode detected. Downloading dataset from blob storage.")
    try:
        from utils.download import download_from_blob
        download_from_blob(storage_account, storage_key, container_name, dataset_folder)
    except Exception as e:
        my_logger.error("Failed to download dataset from blob: %s", e)
        raise e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create DataLoader based on model type
    if args.model_type in ["vgg", "vit"]:
        dataset = MosmedDataset2DBinary(root_dir=dataset_folder)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    elif args.model_type == "lstm_attn":
        dataset = MosmedSequenceDataset2DBinary(root_dir=dataset_folder, sequence_length=30)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    else:
        raise ValueError("Unsupported model type.")

    # -------------------------
    # Model Definitions
    # -------------------------
    import torchvision.models as models
    import torch.nn as nn
    import timm

    class VGG_Net(nn.Module):
        def __init__(self, num_classes=2):
            super(VGG_Net, self).__init__()
            self.model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
            in_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(in_features, num_classes)
        def forward(self, x):
            x = x.repeat(1, 3, 1, 1)
            return self.model(x)

    class ViTModel(nn.Module):
        def __init__(self, num_classes=2, model_name='vit_base_patch16_224'):
            super(ViTModel, self).__init__()
            self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        def forward(self, x):
            x = x.repeat(1, 3, 1, 1)
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            return self.model(x)

    class VGGFeatureExtractor(nn.Module):
        def __init__(self, vgg_weights_path):
            super(VGGFeatureExtractor, self).__init__()
            self.vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
            in_features = self.vgg.classifier[6].in_features
            self.vgg.classifier[6] = nn.Identity()
            state_dict = torch.load(vgg_weights_path, map_location='cpu')
            self.vgg.load_state_dict(state_dict, strict=False)
            self.embedding_dim = in_features
        def forward(self, x):
            x = x.repeat(1, 3, 1, 1)
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            features = self.vgg(x)
            return features

    class TemporalAttention(nn.Module):
        def __init__(self, hidden_dim):
            super(TemporalAttention, self).__init__()
            self.attn_score = nn.Linear(hidden_dim, 1, bias=False)
        def forward(self, lstm_outputs):
            energies = self.attn_score(lstm_outputs).squeeze(-1)
            attn_weights = torch.softmax(energies, dim=1)
            context = torch.sum(lstm_outputs * attn_weights.unsqueeze(-1), dim=1)
            return context, attn_weights

    class LSTM_AttnNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
            super(LSTM_AttnNet, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.dropout = nn.Dropout(dropout_rate)
            self.attention = TemporalAttention(hidden_size)
            self.fc = nn.Linear(hidden_size, num_classes)
        def forward(self, x):
            lstm_outputs, _ = self.lstm(x)
            context, attn_weights = self.attention(lstm_outputs)
            context = self.dropout(context)
            logits = self.fc(context)
            return logits, attn_weights

    # LSTM model without attention (for lstm_vgg) based on your training script
    class LSTM_Net(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
            super(LSTM_Net, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.dropout = nn.Dropout(dropout_rate)
            self.fc = nn.Linear(hidden_size, num_classes)
        def forward(self, x):
            out, _ = self.lstm(x)       # out: [batch, seq_len, hidden_size]
            out = out[:, -1, :]         # take the final time step
            out = self.dropout(out)
            out = self.fc(out)
            return out

    # -------------------------
    # Load the appropriate model and run evaluation
    # -------------------------
    model_uri = args.model_uri
    my_logger.info(f"Downloading pretrained model from blob: {model_uri}")
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    from utils.download import download_from_blob, download_from_blob_with_access_key
    storage_account_key = os.getenv('AZURE_STORAGE_KEY')
    download_from_blob_with_access_key(model_uri, storage_account_key, args.model_path)
    my_logger.info(f"Pretrained model downloaded to {args.model_path}")

    if args.model_type == "vgg":
        model = VGG_Net(num_classes=2).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        my_logger.info("Evaluating VGG Model:")
        acc, auc_val, f1, precision, recall, all_labels, all_preds = evaluate_model(model, loader, device)
    elif args.model_type == "vit":
        model = ViTModel(num_classes=2).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        my_logger.info("Evaluating ViT Model:")
        acc, auc_val, f1, precision, recall, all_labels, all_preds = evaluate_model(model, loader, device)
    elif args.model_type == "lstm_attn":
        if not args.vgg_model_uri:
            raise ValueError("For lstm_attn, --vgg_model_uri is required")    
        vgg_model_uri = args.vgg_model_uri
        my_logger.info(f"Downloading pretrained VGG model from blob: {vgg_model_uri}")
        os.makedirs(os.path.dirname(args.vgg_model_path), exist_ok=True)
        from utils.download import download_from_blob, download_from_blob_with_access_key
        storage_account_key = os.getenv("AZURE_STORAGE_KEY")
        download_from_blob_with_access_key(vgg_model_uri, storage_account_key, args.vgg_model_path)
        my_logger.info(f"Pretrained VGG model downloaded to {args.vgg_model_path}")

        feature_extractor = VGGFeatureExtractor(vgg_weights_path=args.vgg_model_path).to(device)
        feature_extractor.eval()
        for param in feature_extractor.parameters():
            param.requires_grad = False
        model = LSTM_AttnNet(input_size=feature_extractor.embedding_dim, hidden_size=128,
                             num_layers=1, num_classes=2, dropout_rate=0.5).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        my_logger.info("Evaluating LSTM + Attention Model:")
        acc, auc_val, f1, precision, recall, all_labels, all_preds = evaluate_model(
            model, loader, device, is_sequence=True, is_lstm_attn=True,
            feature_extractor=feature_extractor
        )
    elif args.model_type == "lstm_vgg":
        # Implementing the LSTM with VGG features (without attention)
        if not args.vgg_model_uri:
            raise ValueError("For lstm_vgg, --vgg_model_uri is required")
        vgg_model_uri = args.vgg_model_uri
        my_logger.info(f"Downloading pretrained VGG model from blob: {vgg_model_uri}")
        os.makedirs(os.path.dirname(args.vgg_model_path), exist_ok=True)
        download_from_blob_with_access_key(vgg_model_uri, storage_account_key, args.vgg_model_path)
        my_logger.info(f"Pretrained VGG model downloaded to {args.vgg_model_path}")

        feature_extractor = VGGFeatureExtractor(vgg_weights_path=args.vgg_model_path).to(device)
        feature_extractor.eval()
        for param in feature_extractor.parameters():
            param.requires_grad = False
        model = LSTM_Net(input_size=feature_extractor.embedding_dim, hidden_size=128,
                         num_layers=1, num_classes=2, dropout_rate=0.5).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        my_logger.info("Evaluating LSTM + VGG Model:")
        acc, auc_val, f1, precision, recall, all_labels, all_preds = evaluate_model(
            model, loader, device, is_sequence=True, is_lstm_attn=False,
            feature_extractor=feature_extractor
        )  
    else:
        raise ValueError("Unsupported model type.")

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    my_logger.info(
        f"Test Metrics:\n"
        f"Accuracy: {acc * 100:.2f}%\n"
        f"AUC: {auc_val:.2f}\n"
        f"F1 Score: {f1:.2f}\n"
        f"Precision: {precision:.2f}\n"
        f"Recall: {recall:.2f}\n"
        f"Confusion Matrix:\n{cm}"
    )

    # Log final test metrics in mlflow
    mlflow.log_metric("test_accuracy", acc * 100, step=0)
    mlflow.log_metric("test_auc", auc_val, step=0)
    mlflow.log_metric("test_f1", f1, step=0)
    mlflow.log_metric("test_precision", precision, step=0)
    mlflow.log_metric("test_recall", recall, step=0)

    # Plot confusion matrix using ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['COVID', 'Non-COVID'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/{args.model_type}_confmat.png")
    my_logger.info(f"Confusion matrix saved as outputs/{args.model_type}_confmat.png")

if __name__ == "__main__":
    main()
