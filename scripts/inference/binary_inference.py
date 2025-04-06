#!/usr/bin/env python
"""
Inference script for CT scan classification.
This program takes as input:
  - a folder containing PNG slices of a CT scan (expects at least 30 slices)
  - the path to the trained model weights,
  - the model type (e.g. "vit", "vgg", "clip", "mae", "lstm_attn", "lstm_vgg")
  - (optionally) the path to the pretrained VGG weights (for lstm_attn and lstm_vgg)

It loads the central 30 slices from the scan folder, applies normalization,
runs inference using the specified model architecture, and prints the predicted class (0 or 1).
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.models as models
import timm

# -------------------------------
# Helper: Image loading & preprocessing
# -------------------------------
def load_and_preprocess_image(image_path):
    """
    Loads an image as grayscale, resizes to 512x512, and normalizes it.
    """
    img = Image.open(image_path).convert('L')  # load in grayscale
    img = img.resize((512, 512))
    img = np.array(img).astype(np.float32)
    mean = np.mean(img)
    std = np.std(img)
    if std > 0:
        img = (img - mean) / std
    else:
        img = img - mean
    # Convert to tensor and add channel dimension -> shape (1, 512, 512)
    tensor_img = torch.tensor(img).unsqueeze(0)
    return tensor_img

# -------------------------------
# Model Definitions (adapted from training scripts)
# -------------------------------

# 1. CLIP-based Binary Model
class CLIPBinaryModel(nn.Module):
    def __init__(self, num_classes=2, model_name="openai/clip-vit-base-patch32"):
        super(CLIPBinaryModel, self).__init__()
        from transformers import CLIPModel  # requires transformers installed
        self.clip = CLIPModel.from_pretrained(model_name)
        hidden_size = self.clip.config.vision_hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: [batch, 1, 512, 512] -> replicate channels, resize to 224x224
        x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        vision_outputs = self.clip.vision_model(pixel_values=x)
        pooled_output = vision_outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

# 2. MAE-based Binary Model
class MAEModel(nn.Module):
    def __init__(self, num_classes=2, model_name='vit_base_patch16_224.mae'):
        super(MAEModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        # x: [batch, 1, 512, 512]
        x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
        return self.model(x)

# 3. Vision Transformer (ViT) Model
class ViTModel(nn.Module):
    def __init__(self, num_classes=2, model_name='vit_base_patch16_224'):
        super(ViTModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
        return self.model(x)

# 4. VGG-based 2D Model
class VGGNet(nn.Module):
    def __init__(self, num_classes=2):
        super(VGGNet, self).__init__()
        self.model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        return self.model(x)

# 5. VGG Feature Extractor for LSTM-based models
class VGGFeatureExtractor(nn.Module):
    def __init__(self, vgg_weights_path):
        super(VGGFeatureExtractor, self).__init__()
        self.vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        in_features = self.vgg.classifier[6].in_features
        # Replace the final classification layer with identity
        self.vgg.classifier[6] = nn.Identity()
        state_dict = torch.load(vgg_weights_path, map_location='cpu')
        self.vgg.load_state_dict(state_dict, strict=False)
        self.embedding_dim = in_features
    
    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
        features = self.vgg(x)
        return features

# 6. LSTM with Attention Model
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attn_score = nn.Linear(hidden_dim, 1, bias=False)
    def forward(self, lstm_outputs):
        energies = self.attn_score(lstm_outputs).squeeze(-1)  # [batch, seq]
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

# 7. LSTM without Attention Model
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # use the final time step
        out = self.dropout(out)
        out = self.fc(out)
        return out

# -------------------------------
# Main Inference Function
# -------------------------------
def main():

    # Scan folder paths
    # NCP
    # default_scan_folder = "data/ccccii/NCP/16/1164"
    # default_scan_folder = "data/mosmed_png_normal/NCP/study_0255/study_0255"    
    # Normal
    # default_scan_folder = "data/ccccii/Normal/750/185"
    default_scan_folder = "data/mosmed_png_normal/Normal/study_0013/study_0013"    


    # Model paths and types
    default_model_type = "vit"
    default_model_path = "models/vit_binary_4epoch_0.00050lr_0.954rec.pth"
    default_vgg_model_path = "models/vgg_pretrained.pth"

    # default_model_type = "vgg"

    # default_model_type = "lstm_attn"
    # default_vgg_model_path = "models/vgg_pretrained.pth"
    
    # default_model_type = "lstm_vgg"
    # default_vgg_model_path = "models/vgg_pretrained.pth"

    # default_model_type = "clip" 

    # default_model_type = "mae"    

    parser = argparse.ArgumentParser(description="Inference on CT scan folder")
    parser.add_argument("--scan_folder", type=str, default=default_scan_folder, help="Path to folder containing CT scan PNG images")
    parser.add_argument("--model_path", type=str, default=default_model_path, help="Path to trained model weights")
    parser.add_argument("--model_type", type=str, default=default_model_type,
                        choices=["vit", "vgg", "clip", "mae", "lstm_attn", "lstm_vgg"],
                        help="Model type used during training")
    parser.add_argument("--vgg_model_path", type=str, default=default_vgg_model_path,
                        help="Path to pretrained VGG weights (required for lstm_attn and lstm_vgg)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validate and list PNG files in the scan folder
    if not os.path.isdir(args.scan_folder):
        print(f"Scan folder {args.scan_folder} does not exist or is not a directory.")
        return

    image_files = sorted([f for f in os.listdir(args.scan_folder) if f.lower().endswith('.png')])
    if len(image_files) < 30:
        print("Not enough slices in scan folder. Need at least 30 PNG images.")
        return

    # Select the central 30 slices
    num_slices = len(image_files)
    start_idx = (num_slices - 30) // 2
    selected_files = image_files[start_idx:start_idx+30]

    # Load and preprocess each slice
    slices = []
    for filename in selected_files:
        path = os.path.join(args.scan_folder, filename)
        tensor_img = load_and_preprocess_image(path)  # shape (1, 512, 512)
        slices.append(tensor_img)

    # For sequence models (LSTM-based) stack slices into a single tensor with shape (1, 30, 1, 512, 512)
    # For 2D models, we will process each slice independently and then average the outputs.
    if args.model_type in ["lstm_attn", "lstm_vgg"]:
        input_tensor = torch.stack(slices)         # (30, 1, 512, 512)
        input_tensor = input_tensor.unsqueeze(0)     # (1, 30, 1, 512, 512)
    else:
        input_tensor = torch.stack(slices)           # (30, 1, 512, 512)

    input_tensor = input_tensor.to(device)

    # Load the appropriate model and run inference
    if args.model_type == "clip":
        model = CLIPBinaryModel(num_classes=2).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        # Process each slice separately and average probabilities
        probs_list = []
        with torch.no_grad():
            for i in range(input_tensor.shape[0]):
                slice_tensor = input_tensor[i].unsqueeze(0)  # (1, 1, 512, 512)
                logits = model(slice_tensor)
                probs = torch.softmax(logits, dim=1)
                probs_list.append(probs)
        avg_probs = torch.mean(torch.cat(probs_list, dim=0), dim=0)
        pred_class = torch.argmax(avg_probs).item()

    elif args.model_type == "mae":
        model = MAEModel(num_classes=2).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        probs_list = []
        with torch.no_grad():
            for i in range(input_tensor.shape[0]):
                slice_tensor = input_tensor[i].unsqueeze(0)
                logits = model(slice_tensor)
                probs = torch.softmax(logits, dim=1)
                probs_list.append(probs)
        avg_probs = torch.mean(torch.cat(probs_list, dim=0), dim=0)
        pred_class = torch.argmax(avg_probs).item()

    elif args.model_type == "vit":
        model = ViTModel(num_classes=2).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        probs_list = []
        with torch.no_grad():
            for i in range(input_tensor.shape[0]):
                slice_tensor = input_tensor[i].unsqueeze(0)
                logits = model(slice_tensor)
                probs = torch.softmax(logits, dim=1)
                probs_list.append(probs)
        avg_probs = torch.mean(torch.cat(probs_list, dim=0), dim=0)
        pred_class = torch.argmax(avg_probs).item()

    elif args.model_type == "vgg":
        model = VGGNet(num_classes=2).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        probs_list = []
        with torch.no_grad():
            for i in range(input_tensor.shape[0]):
                slice_tensor = input_tensor[i].unsqueeze(0)
                logits = model(slice_tensor)
                probs = torch.softmax(logits, dim=1)
                probs_list.append(probs)
        avg_probs = torch.mean(torch.cat(probs_list, dim=0), dim=0)
        pred_class = torch.argmax(avg_probs).item()

    elif args.model_type == "lstm_attn":
        if args.vgg_model_path == "":
            print("For lstm_attn, --vgg_model_path is required.")
            return
        # Load frozen VGG feature extractor
        feature_extractor = VGGFeatureExtractor(vgg_weights_path=args.vgg_model_path).to(device)
        feature_extractor.eval()
        model = LSTM_AttnNet(input_size=feature_extractor.embedding_dim,
                             hidden_size=128, num_layers=1, num_classes=2, dropout_rate=0.5).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        # For sequence models: reshape input to (batch*seq, channels, H, W)
        b, seq, c, h, w = input_tensor.shape
        input_seq = input_tensor.view(b * seq, c, h, w)
        with torch.no_grad():
            features = feature_extractor(input_seq)
            features = features.view(b, seq, -1)
            logits, attn_weights = model(features)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

    elif args.model_type == "lstm_vgg":
        if args.vgg_model_path == "":
            print("For lstm_vgg, --vgg_model_path is required.")
            return
        feature_extractor = VGGFeatureExtractor(vgg_weights_path=args.vgg_model_path).to(device)
        feature_extractor.eval()
        model = LSTMNet(input_size=feature_extractor.embedding_dim,
                        hidden_size=128, num_layers=1, num_classes=2, dropout_rate=0.5).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        b, seq, c, h, w = input_tensor.shape
        input_seq = input_tensor.view(b * seq, c, h, w)
        with torch.no_grad():
            features = feature_extractor(input_seq)
            features = features.view(b, seq, -1)
            logits = model(features)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

    else:
        print("Unsupported model type.")
        return

    print(f"Predicted class: {pred_class}")

if __name__ == "__main__":
    main()
