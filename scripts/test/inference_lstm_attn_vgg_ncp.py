#!/usr/bin/env python
"""
Test LSTM+Attention VGG model - INFERENCE ONLY
Para dataset com apenas uma classe (NCP)
"""

import argparse
import os
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from PIL import Image
from pathlib import Path

from utils.log_config import get_custom_logger

logger = get_custom_logger('test_lstm_attn_inference')


class VGGFeatureExtractor(nn.Module):
    """VGG16-BN backbone feature extractor"""
    def __init__(self, weights_path: str):
        super().__init__()
        backbone = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        emb_dim = backbone.classifier[6].in_features
        backbone.classifier[6] = nn.Identity()
        
        # Load pretrained weights
        state = torch.load(weights_path, map_location='cpu')
        backbone.load_state_dict(state, strict=False)
        
        self.backbone = backbone
        self.embedding_dim = emb_dim

    def forward(self, x):
        # x: [B, 1, H, W] → to 3-channel & resize
        x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(x, size=(224, 224),
                                      mode='bilinear', align_corners=False)
        return self.backbone(x)


class LSTMAttentionClassifier(nn.Module):
    """LSTM + temporal attention sequence classifier"""
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 1, num_classes: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [B, seq_len, input_dim]
        outputs, _ = self.lstm(x)
        weights = torch.softmax(self.attn(outputs).squeeze(-1), dim=1)  # [B, seq_len]
        context = (outputs * weights.unsqueeze(-1)).sum(dim=1)          # [B, hidden_dim]
        return self.fc(self.dropout(context))


class NCPSequenceDataset(Dataset):
    """Dataset para carregar apenas sequências NCP"""
    def __init__(self, root_dir, sequence_length=30):
        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.sequences = []
        
        # Lista todos os pacientes NCP
        ncp_dir = self.root_dir / "NCP"
        if not ncp_dir.exists():
            raise ValueError(f"Diretório NCP não encontrado: {ncp_dir}")
        
        for patient_dir in sorted(ncp_dir.iterdir()):
            if patient_dir.is_dir():
                for scan_dir in sorted(patient_dir.iterdir()):
                    if scan_dir.is_dir():
                        images = sorted(list(scan_dir.glob("*.png")))
                        if len(images) >= sequence_length:
                            self.sequences.append({
                                'path': scan_dir,
                                'patient': patient_dir.name,
                                'images': images[:sequence_length]
                            })
        
        logger.info(f"Loaded {len(self.sequences)} NCP sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        images = []
        
        for img_path in seq_info['images']:
            img = Image.open(img_path).convert('L')
            # Resize to consistent size
            img = img.resize((512, 512), Image.Resampling.BILINEAR)
            img_array = np.array(img, dtype=np.float32) / 255.0
            images.append(img_array)
        
        # Stack to [seq_len, H, W]
        images_array = np.stack(images, axis=0)
        # Add channel dim: [seq_len, 1, H, W]
        images_tensor = torch.from_numpy(images_array).unsqueeze(1)
        
        return images_tensor, seq_info['patient']


def test_model(feat_ext, clf, loader, device, outputs_dir):
    logger.info("Starting inference on NCP sequences with LSTM+Attention...")
    feat_ext.eval()
    clf.eval()

    all_preds = []
    all_probs = []
    all_patients = []

    with torch.no_grad():
        for imgs, patient_ids in loader:
            B, S, C, H, W = imgs.shape
            imgs = imgs.to(device)

            # Extract features in chunks
            flat = imgs.view(B * S, C, H, W)
            feats_chunks = []
            chunk_size = 64
            for i in range(0, flat.size(0), chunk_size):
                with torch.amp.autocast('cpu', enabled=False):
                    feats_chunks.append(feat_ext(flat[i:i+chunk_size]))
            feats = torch.cat(feats_chunks, dim=0).view(B, S, -1)

            # Classify sequence
            with torch.amp.autocast('cpu', enabled=False):
                logits = clf(feats)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_patients.extend(patient_ids)

    # Resultados
    preds_arr = np.array(all_preds)
    probs_arr = np.array(all_probs)
    
    logger.info("="*80)
    logger.info("INFERENCE RESULTS (LSTM+Attention):")
    logger.info("="*80)
    
    ncp_predicted = (preds_arr == 1).sum()
    normal_predicted = (preds_arr == 0).sum()
    
    logger.info(f"Total sequences: {len(preds_arr)}")
    logger.info(f"Predicted as NCP (COVID-19+): {ncp_predicted} ({ncp_predicted/len(preds_arr)*100:.1f}%)")
    logger.info(f"Predicted as Normal: {normal_predicted} ({normal_predicted/len(preds_arr)*100:.1f}%)")
    
    # Salva resultados detalhados
    os.makedirs(outputs_dir, exist_ok=True)
    results_file = os.path.join(outputs_dir, "lstm_attn_inference_results.txt")
    with open(results_file, 'w') as f:
        f.write("Patient ID\tPrediction\tProb_Normal\tProb_NCP\n")
        for patient, pred, prob in zip(all_patients, preds_arr, probs_arr):
            pred_label = "NCP" if pred == 1 else "Normal"
            f.write(f"{patient}\t{pred_label}\t{prob[0]:.4f}\t{prob[1]:.4f}\n")
    
    logger.info(f"Detailed results saved to {results_file}")
    logger.info("="*80)
    
    return {
        "total": len(preds_arr),
        "ncp_predicted": ncp_predicted,
        "normal_predicted": normal_predicted
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test LSTM+Attention VGG model inference on NCP data"
    )
    parser.add_argument("--vgg_model_path", type=str, required=True,
                        help="Path to VGG backbone .pth file")
    parser.add_argument("--lstm_model_path", type=str, required=True,
                        help="Path to LSTM+Attention classifier .pth file")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Path to test data directory")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for testing")
    parser.add_argument("--sequence_length", type=int, default=30,
                        help="Number of slices per sequence")
    parser.add_argument("--outputs_dir", type=str, default="outputs",
                        help="Directory to save outputs")
    args = parser.parse_args()

    # Setup dataset
    logger.info(f"Loading NCP sequences from {args.test_dir}")
    test_ds = NCPSequenceDataset(
        root_dir=args.test_dir,
        sequence_length=args.sequence_length
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load VGG feature extractor
    logger.info(f"Loading VGG feature extractor from {args.vgg_model_path}")
    feat_ext = VGGFeatureExtractor(args.vgg_model_path)
    feat_ext.to(device)

    # Load LSTM+Attention classifier
    logger.info(f"Loading LSTM+Attention classifier from {args.lstm_model_path}")
    clf = LSTMAttentionClassifier(
        input_dim=feat_ext.embedding_dim,
        hidden_dim=128,
        num_layers=1,
        num_classes=2,
        dropout=0.5
    )
    clf.load_state_dict(torch.load(args.lstm_model_path, map_location=device))
    clf.to(device)

    # Inference
    results = test_model(feat_ext, clf, test_loader, device, 
                        outputs_dir=args.outputs_dir)


if __name__ == "__main__":
    main()
