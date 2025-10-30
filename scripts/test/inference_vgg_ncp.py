#!/usr/bin/env python
"""
Test VGG binary model with soft voting - INFERENCE ONLY
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

logger = get_custom_logger('test_vgg_inference')


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


def test_model(model, test_loader, device, outputs_dir):
    logger.info("Starting inference on NCP sequences...")
    model.eval()

    all_preds = []
    all_probs = []
    all_patients = []

    with torch.no_grad():
        for seq_slices, patient_ids in test_loader:
            B, L, C, H, W = seq_slices.shape
            seq_slices = seq_slices.to(device)

            # Chunked inference
            chunk_size = 10
            slice_logits_chunks = []
            for start in range(0, L, chunk_size):
                end = min(start + chunk_size, L)
                flat = seq_slices[:, start:end].reshape(-1, C, H, W)
                # Replicate to 3-channel and resize
                flat = flat.repeat(1, 3, 1, 1)
                flat = nn.functional.interpolate(
                    flat, size=(224, 224), mode='bilinear', align_corners=False
                )
                with autocast():
                    out = model(flat)
                slice_logits_chunks.append(out.view(B, -1, 2))

            slice_logits = torch.cat(slice_logits_chunks, dim=1)  # (B, L, 2)

            # Soft Voting
            slice_preds = slice_logits.argmax(dim=2)            # (B, L)
            votes = slice_preds.sum(dim=1)                      # (B,)
            preds = (votes > (L // 2)).long()                   # (B,)
            prob_pos = votes.float() / L                        # (B,)
            probs = torch.stack([1 - prob_pos, prob_pos], dim=1) # (B, 2)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_patients.extend(patient_ids)

    # Resultados
    preds_arr = np.array(all_preds)
    probs_arr = np.array(all_probs)
    
    logger.info("="*80)
    logger.info("INFERENCE RESULTS:")
    logger.info("="*80)
    
    ncp_predicted = (preds_arr == 1).sum()
    normal_predicted = (preds_arr == 0).sum()
    
    logger.info(f"Total sequences: {len(preds_arr)}")
    logger.info(f"Predicted as NCP (COVID-19+): {ncp_predicted} ({ncp_predicted/len(preds_arr)*100:.1f}%)")
    logger.info(f"Predicted as Normal: {normal_predicted} ({normal_predicted/len(preds_arr)*100:.1f}%)")
    
    # Salva resultados detalhados
    os.makedirs(outputs_dir, exist_ok=True)
    results_file = os.path.join(outputs_dir, "vgg_inference_results.txt")
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
        description="Test VGG model inference on NCP data"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pretrained .pth model file")
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

    # Load VGG model
    logger.info("Creating VGG16-BN model...")
    model = models.vgg16_bn(weights=None)
    # Modify classifier for binary classification
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 2)
    
    logger.info(f"Loading pretrained weights from {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # Inference
    results = test_model(model, test_loader, device, outputs_dir=args.outputs_dir)


if __name__ == "__main__":
    main()
