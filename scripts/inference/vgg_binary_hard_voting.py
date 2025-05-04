#!/usr/bin/env python
"""
Batch inference for pretrained VGG Binary Sequence Classifier.

For each scan folder under your dataset root (e.g. Normal/2318/773),
this script:
  - loads the central 30 slices,
  - runs slice-wise majority voting,
  - logs a line "relative_path -> predicted_label".
"""
import argparse
import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models
from utils.download import download_from_blob, download_from_blob_with_access_key
from utils.log_config import get_custom_logger

my_logger = get_custom_logger('batch_infer_vgg_binary')

class VGG_Net(nn.Module):
    """pretrained VGG16-BN wrapper for 2-class output."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        if x.shape[1] == 1:             # single-channel -> 3-channel
            x = x.repeat(1, 3, 1, 1)
        return self.model(x)

def preprocess_slice(slice_path):
    img = Image.open(slice_path).convert('L').resize((512, 512))
    arr = np.array(img, dtype=np.float32)
    m, s = arr.mean(), arr.std()
    if s > 0:
        arr = (arr - m) / s
    else:
        arr = arr - m
    tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
    return tensor

def infer_scan(scan_path, model, device):
    all_files = sorted(os.listdir(scan_path))
    if len(all_files) < 30:
        my_logger.warning(f"Skipping {scan_path}: only {len(all_files)} slices")
        return None

    # pick central 30
    start = (len(all_files) - 30) // 2
    files30 = all_files[start : start + 30]

    # build tensor [30,1,512,512]
    slices = [preprocess_slice(os.path.join(scan_path, f)) for f in files30]
    seq = torch.stack(slices).to(device)           # (30,1,512,512)

    # chunked inference
    chunk_size = 10
    logits_chunks = []
    with torch.no_grad():
        for i in range(0, 30, chunk_size):
            chunk = seq[i : i + chunk_size]        # (<=10,1,512,512)
            out = model(chunk)                     # (<=10,2)
            logits_chunks.append(out.view(1, -1, 2))
        logits = torch.cat(logits_chunks, dim=1)   # (1,30,2)

        slice_preds = logits.argmax(dim=2).view(-1)  # (30,)
        votes = int(slice_preds.sum().item())
        pred = 1 if votes > (30 // 2) else 0

    return pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_uri",   required=True,
                        help="Blob URI to pretrained .pth file")
    parser.add_argument("--dataset_dir", required=True,
                        help="Local folder for downloaded test data")
    args = parser.parse_args()

    load_dotenv()
    sa   = os.getenv("AZURE_STORAGE_ACCOUNT")
    sk   = os.getenv("AZURE_STORAGE_KEY")
    cont = os.getenv("BLOB_CONTAINER")

    my_logger.info("Downloading test dataset...")
    download_from_blob(sa, sk, cont, args.dataset_dir)

    # download model
    local_model = "models/vgg_binary.pth"
    my_logger.info(f"Downloading model from {args.model_uri} to {local_model}")
    download_from_blob_with_access_key(args.model_uri, sk, local_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG_Net(num_classes=2).to(device)
    model.load_state_dict(torch.load(local_model, map_location=device))
    model.eval()
    my_logger.info(f"Model loaded on {device}")

    root = args.dataset_dir
    # walk each scan under Normal/ and NCP/
    for class_name in ["Normal", "NCP"]:
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir):
            continue

        for patient in sorted(os.listdir(class_dir)):
            patient_dir = os.path.join(class_dir, patient)
            if not os.path.isdir(patient_dir):
                continue

            for scan in sorted(os.listdir(patient_dir)):
                scan_path = os.path.join(patient_dir, scan)
                if not os.path.isdir(scan_path):
                    continue

                rel = os.path.relpath(scan_path, root).replace("\\", "/")
                pred = infer_scan(scan_path, model, device)
                if pred is not None:
                    my_logger.info(f"{rel} -> {pred}")

if __name__ == "__main__":
    main()
