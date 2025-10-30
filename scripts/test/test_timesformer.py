#!/usr/bin/env python3
"""
Test pretrained TimeSformer Binary Sequence Classifier

This script:
  - Downloads test data from Azure Blob Storage.
  - Loads a pretrained TimeSformer model + processor.
  - Runs evaluation on the test dataset.
  - Calculates loss, accuracy, recall, precision, F1, AUC.
  - Saves a confusion matrix plot to outputs/.
"""

import argparse
import os
import time

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from dotenv import load_dotenv
from transformers import (
    TimesformerConfig,
    TimesformerForVideoClassification,
    AutoImageProcessor
)
from PIL import Image

from datasets.raster_dataset import DatasetSequence2DBinary
from utils.log_config import get_custom_logger
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from utils.download import download_from_blob, download_from_blob_with_access_key


logger = get_custom_logger('test_timesformer_binary')


def test_model(model, processor, test_loader, criterion, device, outputs_dir):
    logger.info("Starting testing phase...")
    model.eval()

    total_loss = total_correct = total_samples = 0
    all_labels, all_preds, all_probs = [], [], []
    start_time = time.time()

    with torch.no_grad():
        for seq, labels in test_loader:
            seq, labels = seq.to(device), labels.to(device)
            B, L, C, H, W = seq.shape

            # Build uint8 RGB frames
            frames = (
                seq.cpu()
                   .repeat(1, 1, 3, 1, 1)
                   .permute(0, 1, 3, 4, 2)
                   .numpy()
                   .clip(0, 255)
                   .astype(np.uint8)
            )
            videos = [
                [Image.fromarray(frames[b, i]) for i in range(L)]
                for b in range(B)
            ]
            inputs = processor(videos, return_tensors="pt").to(device)

            with autocast():
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

            total_loss += loss.item() * B
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=-1)

            total_correct += (preds == labels).sum().item()
            total_samples += B

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    # Aggregate metrics
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    labels_arr = np.array(all_labels)
    preds_arr = np.array(all_preds)
    probs_arr = np.array(all_probs)

    recall = recall_score(labels_arr, preds_arr, average='binary', pos_label=1, zero_division=0)
    precision = precision_score(labels_arr, preds_arr, average='binary', pos_label=1, zero_division=0)
    f1 = f1_score(labels_arr, preds_arr, average='binary', pos_label=1, zero_division=0)
    try:
        fpr, tpr, _ = roc_curve(labels_arr, probs_arr[:, 1], pos_label=1)
        auc_score = auc(fpr, tpr)
    except Exception:
        auc_score = 0.0

    logger.info(
        f"Test Results: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%, "
        f"Recall={recall:.2f}, Precision={precision:.2f}, "
        f"F1={f1:.2f}, AUC={auc_score:.2f}"
    )
    mlflow.log_metrics({
        "test_loss": avg_loss,
        "test_accuracy": accuracy,
        "test_recall": recall,
        "test_precision": precision,
        "test_f1": f1,
        "test_auc": auc_score
    })

    # Confusion matrix
    cm = confusion_matrix(labels_arr, preds_arr)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    os.makedirs(outputs_dir, exist_ok=True)
    cm_path = os.path.join(outputs_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {cm_path}")

    elapsed = time.time() - start_time
    logger.info(f"Testing completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Test pretrained TimeSformer binary sequence model"
    )
    parser.add_argument(
        '--model_dir',
        required=True,
        help="Path to the pretrained model + processor directory"
    )
    parser.add_argument(
        '--config_json_uri',
        required=True,
    )
    parser.add_argument(
        '--model_safetensors_uri',
        required=True,
    )
    parser.add_argument(
        '--preprocessor_config_uri',
        required=True,
    )
    parser.add_argument(
        '--test_dir',
        required=True,
        help="Local folder for test data (will be downloaded)"
    )
    parser.add_argument(
        '--sequence_length',
        type=int,
        default=30
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1
    )
    args = parser.parse_args()

    load_dotenv()
    sa, sk, cont = (
        os.getenv('AZURE_STORAGE_ACCOUNT'),
        os.getenv('AZURE_STORAGE_KEY'),
        os.getenv('BLOB_CONTAINER')
    )
    logger.info("Downloading test data from Azure Blob Storage…")
    download_from_blob(sa, sk, cont, args.test_dir)

    # -------------------------
    # Load the appropriate model and run evaluation
    # -------------------------
    logger.info(f"Downloading pretrained model from blob")
    # os.makedirs(os.path.dirname(args.model_dir), exist_ok=True)
    parent = os.path.dirname(args.model_dir)
    if parent:
        os.makedirs(parent, exist_ok=True)

    storage_account_key = os.getenv('AZURE_STORAGE_KEY')
    download_from_blob_with_access_key(args.config_json_uri, storage_account_key, args.model_dir + '/config.json')
    download_from_blob_with_access_key(args.model_safetensors_uri, storage_account_key, args.model_dir + '/model.safetensors')
    download_from_blob_with_access_key(args.preprocessor_config_uri, storage_account_key, args.model_dir + '/preprocessor_config.json')
    
    # Initialize MLflow
    mlflow.start_run()

    # Build test loader
    test_ds = DatasetSequence2DBinary(
        dataset_folder=args.test_dir,
        sequence_length=args.sequence_length
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Build model + processor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = TimesformerConfig.from_pretrained(args.model_dir)
    model = TimesformerForVideoClassification.from_pretrained(
        args.model_dir,
        config=config
    ).to(device)
    processor = AutoImageProcessor.from_pretrained(args.model_dir)

    criterion = nn.CrossEntropyLoss()
    test_model(
        model,
        processor,
        test_loader,
        criterion,
        device,
        outputs_dir="outputs"
    )

    mlflow.end_run()


if __name__ == "__main__":
    main()
