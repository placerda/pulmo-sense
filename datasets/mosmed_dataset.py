import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.log_config import get_custom_logger

my_logger = get_custom_logger('mosmed_dataset')

class MosmedDataset2DBinary(Dataset):
    """
    A binary dataset for Mosmed that mimics the structure of CCCCIIDataset2DBinary.
    Assumes folder structure:
      mosmed/NCP/study_xxxx/study_xxxx/0000.png, 0001.png, ...
      mosmed/Normal/study_xxxx/study_xxxx/0000.png, ...
    """
    def __init__(self, root_dir, transform=None, max_samples=0):
        self.root_dir = root_dir
        self.transform = transform
        # Binary classes: NCP and Normal
        self.classes = {'NCP': 0, 'Normal': 1}
        self.max_samples = max_samples
        my_logger.info(f"MosmedDataset2DBinary: Initialized with root_dir: {root_dir}")
        self.data = self._gather_data()

    def _gather_data(self):
        data = []
        sample_counts = {0: 0, 1: 0}
        for label, class_idx in self.classes.items():
            class_dir = os.path.join(self.root_dir, label)
            if not os.path.exists(class_dir):
                my_logger.error(f"Directory {class_dir} does not exist.")
                continue
            patients = os.listdir(class_dir)
            for patient in patients:
                patient_dir = os.path.join(class_dir, patient)
                # Expect a subfolder with the same patient/study name
                scan_dir = os.path.join(patient_dir, patient)
                if not os.path.isdir(scan_dir):
                    my_logger.warning(f"Scan directory {scan_dir} not found, skipping.")
                    continue
                slices = sorted(os.listdir(scan_dir))
                if len(slices) < 30:
                    my_logger.warning(f"Not enough slices in {scan_dir} (found {len(slices)}), skipping.")
                    continue
                num_slices = len(slices)
                start_idx = max(0, (num_slices - 30) // 2)
                end_idx = start_idx + 30
                for slice_idx in range(start_idx, end_idx):
                    data.append((scan_dir, slice_idx, class_idx))
                    sample_counts[class_idx] += 1
                    if self.max_samples > 0 and len(data) >= self.max_samples:
                        return data
        my_logger.info(f"Collected {len(data)} samples. Sample counts: {sample_counts}")
        return data

    def __len__(self):
        return len(self.data)

    def _load_scan(self, scan_dir, slice_idx):
        slice_files = sorted(os.listdir(scan_dir))
        slice_file = slice_files[slice_idx]
        slice_path = os.path.join(scan_dir, slice_file)
        img = Image.open(slice_path).convert('L')
        img = img.resize((512, 512))
        img = np.array(img).astype(np.float32)
        mean = np.mean(img)
        std = np.std(img)
        if std > 0:
            img = (img - mean) / std
        else:
            my_logger.info(f"Zero std in image {slice_path}")
            img = img - mean
        return img

    def _extract_patient_id(self, scan_dir):
        # Extract numeric parts from the directory name as patient ID
        return int(''.join(filter(str.isdigit, os.path.basename(scan_dir))))

    def __getitem__(self, idx):
        scan_dir, slice_idx, label = self.data[idx]
        img = self._load_scan(scan_dir, slice_idx)
        patient_id = self._extract_patient_id(scan_dir)
        if self.transform:
            img = self.transform(img)
        # Ensure shape is (1, 512, 512)
        img = torch.tensor(img).unsqueeze(0)
        return img, torch.tensor(patient_id).long(), torch.tensor(label).long()


class MosmedSequenceDataset2DBinary(Dataset):
    """
    Sequence dataset version for models that process sequences (e.g. LSTM-based methods).
    This groups slices by patient and forms sequences of 30 consecutive slices.
    """
    def __init__(self, root_dir, sequence_length=30, max_samples=None):
        self.sequence_length = sequence_length
        self.max_samples = max_samples
        # Initialize the base dataset
        base_dataset = MosmedDataset2DBinary(root_dir, transform=None, max_samples=max_samples)
        data = []
        for i in range(len(base_dataset)):
            slice_img, patient_id, label = base_dataset[i]
            data.append((slice_img, patient_id, label))
        # Group slices by patient_id
        patient_slices = {}
        patient_labels = {}
        for slice_img, patient_id, label in data:
            pid = patient_id.item()
            if pid not in patient_slices:
                patient_slices[pid] = []
                patient_labels[pid] = []
            patient_slices[pid].append(slice_img)
            patient_labels[pid].append(label)
        self.sequences = []
        self.labels = []
        for pid in patient_slices:
            slices = patient_slices[pid]
            labels = patient_labels[pid]
            num_full_sequences = len(slices) // self.sequence_length
            for i in range(num_full_sequences):
                seq = slices[i * self.sequence_length:(i + 1) * self.sequence_length]
                # Use majority vote for the sequence label
                seq_labels = [l.item() for l in labels[i * self.sequence_length:(i + 1) * self.sequence_length]]
                majority_label = max(set(seq_labels), key=seq_labels.count)
                self.sequences.append(seq)
                self.labels.append(majority_label)
        if self.max_samples and self.max_samples > 0:
            self.sequences = self.sequences[:self.max_samples]
            self.labels = self.labels[:self.max_samples]
        my_logger.info(f"MosmedSequenceDataset2DBinary: Total sequences: {len(self.sequences)}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        seq = torch.stack(seq)  # Shape: (sequence_length, 1, 512, 512)
        return seq, torch.tensor(label).long()
