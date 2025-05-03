import os
import random
from collections import Counter
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.log_config import get_custom_logger

my_logger = get_custom_logger('raster_dataset')

class Dataset2DBinary(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=0):
        """
        A binary dataset class that uses only NCP and Normal classes.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = {'NCP': 1, 'Normal': 0}
        self.num_classes = len(self.classes)
        self.max_samples = max_samples
        
        my_logger.info(f"Dataset2DBinary: Dataset initialized with root_dir: {self.root_dir}")
        my_logger.info(f"Dataset2DBinary: Transform: {self.transform}")
        my_logger.info(f"Dataset2DBinary: Max samples: {self.max_samples}")
        my_logger.info(f"Dataset2DBinary: Number of classes: {self.num_classes}")
        my_logger.info(f"Dataset2DBinary: Labels : {self.classes}")
        
        # Gather data properly and then extract patient IDs and labels
        self.data = self._gather_data()
        self.patient_ids = [self._extract_patient_id(scan_path) for scan_path, _, _ in self.data]
        self.labels = [label for _, _, label in self.data]

    def _gather_data(self):
        data = []
        patient_list = []
        sample_counts = {class_idx: 0 for class_idx in self.classes.values()}

        my_logger.info(f"Dataset2DBinary: Current working directory: {os.getcwd()}")
        my_logger.info(f"Dataset2DBinary: Contents of root directory ({self.root_dir}): {os.listdir(self.root_dir)}")

        for label, class_idx in self.classes.items():
            class_dir = os.path.join(self.root_dir, label)
            patients = os.listdir(class_dir)
            for patient_id in patients:
                patient_dir = os.path.join(class_dir, patient_id)
                for scan_folder in os.listdir(patient_dir):
                    scan_path = os.path.join(patient_dir, scan_folder)
                    slices = os.listdir(scan_path)
                    if len(slices) >= 30:
                        patient_list.append((scan_path, class_idx))

        random.seed(42)
        random.shuffle(patient_list)

        count = 0
        for scan_path, class_idx in patient_list:
            slices = sorted(os.listdir(scan_path))
            num_slices = len(slices)
            start_idx = max(0, (num_slices - 30) // 2)
            end_idx = min(start_idx + 30, num_slices)

            for slice_idx in range(start_idx, end_idx):
                data.append((scan_path, slice_idx, class_idx))
                sample_counts[class_idx] += 1
                count += 1
                if self.max_samples > 0 and count >= self.max_samples:
                    my_logger.info(f"Dataset2DBinary: Total samples : {len(data)}")
                    my_logger.info(f"Dataset2DBinary: Samples per class : {sample_counts}")
                    return data

        my_logger.info(f"Dataset2DBinary: Total samples : {len(data)}")
        my_logger.info(f"Dataset2DBinary: Samples per class : {sample_counts}")
        return data

    def __len__(self):
        return len(self.data)

    def _load_scan(self, scan_path, slice_idx):
        slice_files = sorted(os.listdir(scan_path))
        slice_file = slice_files[slice_idx]
        slice_path = os.path.join(scan_path, slice_file)        
        img = Image.open(slice_path).convert('L')
        img = img.resize((512, 512))
        img = np.array(img).astype(np.float32)
        mean = np.mean(img)
        std = np.std(img)
        if std > 0:
            img = (img - mean) / std
        else:
            my_logger.info(f"Dataset2DBinary: Image with zero std encountered: {scan_path} - slice {slice_idx}")
            img = img - mean
        return img

    def _extract_patient_id(self, scan_path):
        patient_id = os.path.basename(os.path.dirname(scan_path))
        return int(patient_id)

    def __getitem__(self, idx):
        scan_path, slice_idx, label = self.data[idx]
        img = self._load_scan(scan_path, slice_idx)
        patient_id = self._extract_patient_id(scan_path)

        if self.transform:
            img = self.transform(img)
        
        img = torch.tensor(img).unsqueeze(0)
        return img, torch.tensor(patient_id).long(), torch.tensor(label).long()

class DatasetSequence2DBinary(Dataset):
    def __init__(self, dataset_folder, sequence_length=30, max_samples=0):
        """
        Create sequences of 2D slices for binary classification (using only NCP and Normal cases).

        :param dataset_folder: Root folder containing subfolders for each class ('NCP' and 'Normal').
        :param sequence_length: Number of slices per sequence.
        :param max_samples: Optional limit on number of sequences.
        """
        self.sequence_length = sequence_length
        self.dataset_folder = dataset_folder
        self.max_samples = max_samples
        
        # Initialize the base binary 2D dataset
        dataset = Dataset2DBinary(dataset_folder, transform=None, max_samples=max_samples)
        data = []
        
        # Load all data from the base dataset
        for i in range(len(dataset)):
            slice_image, patient_id, label = dataset[i]
            data.append((slice_image, patient_id, label))
        
        # Group slices and labels by patient_id
        patient_slices = {}
        patient_labels = {}
        for slice_image, patient_id, label in data:
            patient_id_int = patient_id.item()
            if patient_id_int not in patient_slices:
                patient_slices[patient_id_int] = []
                patient_labels[patient_id_int] = []
            patient_slices[patient_id_int].append(slice_image)
            patient_labels[patient_id_int].append(label)
        
        self.sequences = []
        self.labels = []
        self.patient_ids = []

        # Iterate over each patient to create sequences of the specified length
        for patient_id_int in patient_slices:
            slices = patient_slices[patient_id_int]
            labels = patient_labels[patient_id_int]
            num_slices = len(slices)
            num_full_sequences = num_slices // self.sequence_length
            
            for i in range(num_full_sequences):
                start_idx = i * self.sequence_length
                end_idx = start_idx + self.sequence_length
                seq_slices = slices[start_idx:end_idx]
                seq_labels = labels[start_idx:end_idx]
                seq_label = self._majority_label(seq_labels)
                self.sequences.append(seq_slices)
                self.labels.append(seq_label)
                self.patient_ids.append(patient_id_int)
        
        # Limit the number of sequences if max_samples is set
        if self.max_samples is not None and self.max_samples > 0:
            self.sequences = self.sequences[:self.max_samples]
            self.labels = self.labels[:self.max_samples]
        
        total_samples = len(self.sequences)
        samples_per_class = Counter(label.item() for label in self.labels)
        sorted_samples = sorted(samples_per_class.items())
        samples_per_class_str = ', '.join([f"{k}: {v}" for k, v in sorted_samples])
        my_logger.info(f"DatasetSequence2DBinary: Total samples: {total_samples}")
        my_logger.info(f"DatasetSequence2DBinary: Samples per class: {samples_per_class_str}")

    def _majority_label(self, labels):
        """Determine the majority label from a list of labels."""
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        return max(label_counts, key=label_counts.get)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_slices = self.sequences[idx]
        label = self.labels[idx]
        # Stack slice images into a tensor; each slice is of shape (1, 512, 512)
        seq_slices = torch.stack(seq_slices)
        return seq_slices, label

    def __init__(self, dataset_folder, sequence_length=30, max_samples=0):
        self.sequence_length = sequence_length
        self.dataset_folder = dataset_folder
        self.max_samples = max_samples
        
        # Initialize the base dataset
        dataset = Dataset2DBinary(dataset_folder, transform=None, max_samples=max_samples)
        
        data = []
        
        # Load all data from the base dataset
        for i in range(len(dataset)):
            slice_image, patient_id, label = dataset[i]
            data.append((slice_image, patient_id, label))
        
        # Group slices and labels by patient_id
        patient_slices = {}
        patient_labels = {}
        for slice_image, patient_id, label in data:
            patient_id_int = patient_id.item()
            if patient_id_int not in patient_slices:
                patient_slices[patient_id_int] = []
                patient_labels[patient_id_int] = []
            patient_slices[patient_id_int].append(slice_image)
            patient_labels[patient_id_int].append(label)
        
        self.sequences = []
        self.labels = []
        
        # Iterate over each patient to create sequences
        for patient_id_int in patient_slices:
            slices = patient_slices[patient_id_int]
            labels = patient_labels[patient_id_int]
            num_slices = len(slices)
            num_full_sequences = num_slices // self.sequence_length
            remaining_slices = num_slices % self.sequence_length
            
            # Create full sequences
            for i in range(num_full_sequences):
                start_idx = i * self.sequence_length
                end_idx = start_idx + self.sequence_length
                seq_slices = slices[start_idx:end_idx]
                seq_labels = labels[start_idx:end_idx]
                label = self._majority_label(seq_labels)
                self.sequences.append(seq_slices)
                self.labels.append(label)
            
        # Limit the number of samples if max_samples is set
        if self.max_samples > 0:
            self.sequences = self.sequences[:self.max_samples]
            self.labels = self.labels[:self.max_samples]
        
        # Log totals
        total_samples = len(self.sequences)
        samples_per_class = Counter(label.item() for label in self.labels)
        sorted_samples = sorted(samples_per_class.items())
        samples_per_class_str = ', '.join([f"{k}: {v}" for k, v in sorted_samples])
        my_logger.info(f"DatasetSequence2D: Total samples: {total_samples}")
        my_logger.info(f"DatasetSequence2D: Samples per class: {samples_per_class_str}")


    def _majority_label(self, labels):
        """Helper method to determine the majority label in a list of labels."""
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        # Return the label with the highest count
        return max(label_counts, key=label_counts.get)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_slices = self.sequences[idx]
        label = self.labels[idx]
        # Convert slice images to tensors and stack them
        seq_slices = torch.stack(seq_slices)
        return seq_slices, label