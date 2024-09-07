import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import random

from utils.log_config import get_custom_logger

my_logger = get_custom_logger('ccccii_dataset')

class CCCCIIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = {'CP': 0, 'NCP': 1, 'Normal': 2}
        self.data = self._gather_data()

    def _gather_data(self):
        data = []
        for label, class_idx in self.classes.items():
            class_dir = os.path.join(self.root_dir, label)
            for patient_id in os.listdir(class_dir):
                patient_dir = os.path.join(class_dir, patient_id)
                for scan_folder in os.listdir(patient_dir):
                    scan_path = os.path.join(patient_dir, scan_folder)
                    slices = os.listdir(scan_path)
                    if len(slices) >= 30:
                        # Store only the first scan with more than 30 slices
                        data.append((scan_path, class_idx))
                        break
        return data

    def __len__(self):
        return len(self.data)

    def _load_scan(self, scan_path):
        slice_files = sorted(os.listdir(scan_path))
        # Select 30 central slices
        start_idx = len(slice_files) // 2 - 15
        selected_slices = slice_files[start_idx:start_idx + 30]
        scan = []
        for slice_file in selected_slices:
            slice_path = os.path.join(scan_path, slice_file)
            img = Image.open(slice_path).convert('L')  # Convert to grayscale
            img = img.resize((512, 512))  # Resize to 512x512
            img = np.array(img)
            scan.append(img)
        scan = np.stack(scan, axis=0)
        scan = scan.astype(np.float32)
        scan = (scan - np.mean(scan)) / np.std(scan)  # Normalize scan intensities
        return scan

    def __getitem__(self, idx):
        scan_path, label = self.data[idx]
        scan = self._load_scan(scan_path)
        if self.transform:
            scan = self.transform(scan)
        # Ensure the scan is of shape (1, 30, 512, 512) by adding a channel dimension
        scan = torch.tensor(scan).unsqueeze(0)  # Add the channel dimension only
        # Return label as a LongTensor
        return scan, torch.tensor(label).long()


class CCCCIIDataset2D(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = {'CP': 0, 'NCP': 1, 'Normal': 2}
        self.num_classes = len(self.classes)  # Number of classes
        self.max_samples = max_samples
        
        my_logger.info(f"[cccii_dataset] Dataset initialized with root_dir: {self.root_dir}")
        my_logger.info(f"[cccii_dataset] Transform: {self.transform}")
        my_logger.info(f"[cccii_dataset] Max samples: {self.max_samples}")
        my_logger.info(f"[cccii_dataset] Number of classes: {self.num_classes}")
        my_logger.info(f"[cccii_dataset] Labels : {self.classes}")
        
        self.data = self._gather_data()

    def _gather_data(self):
        data = []
        patient_list = []
        sample_counts = {class_idx: 0 for class_idx in self.classes.values()}  # Initialize counters per class

        my_logger.info(f"[cccii_dataset] Current working directory: {os.getcwd()}") 
        my_logger.info(f"[cccii_dataset] Gathering data from root directory: {self.root_dir}")
        my_logger.info(f"[cccii_dataset] Contents of root directory: {os.listdir(self.root_dir)}")

        # Collect all patients across all classes
        for label, class_idx in self.classes.items():
            class_dir = os.path.join(self.root_dir, label)
            patients = os.listdir(class_dir)
            for patient_id in patients:
                patient_dir = os.path.join(class_dir, patient_id)
                for scan_folder in os.listdir(patient_dir):
                    scan_path = os.path.join(patient_dir, scan_folder)
                    slices = os.listdir(scan_path)
                    if len(slices) >= 30:
                        patient_list.append((scan_path, class_idx))  # Collect scan paths with class labels

        random.seed(42)  # Set the random seed for reproducibility
        random.shuffle(patient_list)  # Shuffle the patient list across all classes

        count = 0
        for scan_path, class_idx in patient_list:
            slices = os.listdir(scan_path)
            for slice_idx in range(len(slices)):
                data.append((scan_path, slice_idx, class_idx))
                sample_counts[class_idx] += 1
                count += 1
                if self.max_samples and count >= self.max_samples:
                    my_logger.info(f"[cccii_dataset] Total samples : {len(data)}")
                    my_logger.info(f"[cccii_dataset] Samples per class : {sample_counts}")
                    return data
            # Move to the next patient once one scan is processed

        my_logger.info(f"[cccii_dataset] Total samples : {len(data)}")
        my_logger.info(f"[cccii_dataset] Samples per class : {sample_counts}")
        return data

    def __len__(self):
        return len(self.data)

    def _load_scan(self, scan_path, slice_idx):
        slice_files = sorted(os.listdir(scan_path))
        slice_file = slice_files[slice_idx]
        slice_path = os.path.join(scan_path, slice_file)
        img = Image.open(slice_path).convert('L')  # Convert to grayscale
        img = img.resize((512, 512))  # Resize to 512x512
        img = np.array(img)
        img = img.astype(np.float32)
        np.seterr(invalid='raise')
        mean = np.mean(img)
        std = np.std(img)        
        if std > 0:
            img = (img - mean) / std  # Normalize intensities only if std is non-zero
        else:
            my_logger.info(f"[cccii_dataset]Image is zero std: {scan_path} - {slice_idx}")            
            img = img - mean  # If std is zero, simply subtract the mean        
        return img

    def _extract_patient_id(self, scan_path):
        # Split the path to get the directory corresponding to the patient
        patient_dir = os.path.dirname(scan_path)
        # Split again to get the name of the patient directory, which is the patient ID
        patient_id = os.path.basename(os.path.dirname(patient_dir))
        return patient_id

    def __getitem__(self, idx):
        scan_path, slice_idx, label = self.data[idx]
        img = self._load_scan(scan_path, slice_idx)
        
        patient_id = self._extract_patient_id(scan_path)

        if self.transform:
            img = self.transform(img)
        
        # Ensure the image is of shape (1, 512, 512) by adding a channel dimension
        img = torch.tensor(img).unsqueeze(0)  # Add the channel dimension
        
        # Return the image, patient ID, and the label
        return img, patient_id, torch.tensor(label).long()

    
class CCCCIIBinaryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = {'Normal': 0, 'CP': 1}
        self.data = self._gather_data()

    def _gather_data(self):
        data = []
        for label, class_idx in self.classes.items():
            class_dir = os.path.join(self.root_dir, label)
            for patient_id in os.listdir(class_dir):
                patient_dir = os.path.join(class_dir, patient_id)
                for scan_folder in os.listdir(patient_dir):
                    scan_path = os.path.join(patient_dir, scan_folder)
                    slices = os.listdir(scan_path)
                    if len(slices) >= 30:
                        # Store only the first scan with more than 30 slices
                        data.append((scan_path, class_idx))
                        break
        return data

    def __len__(self):
        return len(self.data)

    def _load_scan(self, scan_path):
        slice_files = sorted(os.listdir(scan_path))
        # Select 30 central slices
        start_idx = len(slice_files) // 2 - 15
        selected_slices = slice_files[start_idx:start_idx + 30]
        scan = []
        for slice_file in selected_slices:
            slice_path = os.path.join(scan_path, slice_file)
            img = Image.open(slice_path).convert('L')  # Convert to grayscale
            img = img.resize((512, 512))  # Resize to 512x512
            img = np.array(img)
            scan.append(img)
        scan = np.stack(scan, axis=0)
        scan = scan.astype(np.float32)
        scan = (scan - np.mean(scan)) / np.std(scan)  # Normalize scan intensities
        return scan

    def __getitem__(self, idx):
        scan_path, label = self.data[idx]
        scan = self._load_scan(scan_path)
        if self.transform:
            scan = self.transform(scan)
        # Ensure the scan is of shape (1, 30, 512, 512) by adding a channel dimension
        scan = torch.tensor(scan).unsqueeze(0)  # Add the channel dimension only
        return scan, torch.tensor(label, dtype=torch.float32)