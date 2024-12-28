import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import random
from collections import Counter
from utils.log_config import get_custom_logger

my_logger = get_custom_logger('ccccii_dataset')

class CCCCIIDataset2D(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=0):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = {'CP': 0, 'NCP': 1, 'Normal': 2}
        self.num_classes = len(self.classes)  # Number of classes
        self.max_samples = max_samples
        
        my_logger.info(f"CCCCIIDataset2D: Dataset initialized with root_dir: {self.root_dir}")
        my_logger.info(f"CCCCIIDataset2D: Transform: {self.transform}")
        my_logger.info(f"CCCCIIDataset2D: Max samples: {self.max_samples}")
        my_logger.info(f"CCCCIIDataset2D: Number of classes: {self.num_classes}")
        my_logger.info(f"CCCCIIDataset2D: Labels : {self.classes}")
        
        self.data = self._gather_data()

    def _gather_data(self):
        data = []
        patient_list = []
        sample_counts = {class_idx: 0 for class_idx in self.classes.values()}  # Initialize counters per class

        my_logger.info(f"CCCCIIDataset2D: Current working directory: {os.getcwd()}")
        my_logger.info(f"CCCCIIDataset2D: Contents of root directory ({self.root_dir}): {os.listdir(self.root_dir)}")

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
            slices = sorted(os.listdir(scan_path))
            num_slices = len(slices)
            # Calculate start and end indices for central 30 slices
            start_idx = max(0, (num_slices - 30) // 2)
            end_idx = start_idx + 30
            end_idx = min(end_idx, num_slices)  # Ensure end_idx doesn't exceed total slices

            for slice_idx in range(start_idx, end_idx):
                data.append((scan_path, slice_idx, class_idx))
                sample_counts[class_idx] += 1
                count += 1
                if self.max_samples > 0 and count >= self.max_samples:
                    my_logger.info(f"CCCCIIDataset2D: Total samples : {len(data)}")
                    my_logger.info(f"CCCCIIDataset2D: Samples per class : {sample_counts}")
                    return data
            # Move to the next patient once one scan is processed

        my_logger.info(f"CCCCIIDataset2D: Total samples : {len(data)}")
        my_logger.info(f"CCCCIIDataset2D: Samples per class : {sample_counts}")
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
            my_logger.info(f"CCCCIIDataset2D:Image is zero std: {scan_path} - {slice_idx}")            
            img = img - mean  # If std is zero, simply subtract the mean        
        return img

    def _extract_patient_id(self, scan_path):
        # Get the immediate parent directory of the scan path
        patient_id = os.path.basename(os.path.dirname(scan_path))
        return int(patient_id)

    def __getitem__(self, idx):
        scan_path, slice_idx, label = self.data[idx]
        img = self._load_scan(scan_path, slice_idx)
        
        patient_id = self._extract_patient_id(scan_path)

        if self.transform:
            img = self.transform(img)
        
        # Ensure the image is of shape (1, 512, 512) by adding a channel dimension
        img = torch.tensor(img).unsqueeze(0)  # Add the channel dimension
        
        # Return the image, patient ID, and the label
        return img, torch.tensor(patient_id).long(), torch.tensor(label).long()


import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
from collections import Counter
from utils.log_config import get_custom_logger

my_logger = get_custom_logger('ccciiidataset')

class CCCIIIDataset3D(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=0):
        """
        A dataset class to load 3D volumes composed of 30 central slices
        from CT scans of COVID-19 (CP), common pneumonia (NCP), and Normal cases.

        :param root_dir: Path to the root directory containing the subfolders 'CP', 'NCP', and 'Normal'.
        :param transform: Optional transform to be applied on each slice.
        :param max_samples: Maximum number of volumes to be loaded. If 0, all are loaded.
        """
        self.root_dir = root_dir
        self.transform = transform
        # Map your classes to numeric labels
        self.classes = {'CP': 0, 'NCP': 1, 'Normal': 2}
        self.num_classes = len(self.classes)  # Number of classes
        self.max_samples = max_samples
        
        my_logger.info(f"CCCIIIDataset3D: Dataset initialized with root_dir: {self.root_dir}")
        my_logger.info(f"CCCIIIDataset3D: Transform: {self.transform}")
        my_logger.info(f"CCCIIIDataset3D: Max samples: {self.max_samples}")
        my_logger.info(f"CCCIIIDataset3D: Number of classes: {self.num_classes}")
        my_logger.info(f"CCCIIIDataset3D: Labels : {self.classes}")
        
        # Gather data (one entry per scan that has at least 30 slices)
        self.data = self._gather_data()

    def _gather_data(self):
        """
        Collect scan paths from all classes and store only those
        with at least 30 slices.
        
        Returns a list of tuples: (scan_path, start_idx, end_idx, class_idx).
        """
        data = []
        patient_list = []
        
        sample_counts = {class_idx: 0 for class_idx in self.classes.values()}  # Counter per class

        my_logger.info(f"CCCIIIDataset3D: Current working directory: {os.getcwd()}")
        my_logger.info(f"CCCIIIDataset3D: Contents of root directory ({self.root_dir}): {os.listdir(self.root_dir)}")
        
        # Collect scans
        for label, class_idx in self.classes.items():
            class_dir = os.path.join(self.root_dir, label)
            patients = os.listdir(class_dir)
            for patient_id in patients:
                patient_dir = os.path.join(class_dir, patient_id)
                for scan_folder in os.listdir(patient_dir):
                    scan_path = os.path.join(patient_dir, scan_folder)
                    slices = os.listdir(scan_path)
                    
                    # Only use scans with at least 30 slices
                    if len(slices) >= 30:
                        patient_list.append((scan_path, class_idx))
        
        # Shuffle data
        random.seed(42)  # For reproducibility
        random.shuffle(patient_list)

        count = 0
        final_data = []
        for scan_path, class_idx in patient_list:
            slices = sorted(os.listdir(scan_path))
            num_slices = len(slices)
            
            # Calculate start and end indices for the central 30 slices
            start_idx = max(0, (num_slices - 30) // 2)
            end_idx = start_idx + 30
            end_idx = min(end_idx, num_slices)

            # Store the information needed to load the 3D volume
            final_data.append((scan_path, start_idx, end_idx, class_idx))
            sample_counts[class_idx] += 1
            count += 1

            # If max_samples > 0, limit the total number of samples
            if self.max_samples > 0 and count >= self.max_samples:
                my_logger.info(f"CCCIIIDataset3D: Total volumes: {len(final_data)}")
                my_logger.info(f"CCCIIIDataset3D: Volumes per class: {sample_counts}")
                return final_data

        my_logger.info(f"CCCIIIDataset3D: Total volumes: {len(final_data)}")
        my_logger.info(f"CCCIIIDataset3D: Volumes per class: {sample_counts}")
        return final_data

    def __len__(self):
        return len(self.data)

    def _load_volume(self, scan_path, start_idx, end_idx):
        """
        Load and stack slices into a 3D volume (shape: 30, 512, 512).

        Each slice is:
        - loaded as grayscale
        - resized to 512x512
        - normalized (mean 0, std 1) if std != 0
        """
        slices = sorted(os.listdir(scan_path))[start_idx:end_idx]
        volume_slices = []

        for slice_file in slices:
            slice_path = os.path.join(scan_path, slice_file)
            img = Image.open(slice_path).convert('L')  # Grayscale
            img = img.resize((512, 512))
            img = np.array(img).astype(np.float32)
            
            mean = np.mean(img)
            std = np.std(img)
            if std > 0:
                img = (img - mean) / std
            else:
                my_logger.info(f"CCCIIIDataset3D: Image has zero std: {slice_path}")
                img = img - mean
            
            volume_slices.append(img)
        
        # Stack into shape (30, 512, 512)
        volume = np.stack(volume_slices, axis=0)
        return volume

    def _extract_patient_id(self, scan_path):
        """
        Extract patient_id from the directory structure.
        Example: root_dir/NCP/9999/9999/<slice_files> => patient_id = 9999
        """
        patient_id = os.path.basename(os.path.dirname(scan_path))
        return int(patient_id)

    def __getitem__(self, idx):
        """
        Return:
            volume: Torch tensor of shape (1, 30, 512, 512)
            patient_id: Tensor containing the patient_id
            label: Tensor containing the class label
        """
        scan_path, start_idx, end_idx, label = self.data[idx]
        volume = self._load_volume(scan_path, start_idx, end_idx)
        patient_id = self._extract_patient_id(scan_path)
        
        # Apply any additional transforms (e.g. augmentations)
        # Note: Typically, 3D transforms differ from 2D transforms, 
        # so use caution if applying standard 2D transforms.
        if self.transform:
            volume = self.transform(volume)  # Transform expected to handle 3D data

        # Convert to torch and add channel dimension => (1, D, H, W)
        volume = torch.tensor(volume).unsqueeze(0)  # shape: (1, 30, 512, 512)

        return volume, torch.tensor(patient_id).long(), torch.tensor(label).long()



class CCCCIIDatasetSequence2D(Dataset):
    def __init__(self, dataset_folder, sequence_length=30, max_samples=None):
        self.sequence_length = sequence_length
        self.dataset_folder = dataset_folder
        self.max_samples = max_samples
        
        # Initialize the base dataset
        dataset = CCCCIIDataset2D(dataset_folder, transform=None, max_samples=max_samples)
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
        my_logger.info(f"CCCCCIDatasetSequence2D: Total samples: {total_samples}")
        my_logger.info(f"CCCCCIDatasetSequence2D: Samples per class: {samples_per_class_str}")


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