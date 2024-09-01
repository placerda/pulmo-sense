import os
import json
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from scipy.ndimage import zoom
import pydicom
from tabulate import tabulate

MOSMED_DATASET_NAME='mosmed'
COVIDCTMD_DATASET_NAME='covidctmd'
LUNA16_DATASET_NAME='luna16'

from utils.log_config import get_custom_logger

my_logger = get_custom_logger('combined_dataset')

# Constants for spacing
NEW_SPACING_XY = 0.6
NEW_SPACING_Z = 8

def normalize(image):
    return (image - image.mean()) / image.std()

def resample(image, old_spacing, new_spacing=[NEW_SPACING_XY, NEW_SPACING_XY, NEW_SPACING_Z]):
    resize_factor = old_spacing / new_spacing
    new_shape = image.shape * resize_factor
    rounded_new_shape = np.round(new_shape)
    resize_factor_actual = rounded_new_shape / image.shape
    new_spacing = old_spacing / resize_factor_actual
    resampled_image = zoom(image, resize_factor_actual, mode='nearest')
    return resampled_image

def center_crop(image, new_shape):
    center = np.array(image.shape) // 2
    start = center - new_shape // 2
    end = start + new_shape
    slices = tuple(slice(start[i], end[i]) for i in range(len(new_shape)))
    return image[slices]

class MosMedDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'covid_labels')
        self.samples = os.listdir(self.image_dir)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.samples[idx], 'image.npy')
        label_path = os.path.join(self.label_dir, self.samples[idx], 'covid_label.json')
        spacing_path = os.path.join(self.image_dir, self.samples[idx], 'spacing.json')

        image = np.load(image_path)
        with open(label_path, 'r') as f:
            label_data = json.load(f)
        with open(spacing_path, 'r') as f:
            spacing = json.load(f)

        image = resample(image, np.array(spacing))
        image = normalize(image)
        
        # Select 30 central slices
        z_center = image.shape[0] // 2
        z_start = z_center - 15
        z_end = z_center + 15
        image = image[z_start:z_end, :, :]
        
        # Resize to 512x512
        image_resized = np.zeros((30, 512, 512))
        for i in range(30):
            image_resized[i] = zoom(image[i], (512 / image.shape[1], 512 / image.shape[2]), mode='nearest')
        
        # Ensure label is processed correctly
        label = 1 if label_data else 0  # 1 for true, 0 for false

        return torch.from_numpy(image_resized), torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)    

    def get_class_counts(self):
        counts = defaultdict(int)
        for sample in self.samples:
            label_path = os.path.join(self.label_dir, sample, 'covid_label.json')
            with open(label_path, 'r') as f:
                label_data = json.load(f)
            label = 1 if label_data else 0
            if label == 1:
                counts['COVID-19 Cases'] += 1
            else:
                counts['Normal Cases'] += 1
        return counts
    
class CovidCtMdDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.cap_cases_dir = os.path.join(root_dir, 'Cap Cases')
        self.covid_cases_dir = os.path.join(root_dir, 'COVID-19 Cases')
        self.normal_cases_dir = os.path.join(root_dir, 'Normal Cases')
        self.samples = self._collect_samples()

    def _collect_samples(self):
        samples = []
        for case_type in ['Cap Cases', 'COVID-19 Cases', 'Normal Cases']:
            case_dir = os.path.join(self.root_dir, case_type)
            for study in os.listdir(case_dir):
                samples.append((case_type, study))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        case_type, study = self.samples[idx]
        case_dir = os.path.join(self.root_dir, case_type, study)
        slices = [pydicom.dcmread(os.path.join(case_dir, f)).pixel_array for f in os.listdir(case_dir)]
        image = np.stack(slices)
        
        image = normalize(image)
        
        # Select 30 central slices
        z_center = image.shape[0] // 2
        z_start = z_center - 15
        z_end = z_center + 15
        image = image[z_start:z_end, :, :]
        
        # Resize to 512x512
        image_resized = np.zeros((30, 512, 512))
        for i in range(30):
            image_resized[i] = zoom(image[i], (512/image.shape[1], 512/image.shape[2]), mode='nearest')
        
        label = 1 if case_type != 'Normal Cases' else 0  # 1 for abnormal, 0 for normal
        return torch.from_numpy(image_resized), torch.tensor(label, dtype=torch.float32)

    def get_class_counts(self):
        counts = defaultdict(int)
        for case_type, _ in self.samples:
            counts[case_type] += 1
        return counts

class Luna16Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.subsets = [os.path.join(root_dir, f'subset{i}') for i in range(10)]
        self.samples = self._collect_samples()

    def _collect_samples(self):
        samples = []
        for subset in self.subsets:
            for study in os.listdir(subset):
                if study.endswith('.mhd'):
                    samples.append((subset, study))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subset, study = self.samples[idx]
        mhd_path = os.path.join(subset, study)
        raw_path = mhd_path.replace('.mhd', '.raw')
        
        image = np.fromfile(raw_path, dtype=np.uint16).reshape([512, 512, -1])  # Assuming fixed size, adapt if needed
        image = np.transpose(image, (2, 0, 1))  # Transpose to [depth, height, width]
        
        image = normalize(image)
        
        # Select 30 central slices
        z_center = image.shape[0] // 2
        z_start = z_center - 15
        z_end = z_center + 15
        image = image[z_start:z_end, :, :]
        
        # Resize to 512x512
        image_resized = np.zeros((30, 512, 512))
        for i in range(30):
            image_resized[i] = zoom(image[i], (512/image.shape[1], 512/image.shape[2]), mode='nearest')
        
        label = 1  # All samples in Luna16 are abnormal
        return torch.from_numpy(image_resized), torch.tensor(label, dtype=torch.float32)

    def get_class_counts(self):
        counts = {'Nodules': len(self.samples)}
        return counts

class CombinedDataset(Dataset):
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], list):
            self.init_list(args[0])
        elif len(args) == 3 and all(isinstance(arg, str) for arg in args):
            # arg 1 is mosmed path, arg 2 is covidctmd path, arg 3 is luna16 path
            self.init_paths(args[0], args[1], args[2])
        else:
            raise ValueError("Invalid arguments. Provide either a list of dataset names or three dataset paths.")

    def init_list(self, dataset_list):
        my_logger.info('Using CombinedDataset...')
        
        self.datasets = []
        self.cumulative_lengths = []

        if MOSMED_DATASET_NAME in dataset_list:
            self.mosmed_dataset = MosMedDataset(f"{MOSMED_DATASET_NAME}/")
            self.datasets.append(self.mosmed_dataset)
            my_logger.info('Using MosMedDataset...')
        
        if COVIDCTMD_DATASET_NAME in dataset_list:
            self.covidctmd_dataset = CovidCtMdDataset(f"{COVIDCTMD_DATASET_NAME}/")
            self.datasets.append(self.covidctmd_dataset)
            my_logger.info('Using CovidCtMdDataset...')
        
        if LUNA16_DATASET_NAME in dataset_list:
            self.luna16_dataset = Luna16Dataset(f"{LUNA16_DATASET_NAME}/")
            self.datasets.append(self.luna16_dataset)
            my_logger.info('Using Luna16Dataset...')
        
        self.cumulative_lengths = self._compute_cumulative_lengths()

    def init_paths(self, mosmed_path, covidctmd_path, luna16_path):
        my_logger.info('Using CombinedDataset with paths...')
        
        self.datasets = []
        self.cumulative_lengths = []

        self.mosmed_dataset = MosMedDataset(mosmed_path)
        self.datasets.append(self.mosmed_dataset)
        my_logger.info('Using MosMedDataset with path...')

        self.covidctmd_dataset = CovidCtMdDataset(covidctmd_path)
        self.datasets.append(self.covidctmd_dataset)
        my_logger.info('Using CovidCtMdDataset with path...')

        self.luna16_dataset = Luna16Dataset(luna16_path)
        self.datasets.append(self.luna16_dataset)
        my_logger.info('Using Luna16Dataset with path...')
        
        self.cumulative_lengths = self._compute_cumulative_lengths()        

    def _compute_cumulative_lengths(self):
        lengths = [len(d) for d in self.datasets]
        cumulative_lengths = np.cumsum(lengths)
        return cumulative_lengths

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths.size > 0 else 0

    def __getitem__(self, idx):
        dataset_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        if dataset_idx == 0:
            return self.datasets[dataset_idx][idx]
        else:
            return self.datasets[dataset_idx][idx - self.cumulative_lengths[dataset_idx - 1]]

    def get_class_counts(self):
        combined_counts = defaultdict(int)
        for dataset in self.datasets:
            dataset_counts = dataset.get_class_counts()
            for key, value in dataset_counts.items():
                combined_counts[key] += value
        return combined_counts

def datasets_statistics(mosmed_root, covidctmd_root, luna16_root):
    mosmed_dataset = MosMedDataset(mosmed_root)
    covidctmd_dataset = CovidCtMdDataset(covidctmd_root)
    luna16_dataset = Luna16Dataset(luna16_root)
    combined_dataset = CombinedDataset(mosmed_root, covidctmd_root, luna16_root)

    datasets = [mosmed_dataset, covidctmd_dataset, luna16_dataset, combined_dataset]
    dataset_names = ['MosMedDataset', 'CovidCtMdDataset', 'Luna16Dataset', 'CombinedDataset']

    samples_table = []
    train_val_table = []

    for dataset, name in zip(datasets, dataset_names):
        class_counts = dataset.get_class_counts()
        total_samples = len(dataset)

        if name == 'CombinedDataset':
            abnormal_samples = (
                class_counts.get('COVID-19 Cases', 0) + 
                class_counts.get('Cap Cases', 0) + 
                class_counts.get('Nodules', 0)
            )
            normal_samples = class_counts.get('Normal Cases', 0)
            
            class_counts_combined = {'Abnormal': abnormal_samples, 'Normal': normal_samples}
            class_counts_str = ', '.join([f"{k}: {v}" for k, v in class_counts_combined.items()])
            percentages_str = ', '.join([f"{k}: {v/total_samples*100:.2f}%" for k, v in class_counts_combined.items()])

            train_normal = 0.8 * normal_samples
            train_abnormal = 0.8 * abnormal_samples
            val_normal = 0.2 * normal_samples
            val_abnormal = 0.2 * abnormal_samples

            train_val_class_split = (
                f"Train: Normal {train_normal:.0f} Abnormal {train_abnormal:.0f} - "
                f"Val: Normal {val_normal:.0f} Abnormal {val_abnormal:.0f}"
            )
            train_val_percentages = (
                f"Train: Normal {train_normal/(train_normal + train_abnormal)*100:.2f}% "
                f"Abnormal {train_abnormal/(train_normal + train_abnormal)*100:.2f}% - "
                f"Val: Normal {val_normal/(val_normal + val_abnormal)*100:.2f}% "
                f"Abnormal {val_abnormal/(val_normal + val_abnormal)*100:.2f}%"
            )
        else:
            class_counts_str = ', '.join([f"{k}: {v}" for k, v in class_counts.items()])
            percentages_str = ', '.join([f"{k}: {v/total_samples*100:.2f}%" for k, v in class_counts.items()])
            
            train_val_class_split = f"Train: " + ' '.join([f"{k.split()[0]} {0.8 * v:.0f}" for k, v in class_counts.items()]) + " - Val: " + ' '.join([f"{k.split()[0]} {0.2 * v:.0f}" for k, v in class_counts.items()])
            train_val_percentages = f"Train: " + ' '.join([f"{k.split()[0]} {0.8 * v/(0.8 * total_samples)*100:.2f}%" for k, v in class_counts.items()]) + " - Val: " + ' '.join([f"{k.split()[0]} {0.2 * v/(0.2 * total_samples)*100:.2f}%" for k, v in class_counts.items()])
        
        samples_table.append([name, total_samples, class_counts_str, percentages_str])
        train_val_table.append([name, train_val_class_split, train_val_percentages])

    samples_markdown_table = tabulate(samples_table, headers=["Dataset", "Samples", "Samples by Class", "Percentages by Class"], tablefmt="pipe")
    train_val_markdown_table = tabulate(train_val_table, headers=["Dataset", "Train/Val by Class", "Percentages by Class"], tablefmt="pipe")
    
    return samples_markdown_table, train_val_markdown_table


# Example usage
# table1, table2 = datasets_statistics('path_to_mosmed_data', 'path_to_covidctmd_data', 'path_to_luna16_data')
# print(table1)
# print(table2)