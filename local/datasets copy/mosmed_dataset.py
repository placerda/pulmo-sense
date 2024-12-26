import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom

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

    def __len__(self):
        return len(self.samples)

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
            image_resized[i] = zoom(image[i], (512/image.shape[1], 512/image.shape[2]), mode='nearest')
        
        label = 1 if label_data.get('covid', False) else 0  # 1 for abnormal, 0 for normal
        return torch.from_numpy(image_resized), torch.tensor(label, dtype=torch.float32)

# Example usage
# dataset = MosMedDataset(root_dir='path_to_data')
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
