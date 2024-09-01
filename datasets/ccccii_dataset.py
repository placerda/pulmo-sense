import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

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
                        # Armazena os caminhos e índices dos 30 slices centrais
                        slice_indices = range(len(slices) // 2 - 15, len(slices) // 2 + 15)
                        for idx in slice_indices:
                            data.append((scan_path, idx, class_idx))
                        break
        return data

    def __len__(self):
        return len(self.data)

    def _load_scan(self, scan_path, slice_idx):
        slice_files = sorted(os.listdir(scan_path))
        slice_file = slice_files[slice_idx]
        slice_path = os.path.join(scan_path, slice_file)
        img = Image.open(slice_path).convert('L')  # Converter para grayscale
        img = img.resize((512, 512))  # Redimensionar para 512x512
        img = np.array(img)
        img = img.astype(np.float32)
        img = (img - np.mean(img)) / np.std(img)  # Normalizar intensidades
        return img

    def _extract_patient_id(self, scan_path):
        # Divide o caminho para obter a parte correspondente ao diretório do paciente
        patient_dir = os.path.dirname(scan_path)
        # Divide novamente para obter o nome do diretório do paciente, que é o ID do paciente
        patient_id = os.path.basename(os.path.dirname(patient_dir))
        return patient_id

    
    def __getitem__(self, idx):
        scan_path, slice_idx, label = self.data[idx]
        img = self._load_scan(scan_path, slice_idx)
        
        patient_id = self._extract_patient_id(scan_path)

        if self.transform:
            img = self.transform(img)
        
        # Garantir que a imagem tenha o formato (1, 512, 512) adicionando uma dimensão de canal
        img = torch.tensor(img).unsqueeze(0)  # Adiciona a dimensão do canal
        
        # Retorna a imagem, o rótulo e o ID do paciente (ou outro identificador relevante)
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

