# PulmoSense: Intelligent Lung Pathology Detection

This repository contains the code, resources, and documentation for the **PulmoSense** research project, which focuses on detecting and classifying COVID-19 and common pneumonia in CT scans using advanced deep learning techniques.

---

## Overview

PulmoSense aims to leverage deep learning to analyze and classify pulmonary pathologies in CT scans, with a specific focus on **COVID-19**, **Non-COVID Pneumonia (NCP)**, and **Normal** cases. This repository includes:

- End-to-end training pipelines for various models.
- Comprehensive scripts for dataset handling, model training, and evaluation.
- Predefined experiments and results documentation.

---

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd placerda-pulmo-sense
   ```

2. Set up a Python environment:
   ```bash
   conda create -n pulmo-sense python=3.11
   conda activate pulmo-sense
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Directory Structure

- **`datasets/`**: Dataset classes and statistics scripts.
- **`environment/`**: Configuration files for the project environment, including Docker.
- **`notebooks/`**: Jupyter notebooks for model training and experimentation.
- **`scripts/`**: Training scripts for models organized by architecture.
- **`shell/`**: Shell scripts for automated dataset downloads and training tasks.
- **`utils/`**: Utility scripts for logging, downloading, and requirement checks.
- **`README.md`**: Documentation for understanding and using this repository.
- **`DATASET.md`**: Details about datasets used in this project.

---

## Usage

### Training Models

To train models, use the shell scripts provided in the `shell/` directory. Example:

```bash
bash shell/train_vgg_multiclass.sh
```

### Adding Custom Models or Datasets

1. Add a dataset handler in the `datasets/` directory.
2. Update or create a training script in `scripts/train/`.
3. Customize shell scripts in `shell/`.

---

## Datasets

Details about datasets, including preparation and usage, are documented in the [DATASET.md](DATASET.md) file. Ensure datasets are placed in the appropriate folder structure before running training scripts.

---

## Results

Performance of trained models is summarized below:

| Model                  | Accuracy (%) | AUC    | F1 Score | Precision | Recall   |
|------------------------|--------------|--------|----------|-----------|----------|
| **VGG**               | 99.08        | 0.999  | 0.990    | 0.990     | 0.991    |
| **LSTM**              | 94.02        | 0.992  | 0.939    | 0.939     | 0.939    |
| **ViT**               | 93.74        | 0.992  | 0.936    | 0.937     | 0.937    |
| **2D CNN**            | 92.44        | 0.987  | 0.922    | 0.923     | 0.923    |
| **3D CNN-LSTM**       | 40.96        | 0.500  | 0.194    | 0.137     | 0.333    |

**Observations**:
- **VGG** achieved the best overall performance.
- Temporal models like **LSTM** demonstrate strong performance for sequence data.
- **3D CNN-LSTM** requires optimization for spatiotemporal feature extraction.

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). Please review the license before use.
