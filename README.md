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

Use the shell scripts in the `shell/` directory to train models. For example:

```bash
bash shell/train_vgg_multiclass.sh
```

> [!Note]
> Alternatively, train models using **Azure Machine Learning** by referencing the notebooks in the `/notebooks` directory. These notebooks include examples for training various models (e.g., CNN, LSTM, VGG, and ViT) using Azure's compute resources.

### Adding Custom Models or Datasets

1. Add a dataset handler to the `datasets/` directory.
2. Create or modify a training script in `scripts/train/`.
3. Update the relevant shell scripts in `shell/` as needed.

---

## Datasets

Details about datasets, including preparation and usage, are documented in the [DATASET.md](DATASET.md) file. Ensure datasets are placed in the appropriate folder structure before running training scripts.

---

## Results

Performance of trained models is summarized below:

| Model                                      | Accuracy (%) | AUC   | F1 Score | Precision | Recall |
|--------------------------------------------|--------------|-------|----------|-----------|--------|
| **VGG**                                    | 99.08        | 0.999 | 0.990    | 0.990     | 0.991  |
| **Attention-based LSTM with VGG features** | 95.19        | 0.993 | 0.951    | 0.949     | 0.954  |
| **Attention-based LSTM with 2D CNN features** | 94.67      | 0.994 | 0.946    | 0.946     | 0.946  |
| **LSTM with VGG features**                 | 94.15        | 0.993 | 0.941    | 0.939     | 0.945  |
| **LSTM with 2D CNN features**              | 94.02        | 0.992 | 0.939    | 0.939     | 0.939  |
| **ViT**                                    | 93.74        | 0.992 | 0.936    | 0.937     | 0.937  |
| **2D CNN**                                 | 92.44        | 0.987 | 0.922    | 0.923     | 0.923  |
| **3D CNN-LSTM**                            | 40.96        | 0.500 | 0.194    | 0.137     | 0.333  |

---

## Notes:

- **VGG** is the top-performing model, demonstrating exceptional feature extraction and generalization capabilities.
- **Attention-based LSTM with VGG features** achieved the second-best performance, leveraging both attention mechanisms and pre-extracted VGG features to enhance model accuracy and robustness.
- **Attention-based LSTM with 2D CNN features** outperforms **LSTM with VGG features**, showcasing the effectiveness of incorporating attention mechanisms with 2D CNN feature extraction in capturing nuanced patterns across slices.
- **LSTM with VGG features** demonstrated strong performance, effectively utilizing VGG-extracted features to model temporal dependencies, though slightly below its attention-based counterparts.
- **LSTM with 2D CNN features** utilized sequential modeling effectively, performing comparably to its VGG-based variant but without the added benefits of attention mechanisms.
- **ViT** continued to show the promise of attention mechanisms but fell short compared to models that leveraged VGG features and LSTMs.
- **2D CNN** performed well but lacked the sequential or attention-based improvements seen in LSTM-based models.
- **3D CNN-LSTM** struggled to capture spatiotemporal relationships effectively, leading to significantly lower performance.

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). Please review the license before use.