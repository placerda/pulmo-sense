# PulmoSense: Intelligent Lung Pathology Detection

This repository hosts the code, resources, and documentation for the doctoral research project "PulmoSense". The project focuses on the detection of COVID-19 amd Common Pneumonia in CT scans using advanced Deep Learning techniques.

## Overview

"PulmoSense" is a project that aims to leverage the power of Deep Learning for the detection and classification of pulmonary pathologies in CT scans. The focus is on identifying cases of COVID-19 and Common Pneumonia. This repository provides comprehensive instructions to reproduce the experiments and access the datasets used in the study.

## Getting Started

To set up the project, follow these steps:

1. Clone the repository by using the command: `git clone <repository_url>`
2. Create a conda environment by using the command: `conda create -n pulmo-sense python=3.11`
3. Activate the conda environment by using the command: `conda activate pulmo-sense`
4. Install the necessary dependencies by using the command: `pip install -r requirements.txt`

## Folder structure

1. **`datasets/`**: Contains different dataset classes.
2. **`models/`**: Contains different model architectures.
3. **`scripts/`**: Contains training, testing, and prediction (inference) scripts for models, organized into separate subdirectories.
   - **`train/`**: Contains scripts for training models.
   - **`test/`**: Contains scripts for testing models.
   - **`predict/`**: Contains scripts for model inference.
4. **`utils/`**: Contains utility functions and helper scripts.
5. **`config/`**: Contains configuration files for different models.
6. **`logs/`**: Directory to store log files generated during training/testing.
7. **`results/`**: Directory to store results such as model checkpoints, evaluation results, etc.
8. **`tests/`**: Contains unit tests, organized into subdirectories for datasets and models to avoid confusion with model testing scripts.
9. **`README.md`**: A markdown file to provide an overview of the project.
10. **`requirements.txt`**: Lists all the Python packages required to run your project.

<!-- ## Usage

To reproduce the experiments and execute the classification model, follow the steps detailed in the "Usage" section. -->

## Datasets

The datasets used in this research are detailed in [this](DATASET.md) page.

## Results

The "Results" section provides a summary of the outcomes from the experiments conducted during this study, particularly those related to the detection of COVID-19 and bacterial pneumonia.

### Dataset Details

The dataset used in this study includes samples distributed across three classes: Normal, COVID-19, and Non-COVID Pneumonia. Below is a summary of the total volumes and class distribution for different models, along with the train-validation split:

| Model                  | Total Samples | Class Distribution (Normal, COVID-19, Non-COVID Pneumonia) | Train Samples | Validation Samples | Network Input      |
|------------------------|---------------|-----------------------------|---------------|---------------------|--------------------|
| **VGG**               | 115,260       | 40,410 / 43,470 / 31,380   | 92,208        | 23,052             | 2D                |
| **ViT**               | 115,260       | 40,410 / 43,470 / 31,380   | 92,208        | 23,052             | 2D                |
| **2D CNN**            | 115,260       | 40,410 / 43,470 / 31,380   | 92,208        | 23,052             | 2D                |
| **3D CNN-LSTM**       | 3,842         | 1,347 / 1,449 / 1,046      | 3,073         | 769                | 3D                |
| **LSTM**              | 3,842         | 1,347 / 1,449 / 1,046      | 3,074         | 768                | Sequence of Features |

**Note:** For the CNN-LSTM and LSTM models, the input differs from other architectures. Instead of raw CT slices, these models use sequences representing the central portion of each CT study, comprising 30 slices. For CNN-LSTM, the input is 3D, while for LSTM, the input is a sequence of feature vectors extracted by the 2D CNN model.

### Results Summary

Below is the comparison of the models based on their performance metrics (validation with hold-out):

| Model                  | Accuracy (%) | AUC    | F1 Score | Precision | Recall   |
|------------------------|--------------|--------|----------|-----------|----------|
| **VGG**               | 99.08        | 0.999  | 0.990    | 0.990     | 0.991    |
| **LSTM**              | 94.02        | 0.992  | 0.939    | 0.939     | 0.939    |
| **ViT**               | 93.74        | 0.992  | 0.936    | 0.937     | 0.937    |
| **2D CNN**            | 92.44        | 0.987  | 0.922    | 0.923     | 0.923    |
| **3D CNN-LSTM**       | 40.96        | 0.500  | 0.194    | 0.137     | 0.333    |

### Observations

1. **VGG** achieved the highest performance across all metrics, indicating its strong ability to generalize and capture relevant features for classification.
2. **LSTM** performed second-best, leveraging the temporal relationships in CT scans effectively.
3. **ViT** demonstrated comparable performance to LSTM, highlighting the potential of attention mechanisms in image classification.
4. **2D CNN** achieved satisfactory results, but it did not outperform models leveraging sequence or attention mechanisms.
5. **3D CNN-LSTM** showed limited performance, suggesting potential issues in capturing spatiotemporal features or the need for further optimization.

## License

This project is licensed under the [LICENSE](https://opensource.org/licenses/MIT) - MIT License.
