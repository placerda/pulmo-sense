# PulmoSense: Intelligent Lung Pathology Detection

This repository hosts the code, resources, and documentation for the doctoral research project "PulmoSense". The project focuses on the detection of COVID-19, Pulmonary Fibrosis, Lung Cancer, Community-acquired pneumonia (CAP), and bacterial pneumonia in CT scans using advanced Deep Learning techniques.

## Overview

"PulmoSense" is a project that aims to leverage the power of Deep Learning for the detection and classification of pulmonary pathologies in CT scans. The focus is on identifying cases of COVID-19, Pulmonary Fibrosis, Lung Cancer, Community-acquired pneumonia (CAP), and bacterial pneumonia. This repository provides comprehensive instructions to reproduce the experiments and access the datasets used in the study.

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


## Usage

To reproduce the experiments and execute the classification model, follow the steps detailed in the "Usage" section.

## Datasets

The datasets used in this research are detailed in [this](DATASET.md) page.

## Results

The "Results" section provides a summary of the outcomes from the experiments conducted during this study, particularly those related to the detection of COVID-19 and bacterial pneumonia.

## Running Tests

Run all unit tests from the project's root directory with Python's `unittest` module using the command: `python -m unittest discover -s tests -v`.

## Contributing

Contributions to this project are welcome. Please refer to the guidelines in the [CONTRIBUTING.md] file.

## License

This project is licensed under the [LICENSE] - [License Name].