# Datasets

This document provides information about the datasets used in the "PulmoSense" project.

## MosMed-1110

### Description

The MosMed-1110 dataset is a publicly available collection of 1110 chest CT scans from patients diagnosed with COVID-19, classifying COVID-19 stages via lung severity, as observed in CT scans. These stages are:

- CT-0: No signs of pneumonia.
- CT-1: Mild stage with ground-glass opacities and less than 25% lung involvement.
- CT-2: Moderate stage with 25% to 50% lung involvement.
- CT-3: Severe stage with 50% to 75% lung involvement, often showing rapid progression.
- CT-4: Critical stage: Over 75% lung involvement with severe symptoms.

The dataset's scans were obtained from municipal hospitals in Moscow, Russia, during the COVID-19 epidemic, from March 1, 2020, to April 25, 2020. This dataset aids in developing AI algorithms and provides a valuable resource for understanding how COVID-19 progresses and affects the lungs.

### Download Instructions

To download the MosMed-1110, you should access and follow the instructions available at this link: [MosMed-1110 Dataset](https://github.com/neuro-ml/COVID-19-Triage?tab=readme-ov-file#mosmed-1110)

### Data Preparation

This repository utilizes numpy structures as data inputs for the deep learning model. The MosMed-1110 dataset, provided in NIfTI format, needs to be converted into numpy for compatibility. Execute the commands below to perform this conversion.

> Using Conda for Python environment management in this project. Install Conda from [Anaconda](https://www.anaconda.com/download/success).

```
conda create -n mosmed python=3.6
conda activate mosmed
pip install -r mosmed_requirements.txt
python preprocessing/preprocessing_mosmed.py -i <mosmed_nifti> -o <mosmed_numpy>
```

In the commands above, `<mosmed_nifti>` and `<mosmed_numpy>` are placeholders for specific paths on your file system. Replace `<mosmed_nifti>` with the path to the 'studies' folder in the downloaded MosMed dataset. Replace `<mosmed_numpy>` with the desired destination path for your prepared data.

For example:

`python preprocessing/preprocessing_mosmed.py -i data/raw/mosmed -o data/processed/mosmed`

Done! Now you can use the dataset as input for PulmoSense.

## Dataset 2

### Description

(Provide a detailed description of the second dataset here.)

### Download Instructions

(Provide instructions on how to download the second dataset here.)

### Data Preparation

(Provide instructions on how to prepare the second dataset for use in the project here.)

(Continue with additional datasets as necessary.)