# 🫁 PulmoSense: Intelligent Lung Pathology Detection

This repository contains the code, resources, and documentation for the **PulmoSense** research project, which focuses on detecting and classifying **COVID-19** and **common pneumonia** in CT scans using advanced deep learning techniques.

---

## 📌 Overview

PulmoSense aims to leverage deep learning to analyze and classify pulmonary pathologies in CT scans, with a specific focus on:
- Common Pneumonia (CP)
- Novel Coronavirus Pneumonia (NCP)
- Normal cases

This repository includes:
- End-to-end training pipelines for various models  
- Scripts for dataset handling, model training, and evaluation  
- Predefined experiments and results documentation  

---

## 🚀 Getting Started

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

## 📁 Directory Structure

| Folder | Description |
|--------|-------------|
| `datasets/` | Dataset classes and statistics scripts |
| `environment/` | Configuration files (e.g., Docker) |
| `notebooks/` | Jupyter notebooks for training and experiments |
| `scripts/` | Training scripts organized by model architecture |
| `shell/` | Shell scripts for dataset download and model training |
| `utils/` | Utility scripts (logging, downloading, checks) |
| `README.md` | Project documentation |
| `DATASET.md` | Dataset details and usage instructions |

---

## ⚙️ Usage

### Training Models

Use the shell scripts in the `shell/` directory to train models:

```bash
bash shell/train_vgg_multiclass.sh
```

> **Note:** You can also train models using **Azure Machine Learning** via the notebooks in `/notebooks`. Examples include training CNN, LSTM, VGG, and ViT models using Azure compute resources.

### Adding Custom Models or Datasets

1. Add a new dataset handler in `datasets/`.
2. Create or modify a training script in `scripts/train/`.
3. Update shell scripts in `shell/` if needed.

---

## 📂 Datasets

Details about dataset preparation and usage are provided in [DATASET.md](DATASET.md).  
Ensure datasets are placed in the expected folder structure before executing training scripts.

---

## 📈 Results

### Binary Classification (0-NCP, 1-Normal)

| Model                                      | Accuracy (%) | AUC   | F1 Score | Precision | Recall |
|-------------------------------------------|--------------|-------|----------|-----------|--------|
| >> LSTM with VGG features                 |              |       |          |           |        |
| >> VGG                                       |             |        |          |           |        |
| >> Attention-based LSTM with VGG features |              |       |          |           |        |
| ViT                                       | 89.82        | 0.981 | 0.869    | 0.952     | 0.800  |



### Multiclass Classification (0-CP, 1-NCP, 2-Normal)

| Model                                      | Accuracy (%) | AUC   | F1 Score | Precision | Recall |
|-------------------------------------------|--------------|-------|----------|-----------|--------|
| >> VGG                                        | 99.08        | 0.999 | 0.990    | 0.990     | 0.991  |
| Attention-based LSTM with VGG features     | 95.19        | 0.993 | 0.951    | 0.949     | 0.954  |
| Attention-based LSTM with 2D CNN features  | 94.67        | 0.994 | 0.946    | 0.946     | 0.946  |
| LSTM with VGG features                     | 94.15        | 0.993 | 0.941    | 0.939     | 0.945  |
| LSTM with 2D CNN features                  | 94.02        | 0.992 | 0.939    | 0.939     | 0.939  |
| ViT                                        | 93.74        | 0.992 | 0.936    | 0.937     | 0.937  |
| 2D CNN                                     | 92.44        | 0.987 | 0.922    | 0.923     | 0.923  |
| 3D CNN-LSTM                                | 40.96        | 0.500 | 0.194    | 0.137     | 0.333  |

---

## 📝 Notes

**Binary Classification**
- LSTM with VGG features achieved the highest accuracy, effectively modeling temporal features extracted by VGG.
- VGG demonstrated excellent standalone feature extraction, providing near-best results.
- Attention-based LSTM with VGG features closely followed, suggesting attention mechanisms enhance temporal feature modeling.
- ViT demonstrated reasonable performance, yet fell short compared to models explicitly modeling sequential information.

**Multiclass Classification**
- VGG emerged as the top-performing model, showcasing superior feature extraction and robust generalization.
- Attention-based LSTM with VGG features secured the second-best performance, effectively combining attention mechanisms with powerful VGG-derived features.
- Attention-based LSTM with 2D CNN features performed notably, underscoring the significance of attention layers in enhancing sequential modeling capabilities.
- LSTM with either VGG or CNN features was effective in capturing temporal relationships among slices, proving their utility in multiclass scenarios.
- ViT displayed potential leveraging self-attention but did not surpass hybrid architectures integrating sequential and spatial data.
- 2D CNN served as a strong baseline, though it lacked the additional performance gains offered by temporal or attention-driven methods.
- 3D CNN-LSTM notably struggled, suggesting difficulties in adequately capturing complex spatiotemporal dependencies in the data.

---

## 📊 Evaluation Metrics Explained

PulmoSense uses several standard metrics to evaluate model performance:

| Metric       | Description |
|--------------|-------------|
| **Accuracy** | Proportion of correct predictions across all samples. May be misleading with imbalanced classes. |
| **Precision (Macro)** | Measures the proportion of relevant predictions. Macro averaging treats all classes equally. |
| **Recall (Macro)** | Indicates the proportion of actual positive instances correctly identified. Macro averaging ensures balanced class-level performance. |
| **F1 Score (Macro)** | Harmonic mean of precision and recall. Reflects overall balance between both metrics across all classes. |
| **AUC (OvR)** | Measures separability of classes via One-vs-Rest strategy, providing a discriminative performance indicator per class. |
| **Confusion Matrix** | Displays correct and incorrect predictions per class, aiding in error analysis and model behavior understanding. |

### Why Macro Averaging?

In multiclass classification — especially with imbalanced datasets — macro averaging ensures fair representation of all classes when computing metrics.

- **Macro averaging**: Equal weight for each class, highlighting model performance across all categories.
- **Micro averaging**: Favors dominant classes, less useful for rare disease detection.
- **Weighted averaging**: Accounts for class frequency but can mask poor performance on minority classes.

Given the clinical relevance of underrepresented conditions, macro metrics provide a fairer and more informative evaluation framework.

---

## 📐 Metric Calculation Details

This section describes how each evaluation metric used in PulmoSense is computed, with a focus on the multiclass setting using **macro averaging**.

### **Accuracy**
> **Definition**: The ratio of correct predictions to total predictions.

**Formula**:
\[
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
\]

---

### **Precision (Macro-Averaged)**
> **Definition**: The proportion of true positive predictions among all positive predictions, averaged across all classes.

**Per-class Precision**:
\[
\text{Precision}_i = \frac{\text{True Positives}_i}{\text{True Positives}_i + \text{False Positives}_i}
\]

**Macro Precision**:
\[
\text{Precision}_{macro} = \frac{1}{N} \sum_{i=1}^{N} \text{Precision}_i
\]
Where **N** is the number of classes.

---

### **Recall (Macro-Averaged)**
> **Definition**: The proportion of true positive predictions among all actual positive instances, averaged across all classes.

**Per-class Recall**:
\[
\text{Recall}_i = \frac{\text{True Positives}_i}{\text{True Positives}_i + \text{False Negatives}_i}
\]

**Macro Recall**:
\[
\text{Recall}_{macro} = \frac{1}{N} \sum_{i=1}^{N} \text{Recall}_i
\]

---

### **F1 Score (Macro-Averaged)**
> **Definition**: The harmonic mean of precision and recall, averaged across all classes.

**Per-class F1 Score**:
\[
\text{F1}_i = \frac{2 \cdot \text{Precision}_i \cdot \text{Recall}_i}{\text{Precision}_i + \text{Recall}_i}
\]

**Macro F1 Score**:
\[
\text{F1}_{macro} = \frac{1}{N} \sum_{i=1}^{N} \text{F1}_i
\]

---

### **AUC (Area Under the ROC Curve, One-vs-Rest)**
> **Definition**: Measures the model's ability to distinguish each class from the others. In multiclass classification, **One-vs-Rest (OvR)** strategy is used.

For each class:
- Treat it as the **positive class**, and all others as **negative**.
- Compute the ROC AUC curve.
- Average the AUCs across all classes.

**AUC (OvR Average)**:
\[
\text{AUC}_{macro} = \frac{1}{N} \sum_{i=1}^{N} \text{AUC}_i
\]

---

### **Confusion Matrix**
> **Definition**: A square matrix where rows represent the actual classes and columns represent the predicted classes.

Each element **(i, j)** in the matrix represents the number of instances from class **i** that were predicted as class **j**.

It helps identify:
- **True Positives** (diagonal cells)
- **False Positives / False Negatives** (off-diagonal cells)
- Patterns of misclassification between classes

---

## 📚 References & Publications

This project is part of an ongoing research effort in the application of deep learning and computer vision to medical imaging. The following publications are related to the PulmoSense project and its underlying techniques:

### Related Publications

- **Lacerda, P., Barros, B., Albuquerque, C., Conci, A.**  
  *Hyperparameter optimization for COVID-19 pneumonia diagnosis based on chest CT*.  
  *Sensors*, **21**(6), 2174, 2021.  
  [🔗 View Paper](https://www.mdpi.com/1424-8220/21/6/2174)

- **Barros, B., Lacerda, P., Albuquerque, C., Conci, A.**  
  *Pulmonary COVID-19: learning spatiotemporal features combining CNN and LSTM networks for lung ultrasound video classification*.  
  *Sensors*, **21**(16), 5486, 2021.  
  [🔗 View Paper](https://www.mdpi.com/1424-8220/21/16/5486)

- **Lacerda, P., Gonzalez, J., Rocha, N., Seixas, F., Albuquerque, C., Clua, E., Conci, A.**  
  *A parallel method for anatomical structure segmentation based on 3D seeded region growing*.  
  *2020 International Joint Conference on Neural Networks (IJCNN)*, IEEE, 2020.  
  [🔗 View Paper](https://ieeexplore.ieee.org/abstract/document/9206630)

### BibTeX Citation

```bibtex
@article{lacerda2021hyperparameter,
  title={Hyperparameter optimization for COVID-19 pneumonia diagnosis based on chest CT},
  author={Lacerda, Paulo and Barros, Bruno and Albuquerque, C{\'e}lio and Conci, Aura},
  journal={Sensors},
  volume={21},
  number={6},
  pages={2174},
  year={2021},
  publisher={MDPI}
}

@article{barros2021pulmonary,
  title={Pulmonary COVID-19: learning spatiotemporal features combining CNN and LSTM networks for lung ultrasound video classification},
  author={Barros, Bruno and Lacerda, Paulo and Albuquerque, Celio and Conci, Aura},
  journal={Sensors},
  volume={21},
  number={16},
  pages={5486},
  year={2021},
  publisher={Multidisciplinary Digital Publishing Institute}
}

@inproceedings{lacerda2020parallel,
  title={A parallel method for anatomical structure segmentation based on 3d seeded region growing},
  author={Lacerda, Paulo and Gonzalez, Jos{\'e} and Rocha, Nazareth and Seixas, Flavio and Albuquerque, C{\'e}lio and Clua, Esteban and Conci, Aura},
  booktitle={2020 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--6},
  year={2020},
  organization={IEEE}
}
```

> Please cite these works if you use or extend this repository in academic or scientific research.
