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

## 🧑‍💻 Methods Overview

### VGG-based Classifier  
*Utilizes deep convolutional neural networks to accurately identify key pathological features from lung CT scans.*

Uses a VGG-16 model pretrained on ImageNet and fine-tuned for CT scan classification. Each axial CT slice is resized to 224×224 and passed through the convolutional backbone of the VGG model. Extracted features are flattened and passed through fully connected layers with batch normalization and dropout layers to enhance generalization and reduce overfitting. The final classification layer outputs logits for binary or multiclass prediction, trained using cross-entropy loss with early stopping based on validation performance.

---

### LSTM with VGG Features  
*Combines spatial feature extraction with sequential modeling to analyze CT scans, capturing the temporal progression across image slices.*

Each slice in a CT scan is first processed by a VGG-16 feature extractor. The resulting sequence of feature vectors (one per slice) is fed into an LSTM layer that captures inter-slice temporal dependencies. The LSTM's output is aggregated using the final hidden state or an average pooling strategy. The aggregated representation is passed through a fully connected classifier. Training uses a sliding window approach to handle varying scan lengths.

---

### Attention-LSTM with VGG Features  
*Enhances standard LSTM models by incorporating attention mechanisms, focusing explicitly on diagnostically relevant CT slices.*

Similar to the LSTM with VGG architecture, this model first extracts VGG-based features from each slice and processes them with an LSTM layer. An attention layer then learns to weigh each time step's output, highlighting more informative slices. The weighted sum is passed to a dense classifier. This helps the model focus on subtle pathology-related frames while reducing noise from irrelevant slices.

---

### Masked Autoencoder (MAE)  
*Leverages unsupervised representation learning by reconstructing obscured image regions, suitable for datasets with incomplete or noisy medical images.*

Uses a Vision Transformer-based MAE pretrained on large datasets, where 75% of input patches are randomly masked. The encoder (ViT) processes the visible patches, and the decoder reconstructs the full image. For classification, only the encoder is retained. Each CT slice is encoded, and the resulting features are aggregated across slices using average pooling or an LSTM. A linear classifier is used for final predictions.

---

### CLIP (Contrastive Language-Image Pretraining)  
*Uses multimodal embeddings combining visual features with semantic textual descriptions to generalize across clinical contexts.*

Applies a pretrained CLIP model that includes a visual encoder (ViT or ResNet) and a text encoder (Transformer). CT images are encoded, and textual class labels such as "Normal", "Common Pneumonia", and "COVID-19" are embedded using the text encoder. Cosine similarity is computed between image and text embeddings, and the most similar class is predicted. This zero-shot setting enables flexible generalization.

---

### Vision Transformer (ViT)  
*Processes CT scans through self-attention mechanisms, capturing long-range dependencies within image patches.*

CT slices are resized and divided into fixed-size patches (e.g., 16×16). Each patch is linearly embedded and enriched with positional encodings. The sequence of patch embeddings is fed into a series of transformer encoder layers with multi-head self-attention. A special classification token is prepended to the sequence, and its final hidden state is used for prediction. For 3D scans, features can be averaged or processed sequentially.

---

## 📁 Directory Structure

| Folder | Description |
|--------|-------------|
| `datasets/` | Dataset classes and statistics scripts |
| `environment/` | Configuration files (e.g., Docker) |
| `cloud_run/` | Python programs to start cloud experiments |
| `scripts/` | Training scripts organized by model architecture |
| `local_run/` | Cloud scripts to run training, test  and evaluation locally |
| `utils/` | Utility scripts (logging, downloading, checks) |
| `README.md` | Project documentation |
| `DATASET.md` | Dataset details and usage instructions |

---

## 📂 Datasets

Details about dataset preparation and usage are provided in [DATASET.md](DATASET.md).  
Ensure datasets are placed in the expected folder structure before executing training scripts.

---

## 📈 Results

### Binary Classification (0-NCP, 1-Normal)

| Method                                | Accuracy (%) | AUC   | F1 Score | Precision | Recall | Training Time |
|---------------------------------------|--------------|-------|----------|-----------|--------|----------------|
| **VGG-based Classifier**              | **98.87**    | 0.999 | 0.987    | 0.979     | 0.997  | ~25 min        |
| LSTM with VGG Features                | 97.80        | 0.997 | 0.977    | 0.977     | 0.978  | ~42 min        |
| Attention-LSTM with VGG Features      | 96.79        | 0.995 | 0.972    | 0.993     | 0.962  | ~48 min        |
| Masked Autoencoder (MAE)              | 96.75        | 0.994 | 0.962    | 0.960     | 0.989  | ~55 min        |
| CLIP                                  | 94.22        | 0.986 | 0.933    | 0.914     | 0.980  | ~30 min        |
| Vision Transformer (ViT)              | 94.19        | 0.984 | 0.933    | 0.952     | 0.954  | ~37 min        |

---

### Multiclass Classification (0-CP, 1-NCP, 2-Normal)

| Model                                      | Accuracy (%) | AUC   | F1 Score | Precision | Recall |
|-------------------------------------------|--------------|-------|----------|-----------|--------|
| **VGG**                                    | **99.08**    | 0.999 | 0.990    | 0.990     | 0.991  |
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
- VGG-based classifiers achieved state-of-the-art performance, highlighting the power of deep CNNs for spatial feature extraction.
- LSTM and Attention-LSTM models demonstrated the value of modeling temporal context across CT slices.
- MAE and CLIP models offered competitive results using unsupervised and multimodal learning respectively.
- ViT models performed well, though hybrid models incorporating sequential modeling were more effective.

**Multiclass Classification**
- VGG again led performance, followed by attention-enhanced sequential models.
- Temporal modeling (via LSTM) with attention layers showed strong generalization.
- Pure ViT and 2D CNN approaches served as solid baselines but lacked the benefits of sequence modeling.
- 3D CNN-LSTM struggled significantly, indicating challenges in complex spatiotemporal modeling for this dataset.

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

### **Accuracy**
![Accuracy](https://latex.codecogs.com/png.image?\dpi{120} \text{Accuracy}=\frac{\text{Number\ of\ Correct\ Predictions}}{\text{Total\ Number\ of\ Predictions}})

### **Precision (Macro-Averaged)**
![Precision Macro](https://latex.codecogs.com/png.image?\dpi{120} \text{Precision}_i=\frac{\text{TP}_i}{\text{TP}_i+\text{FP}_i}\quad,\quad\text{Precision}_{macro}=\frac{1}{N}\sum_{i=1}^{N}\text{Precision}_i)

### **Recall (Macro-Averaged)**
![Recall Macro](https://latex.codecogs.com/png.image?\dpi{120} \text{Recall}_i=\frac{\text{TP}_i}{\text{TP}_i+\text{FN}_i}\quad,\quad\text{Recall}_{macro}=\frac{1}{N}\sum_{i=1}^{N}\text{Recall}_i)

### **F1 Score (Macro-Averaged)**
![F1 Macro](https://latex.codecogs.com/png.image?\dpi{120} \text{F1}_i=\frac{2\cdot\text{Precision}_i\cdot\text{Recall}_i}{\text{Precision}_i+\text{Recall}_i}\quad,\quad\text{F1}_{macro}=\frac{1}{N}\sum_{i=1}^{N}\text{F1}_i)

### **AUC (One-vs-Rest)**
![AUC Macro](https://latex.codecogs.com/png.image?\dpi{120} \text{AUC}_{macro}=\frac{1}{N}\sum_{i=1}^{N}\text{AUC}_i)

### **Confusion Matrix**
> Each element (i, j) indicates how many instances of class i were predicted as class j.

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