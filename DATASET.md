## Dataset: China Consortium of Chest CT Image Investigation (CC-CCII)

### Description

The dataset is based on the China Consortium of Chest CT Image Investigation (CC-CCII). It includes CT images and metadata constructed from cohorts associated with the consortium. All CT images are classified into three categories:

1. **Novel Coronavirus Pneumonia (NCP)** due to SARS-CoV-2 virus infection.
2. **Common Pneumonia (CP)**.
3. **Normal controls**.

This dataset aims to assist clinicians and researchers worldwide in combating the COVID-19 pandemic.

**Download page:** [CC-CCII Dataset](http://ncov-ai.big.ac.cn/download)

**Reference:**  
Kang Zhang, Xiaohong Liu, Jun Shen, et al. Jianxing He, Tianxin Lin, Weimin Li, Guangyu Wang. (2020). Clinically Applicable AI System for Accurate Diagnosis, Quantitative Measurements and Prognosis of COVID-19 Pneumonia Using Computed Tomography. *Cell*, DOI: [10.1016/j.cell.2020.04.045](https://www.cell.com/cell/fulltext/S0092-8674\(20\)30551-1?rss=yes)

---

### Dataset Statistics

| Class   | Patients | % Patients | Volumes | % Volumes | Slices | % Slices |
|:--------|---------:|:-----------|--------:|:----------|-------:|:---------|
| CP      | 964      | 35.57%     | 1347    | 35.06%    | 40410  | 35.06%   |
| NCP     | 897      | 33.10%     | 1449    | 37.71%    | 43470  | 37.71%   |
| Normal  | 849      | 31.33%     | 1046    | 27.23%    | 31380  | 27.23%   |
| **Total** | 2710   | 100.00%    | 3842    | 100.00%   | 115260 | 100.00%  |

---

### Download Instructions

1. Navigate to the official dataset page: [http://ncov-ai.big.ac.cn/download](http://ncov-ai.big.ac.cn/download).
2. Follow the instructions to download the dataset to your local system.

---

### Data Preparation

1. Extract all dataset zip files into a `data` folder.

---

## Dataset: Moscow COVID-19 CT Dataset (MosMed)

### Description

The dataset is based on the Moscow COVID-19 CT Dataset (MosMed), which provides chest CT scans collected from patients with and without signs of COVID-19 infection. In this project, the dataset was normalized and only studies with more than 30 slices were retained for consistency.

Notably, this version of the dataset contains only two categories:

1. **Novel Coronavirus Pneumonia (NCP)** — cases showing signs of COVID-19-related pneumonia.
2. **Normal controls** — CT scans without signs of pneumonia.

**Download page:** [MosMedData](https://mosmed.ai/datasets/covid19_1110/)

**Reference:**  
Morozov, S., Andreychenko, A., Pavlov, N., Vladzymyrskyy, A., Ledikhova, N., Gombolevskiy, V., Blokhin, I., Gelezhe, P., Gonchar, A., Andrianova, I. (2020). MosMedData: Chest CT Scans with COVID-19 Related Findings. *arXiv preprint*, [arXiv:2005.06465](https://arxiv.org/abs/2005.06465)

---

### Dataset Statistics

| Class   | Patients | % Patients | Volumes | % Volumes | Slices | % Slices |
|:--------|---------:|:-----------|--------:|:----------|-------:|:---------|
| CP      | 0        | 0.00%      | 0       | 0.00%     | 0      | 0.00%    |
| NCP     | 856      | 77.12%     | 855     | 77.10%    | 25650  | 77.10%   |
| Normal  | 254      | 22.88%     | 254     | 22.90%    | 7620   | 22.90%   |
| **Total** | 1110   | 100.00%    | 1109    | 100.00%   | 33270  | 100.00%  |

---

### Download Instructions

1. Visit the official dataset page: [https://mosmed.ai/datasets/covid19_1110/](https://mosmed.ai/datasets/covid19_1110/)
2. Download the dataset archive and extract its contents into the `data` folder.

---

### Data Preparation

1. Normalize and filter the CT scans to include only those with at least 30 slices.
2. Organize them under `data/mosmed_normalized` with subfolders `NCP` and `Normal`.

---

### Additional Consideration: MosMed Normalization for Domain Adaptation

To improve contrast consistency and reduce domain shift, the MosMed dataset was normalized using the intensity distribution (mean and standard deviation) computed from the CC-CCII dataset. This normalization aligns the brightness and contrast levels of MosMed images with those of CC-CCII, minimizing discrepancies caused by differences in image acquisition protocols or equipment. This approach, known as **domain adaptation via distribution matching**, enhances the model's ability to generalize by reducing reliance on dataset-specific artifacts and emphasizing anatomical and pathological features.
