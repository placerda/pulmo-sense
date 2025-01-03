# Datasets

This document provides detailed information about the datasets used in the "PulmoSense" project.

---

## Dataset: China Consortium of Chest CT Image Investigation (CC-CCII)

### Description

The dataset is based on the China Consortium of Chest CT Image Investigation (CC-CCII). It includes CT images and metadata constructed from cohorts associated with the consortium. All CT images are classified into three categories:

1. **Novel Coronavirus Pneumonia (NCP)** due to SARS-CoV-2 virus infection.
2. **Common Pneumonia (CP)**.
3. **Normal controls**.

This dataset aims to assist clinicians and researchers worldwide in combating the COVID-19 pandemic.

For this project, I collected all patients across all classes but included only the studies with more than 30 slices. Additionally, a patient-level StratifiedKFold was used for generating slice statistics, ensuring no data leakage during the data split process.

**Download page:** [CC-CCII Dataset](http://ncov-ai.big.ac.cn/download)

**Reference:**
Kang Zhang, Xiaohong Liu, Jun Shen, et al. Jianxing He, Tianxin Lin, Weimin Li, Guangyu Wang. (2020). Clinically Applicable AI System for Accurate Diagnosis, Quantitative Measurements and Prognosis of COVID-19 Pneumonia Using Computed Tomography. *Cell*, DOI: [10.1016/j.cell.2020.04.045](https://www.cell.com/cell/fulltext/S0092-8674\(20\)30551-1?rss=yes)

---

### Dataset Statistics

#### Slice-Based Overall Statistics

| CP    | CP %   | NCP   | NCP %  | Normal | Normal % | Total  | Total % |
| ----- | ------ | ----- | ------ | ------ | -------- | ------ | ------- |
| 40410 | 35.06% | 43470 | 37.71% | 31380  | 27.23%   | 115260 | 100.00% |

#### Slice-Based Train/Val Statistics

| Split | CP    | CP %   | NCP   | NCP %  | Normal | Normal % | Total | Total % |
| ----- | ----- | ------ | ----- | ------ | ------ | -------- | ----- | ------- |
| Train | 32730 | 28.40% | 34860 | 30.24% | 24870  | 21.58%   | 92460 | 80.22%  |
| Val   | 7680  | 6.66%  | 8610  | 7.47%  | 6510   | 5.65%    | 22800 | 19.78%  |

#### Patient Counts

| Class  | Total Patients | % Patients | Train Patients | Train % | Val Patients | Val %  |
| ------ | -------------- | ---------- | -------------- | ------- | ------------ | ------ |
| CP     | 964            | 35.57%     | 771            | 28.45%  | 193          | 7.12%  |
| NCP    | 897            | 33.10%     | 717            | 26.46%  | 180          | 6.64%  |
| Normal | 849            | 31.33%     | 680            | 25.09%  | 169          | 6.24%  |
| Total  | 2710           | 100.00%    | 2168           | 80.00%  | 542          | 20.00% |

#### Volume-Based Overall Statistics

| CP   | CP %   | NCP  | NCP %  | Normal | Normal % | Total | Total % |
| ---- | ------ | ---- | ------ | ------ | -------- | ----- | ------- |
| 1347 | 35.06% | 1449 | 37.71% | 1046   | 27.23%   | 3842  | 100.00% |

#### Volume-Based Train/Val Statistics

| Class  | Train Volumes | Train % of Total | Val Volumes | Val % of Total |
| ------ | ------------- | ---------------- | ----------- | -------------- |
| CP     | 1077          | 28.03%           | 270         | 7.03%          |
| NCP    | 1159          | 30.17%           | 290         | 7.55%          |
| Normal | 837           | 21.79%           | 209         | 5.44%          |
| Total  | 3073          | 79.98%           | 769         | 20.02%         |

---

### Download Instructions

1. Navigate to the official dataset page: [http://ncov-ai.big.ac.cn/download](http://ncov-ai.big.ac.cn/download).
2. Follow the instructions to download the dataset to your local system.

---

### Data Preparation

1. Extract all dataset zip files into a `data` folder.
