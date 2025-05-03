# Dataset: China Consortium of Chest CT Image Investigation (CC-CCII)

## Description

The China Consortium of Chest CT Image Investigation (CC-CCII) dataset consists of chest CT images collected from patients diagnosed with:

1. **Novel Coronavirus Pneumonia (NCP)** caused by SARS-CoV-2.
2. **Common Pneumonia (CP)**.
3. **Normal controls** without signs of pneumonia.

The dataset contains thousands of CT studies and is a valuable resource for developing and validating artificial intelligence models in the field of medical imaging, particularly for pneumonia classification and COVID-19 diagnosis.

**Download page:** [CC-CCII Dataset](http://ncov-ai.big.ac.cn/download)  
**Reference:**  
Kang Zhang, Xiaohong Liu, Jun Shen, et al. (2020). *Clinically Applicable AI System for Accurate Diagnosis, Quantitative Measurements and Prognosis of COVID-19 Pneumonia Using Computed Tomography*. Cell. DOI: [10.1016/j.cell.2020.04.045](https://www.cell.com/cell/fulltext/S0092-8674\(20\)30551-1?rss=yes)

---

## Dataset Statistics (Original `ccccii`)

| Class   | Patients | % Patients | Volumes | % Volumes | Slices | % Slices |
|:--------|---------:|:-----------|--------:|:----------|-------:|:---------|
| CP      | 964      | 35.57%     | 1347    | 35.06%    | 154802 | 38.82%   |
| NCP     | 897      | 33.10%     | 1449    | 37.71%    | 149036 | 37.37%   |
| Normal  | 849      | 31.33%     | 1046    | 27.23%    | 94939  | 23.81%   |
| **Total** | 2710   | 100.00%    | 3842    | 100.00%   | 398777 | 100.00%  |

---

## Data Selection

To ensure consistency across all training inputs, a two-step automatic filtering process was applied:

1. **Minimum slice count (≥ 30):**  
   Only CT studies with at least 30 slices were included to ensure sufficient anatomical continuity.

2. **Sequence continuity (standard deviation threshold = 50):**  
   The slice sequence of each scan was evaluated based on the standard deviation of per-pixel absolute differences between consecutive slices. Studies with high variability, indicating disjoint exams, were excluded.

This selection ensures that all remaining studies are long and anatomically consistent — ideal for sequence-based models learning spatial and temporal patterns.

### Dataset Statistics (`ccccii_selected`)

| Class   | Patients | % Patients | Volumes | % Volumes | Slices | % Slices |
|:--------|---------:|:-----------|--------:|:----------|-------:|:---------|
| CP      | 580      | 27.27%     | 1124    | 32.63%    | 119310 | 36.18%   |
| NCP     | 796      | 37.42%     | 1348    | 39.13%    | 129040 | 39.13%   |
| Normal  | 751      | 35.31%     | 973     | 28.24%    | 81409  | 24.69%   |
| **Total** | 2127   | 100.00%    | 3445    | 100.00%   | 329759 | 100.00%  |

---

## Segmentation Filtering

Upon review, some studies showed evidence of **lung segmentation** (black backgrounds around segmented lung areas), while others retained the **full-body CT slices**.

Since mixing these two input distributions could harm model performance, we applied a mean pixel intensity threshold:

- **Segmented scans:** mean pixel intensity < 50  
- **Non-segmented scans:** mean pixel intensity ≥ 50

The mean pixel intensity was calculated as:

```
Mean Intensity = (1 / N) * sum(mean(slice_i) for i in 1..N)
```
Where:
- `N` is the number of slices in the study,
- `mean(slice_i)` is the mean pixel value of slice `i` after grayscale conversion.

This effectively separated segmented and non-segmented scans.

---

## Final Dataset: `ccccii_selected_nonsegmented`

| Class   | Patients | % Patients | Volumes | % Volumes | Slices | % Slices |
|:--------|---------:|:-----------|--------:|:----------|-------:|:---------|
| CP      | 577      | 40.21%     | 1119    | 41.00%    | 118945 | 45.09%   |
| NCP     | 707      | 49.27%     | 1237    | 45.33%    | 113279 | 42.94%   |
| Normal  | 151      | 10.52%     | 373     | 13.67%    | 31572  | 11.97%   |
| **Total** | 1435   | 100.00%    | 2729    | 100.00%   | 263796 | 100.00%  |

---

## Cross-Validation Setup (5-Fold Stratified by Patient)

To evaluate model performance reliably, we applied **5-fold cross-validation**, stratified by patient to preserve class distribution and avoid patient overlap between folds.

Each fold consists of:

- ~80% of patients for **training**,
- ~20% of patients for **validation**.

The test set remains separate and fixed.

---

### Fold 1 Example

**Training Set:**

| Class   | Patients | % Patients | Volumes | % Volumes | Slices | % Slices |
|:--------|---------:|:-----------|--------:|:----------|-------:|:---------|
| CP      | 415      | 40.25%     | 816     | 41.63%    | 88834  | 46.43%   |
| NCP     | 508      | 49.27%     | 886     | 45.20%    | 80733  | 42.20%   |
| Normal  | 108      | 10.48%     | 258     | 13.16%    | 21746  | 11.37%   |

**Validation Set:**

| Class   | Patients | % Patients | Volumes | % Volumes | Slices | % Slices |
|:--------|---------:|:-----------|--------:|:----------|-------:|:---------|
| CP      | 104      | 40.15%     | 189     | 39.13%    | 18034  | 40.28%   |
| NCP     | 128      | 49.42%     | 222     | 45.96%    | 20502  | 45.79%   |
| Normal  | 27       | 10.42%     | 72      | 14.91%    | 6235   | 13.93%   |

---

### Test Set (Fixed)

| Class   | Patients | % Patients | Volumes | % Volumes | Slices | % Slices |
|:--------|---------:|:-----------|--------:|:----------|-------:|:---------|
| CP      | 58       | 40.00%     | 114     | 39.86%    | 12077  | 43.58%   |
| NCP     | 71       | 48.97%     | 129     | 45.10%    | 12044  | 43.46%   |
| Normal  | 16       | 11.03%     | 43      | 15.03%    | 3591   | 12.96%   |

---

## Summary of Cross-Validation Strategy

- **5 stratified folds**, no patient overlap across train/validation splits.
- **Fixed test set** reserved for final evaluation.
- Balanced representation across all classes.

This setup allows robust and fair performance evaluation of models across different folds while keeping a fully unseen test set for final benchmarking.
