## Dataset: China Consortium of Chest CT Image Investigation (CC-CCII)

### Description

The China Consortium of Chest CT Image Investigation (CC-CCII) dataset consists of chest CT images collected from patients diagnosed with:

1. **Novel Coronavirus Pneumonia (NCP)** due to SARS-CoV-2.
2. **Common Pneumonia (CP)**.
3. **Normal controls** with no signs of pneumonia.

The dataset includes thousands of CT studies and serves as a valuable resource for developing and validating artificial intelligence models in the field of medical imaging, particularly in the context of pneumonia classification and COVID-19 diagnosis.

**Download page:** [CC-CCII Dataset](http://ncov-ai.big.ac.cn/download)  
**Reference:**  
Kang Zhang, Xiaohong Liu, Jun Shen, et al. (2020). *Clinically Applicable AI System for Accurate Diagnosis, Quantitative Measurements and Prognosis of COVID-19 Pneumonia Using Computed Tomography*. Cell. DOI: [10.1016/j.cell.2020.04.045](https://www.cell.com/cell/fulltext/S0092-8674\(20\)30551-1?rss=yes)

---

### Dataset Statistics (ccccii)

| Class   |   Total Patients | % Patients   |   Total Volumes | % Volume   |   Total Slices | % Slices   |
|:--------|-----------------:|:-------------|----------------:|:-----------|---------------:|:-----------|
| CP      |              964 | 35.57%       |            1347 | 35.06%     |         154802 | 38.82%     |
| NCP     |              897 | 33.10%       |            1449 | 37.71%     |         149036 | 37.37%     |
| Normal  |              849 | 31.33%       |            1046 | 27.23%     |          94939 | 23.81%     |
| **Total** |           2710 | 100.00%      |            3842 | 100.00%    |         398777 | 100.00%    |

---

### Data Selection

To ensure consistency across all training inputs, we applied a two-step automatic filtering process to the full dataset:

1. **Minimum slice count (≥ 30):**  
   Only CT studies containing at least 30 slices were included. Shorter scans were discarded due to insufficient anatomical continuity.

2. **Sequence continuity (std‑dev threshold = 50):**  
   Each scan’s slice sequence was evaluated based on the standard deviation of per-pixel absolute differences between consecutive slices. Studies with high variability were assumed to contain multiple, disjoint exams and were excluded.

This selection process ensures that all remaining studies are long and anatomically consistent—ideal for sequence-based models that learn from spatial and temporal patterns.

---

### Dataset Statistics (ccccii_selected)

| Class   |   Total Patients | % Patients   |   Total Volumes | % Volume   |   Total Slices | % Slices   |
|:--------|-----------------:|:-------------|----------------:|:-----------|---------------:|:-----------|
| CP      |              964 | 35.57%       |            1347 | 35.06%     |         154802 | 38.82%     |
| NCP     |              897 | 33.10%       |            1449 | 37.71%     |         149036 | 37.37%     |
| Normal  |              849 | 31.33%       |            1046 | 27.23%     |          94939 | 23.81%     |
| **Total** |           2710 | 100.00%      |            3842 | 100.00%    |         398777 | 100.00%    |

---

### Segmentation Filtering

Upon reviewing the selected studies, we observed that some of them had undergone **lung segmentation preprocessing**, while others retained the **original full-body CT slices**. Because these two formats represent very different input distributions, mixing them could negatively affect model performance.

To resolve this, we analyzed the average pixel intensity of each study. Segmented scans typically exhibit significantly **lower mean pixel values** due to the black background surrounding segmented lung regions. We classified studies accordingly using a threshold-based filter:

- **Segmented scans**: mean pixel intensity < 50  
- **Non-segmented scans**: mean pixel intensity ≥ 50  

The **mean pixel intensity** for each study was calculated by taking the grayscale version of all slices in a study and computing the average of all pixel values across all slices:

```
Mean Intensity = (1 / N) * sum(mean(slice_i) for i in 1..N)
```

Where:
- `N` is the number of slices in the study,
- `mean(slice_i)` is the average pixel value of slice `i` after conversion to grayscale.

This method effectively identifies whether a scan has large black regions typical of segmentation masks.

--- 

### Final Dataset Statistics (ccccii_selected_nonsegmented)

| Class   |   Total Patients | % Patients   |   Total Volumes | % Volume   |   Total Slices | % Slices   |
|:--------|-----------------:|:-------------|----------------:|:-----------|---------------:|:-----------|
| CP      |              577 | 40.21%       |            1119 | 41.00%     |         118945 | 45.09%     |
| NCP     |              707 | 49.27%       |            1237 | 45.33%     |         113279 | 42.94%     |
| Normal  |              151 | 10.52%       |             373 | 13.67%     |          31572 | 11.97%     |
| **Total** |           1435 | 100.00%      |            2729 | 100.00%    |         263796 | 100.00%    |

---

### Train-Test Split (90/10 Stratified by Patient)

To evaluate model generalization, the filtered dataset was split into train and test sets using a **patient-level stratified split**, preserving the class distribution and ensuring no patient appears in both sets.

#### Dataset Statistics 

| Class   |   Total Patients | % Patients   |   Total Volumes | % Volume   |   Total Slices | % Slices   |
|:--------|-----------------:|:-------------|----------------:|:-----------|---------------:|:-----------|
| CP      |              519 | 40.23%       |            1005 | 41.14%     |         106868 | 45.27%     |
| NCP     |              636 | 49.30%       |            1108 | 45.35%     |         101235 | 42.88%     |
| Normal  |              135 | 10.47%       |             330 | 13.51%     |          27981 | 11.85%     |
| **Total** |           1290 | 100.00%      |            2443 | 100.00%    |         236084 | 100.00%    |

#### Dataset Statistics 

| Class   |   Total Patients | % Patients   |   Total Volumes | % Volume   |   Total Slices | % Slices   |
|:--------|-----------------:|:-------------|----------------:|:-----------|---------------:|:-----------|
| CP      |               58 | 40.00%       |             114 | 39.86%     |          12077 | 43.58%     |
| NCP     |               71 | 48.97%       |             129 | 45.10%     |          12044 | 43.46%     |
| Normal  |               16 | 11.03%       |              43 | 15.03%     |           3591 | 12.96%     |
| **Total** |            145 | 100.00%      |             286 | 100.00%    |          27712 | 100.00%    |

---

### Final Split: Training and Validation

The final non-segmented dataset was split into three parts:

- **90%** of the dataset was reserved for training and validation,
- **10%** was kept as a separate test set (as described above).

From the 90% reserved for training and validation, we further applied a **patient-level stratified split** to divide the data into:

- **80% for training**, and  
- **20% for validation**.

This approach ensures a consistent and balanced representation of all classes across both subsets, without any patient overlap between training, validation, or test sets.

#### Train set

| Class   |   Total Patients | % Patients   |   Total Volumes | % Volume   |   Total Slices | % Slices   |
|:--------|-----------------:|:-------------|----------------:|:-----------|---------------:|:-----------|
| CP      |              415 | 40.21%       |             798 | 40.49%     |          83050 | 44.15%     |
| NCP     |              509 | 49.32%       |             893 | 45.31%     |          81709 | 43.43%     |
| Normal  |              108 | 10.47%       |             280 | 14.21%     |          23364 | 12.42%     |
| **Total** |           1032 | 100.00%      |            1971 | 100.00%    |         188123 | 100.00%    |

#### Validation set

| Class   |   Total Patients | % Patients   |   Total Volumes | % Volume   |   Total Slices | % Slices   |
|:--------|-----------------:|:-------------|----------------:|:-----------|---------------:|:-----------|
| CP      |              104 | 40.31%       |             207 | 43.86%     |          23818 | 49.66%     |
| NCP     |              127 | 49.22%       |             215 | 45.55%     |          19526 | 40.71%     |
| Normal  |               27 | 10.47%       |              50 | 10.59%     |           4617 | 9.63%      |
| **Total** |            258 | 100.00%      |             472 | 100.00%    |          47961 | 100.00%    |
