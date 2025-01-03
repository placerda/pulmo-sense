import sys
import os
import json
from collections import defaultdict
from tabulate import tabulate

class DatasetStatistics:
    def __init__(self, dataset_root):
        self.ccccii_dataset = self._load_dataset(dataset_root)

    def _load_dataset(self, root_dir):
        dataset = defaultdict(int)
        labels = ['CP', 'NCP', 'Normal']

        for label in labels:
            label_dir = os.path.join(root_dir, label)
            for patient_id in os.listdir(label_dir):
                patient_dir = os.path.join(label_dir, patient_id)
                scan_dirs = [os.path.join(patient_dir, scan_id) for scan_id in os.listdir(patient_dir)]
                scan_dirs = [d for d in scan_dirs if len(os.listdir(d)) >= 30]  # Only keep scans with at least 30 slices
                if scan_dirs:
                    dataset[label] += 1
        return dataset

    def get_statistics(self):
        datasets = [
            self.ccccii_dataset
        ]
        dataset_names = [
            'CCCCIIDataset'
        ]

        samples_table = []
        train_val_table = []

        for dataset, name in zip(datasets, dataset_names):
            total_samples = sum(dataset.values())
            class_counts_str = ', '.join([f"{k}: {v}" for k, v in dataset.items()])
            percentages_str = ', '.join([f"{k}: {v/total_samples*100:.2f}%" for k, v in dataset.items()])

            train_val_class_split = f"Train: " + ' '.join([f"{k.split()[0]} {0.8 * v:.0f}" for k, v in dataset.items()]) + " - Val: " + ' '.join([f"{k.split()[0]} {0.2 * v:.0f}" for k, v in dataset.items()])
            train_val_percentages = f"Train: " + ' '.join([f"{k.split()[0]} {0.8 * v/(0.8 * total_samples)*100:.2f}%" for k, v in dataset.items()]) + " - Val: " + ' '.join([f"{k.split()[0]} {0.2 * v/(0.2 * total_samples)*100:.2f}%" for k, v in dataset.items()])

            samples_table.append([name, total_samples, class_counts_str, percentages_str])
            train_val_table.append([name, train_val_class_split, train_val_percentages])

        samples_markdown_table = tabulate(samples_table, headers=["Dataset", "Samples", "Samples by Class", "Percentages by Class"], tablefmt="pipe")
        train_val_markdown_table = tabulate(train_val_table, headers=["Dataset", "Train/Val by Class", "Percentages by Class"], tablefmt="pipe")
        
        return samples_markdown_table, train_val_markdown_table

def main(dataset_root):
    dataset_stats = DatasetStatistics(dataset_root)
    table1, table2 = dataset_stats.get_statistics()
    print(table1)
    print("")    
    print(table2)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <ccccii_root>")
        sys.exit(1)

    dataset_root = sys.argv[1]
    
    main(dataset_root)


# Example:
# python -m datasets.dataset_statistics data/ccccii