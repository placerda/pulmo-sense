#!/usr/bin/env python3
"""
File: /datasets/dataset_statistics.py

This script computes a combined patient, volume, and slice summary table for a dataset.
It navigates through the directory structure under a specified root directory
and processes each dataset subdirectory separately. Any subdirectory named "ignore"
will be skipped.

Usage:
    python -m datasets.dataset_statistics <data_root>
Example:
    python -m datasets.dataset_statistics ./data
"""

import sys
import os
import argparse
from collections import defaultdict
from tabulate import tabulate
import logging
import warnings

warnings.filterwarnings("ignore", message=r".*found in sys.modules after import.*")

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger('dataset_statistics')

class DatasetStatistics:
    def __init__(self, dataset_root, volume_size=30):
        self.dataset_root = dataset_root
        self.volume_size = volume_size
        self.classes = ['CP', 'NCP', 'Normal']
        self.logger = setup_logging()
        self.logger.info(f"Initialized with volume_size={self.volume_size}")

        # Counters
        self.slice_counts = defaultdict(int)
        self.volume_counts = defaultdict(int)
        self.patients_per_class = defaultdict(int)

        self.total_slices = 0
        self.total_volumes = 0
        self.total_patients = 0

        # Percent maps
        self.slice_percents = {}
        self.volume_percents = {}
        self.patient_percents = {}

        # Workflow
        self._gather()

    def _gather(self):
        self.logger.info(f"Gathering statistics from: {self.dataset_root}")
        unique_patients = {}
        for cls_idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(self.dataset_root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for pid in os.listdir(cls_dir):
                patient_dir = os.path.join(cls_dir, pid)
                if not os.path.isdir(patient_dir):
                    continue
                unique_patients[pid] = cls_idx
                for scan in os.listdir(patient_dir):
                    scan_dir = os.path.join(patient_dir, scan)
                    if not os.path.isdir(scan_dir):
                        continue
                    count = len([f for f in os.listdir(scan_dir)
                                 if os.path.isfile(os.path.join(scan_dir, f))])
                    if count >= self.volume_size:
                        self.slice_counts[cls] += count
                        self.volume_counts[cls] += 1
        # Patient counts
        for pid, lbl in unique_patients.items():
            cls = self.classes[lbl]
            self.patients_per_class[cls] += 1

        # Totals
        self.total_patients = sum(self.patients_per_class.values())
        self.total_volumes = sum(self.volume_counts.values())
        self.total_slices = sum(self.slice_counts.values())

        # Percents
        if self.total_patients > 0:
            for cls in self.classes:
                self.patient_percents[cls] = self.patients_per_class[cls] / self.total_patients * 100
        if self.total_volumes > 0:
            for cls in self.classes:
                self.volume_percents[cls] = self.volume_counts[cls] / self.total_volumes * 100
        if self.total_slices > 0:
            for cls in self.classes:
                self.slice_percents[cls] = self.slice_counts[cls] / self.total_slices * 100

    def get_table(self):
        # Combined table headers
        headers = [
            "Class",
            "Total Patients", "% Patients",
            "Total Volumes", "% Volume",
            "Total Slices", "% Slices"
        ]
        rows = []
        # Rows per class
        for cls in self.classes:
            rows.append([
                cls,
                self.patients_per_class[cls], f"{self.patient_percents.get(cls,0):.2f}%",
                self.volume_counts[cls], f"{self.volume_percents.get(cls,0):.2f}%",
                self.slice_counts[cls], f"{self.slice_percents.get(cls,0):.2f}%"
            ])
        # Total row
        rows.append([
            "Total",
            self.total_patients, "100.00%",
            self.total_volumes, "100.00%",
            self.total_slices, "100.00%"
        ])
        return tabulate(rows, headers=headers, tablefmt="pipe")


def main():
    parser = argparse.ArgumentParser(
        description="Combined patient, volume, and slice statistics table."
    )
    parser.add_argument("dataset_root", type=str,
                        help="Path to the data root directory containing multiple datasets (e.g. ./data)")
    parser.add_argument("--volume_size", type=int, default=30,
                        help="Minimum number of slices to consider one volume (default=30)")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_root):
        sys.exit(f"Error: {args.dataset_root} not found.")

    datasets = sorted([
        d for d in os.listdir(args.dataset_root)
        if os.path.isdir(os.path.join(args.dataset_root, d)) and d.lower() != "ignore"
    ])

    if not datasets:
        sys.exit("No valid dataset directories found in the provided root.")

    for dataset in datasets:
        stats = DatasetStatistics(
            dataset_root=os.path.join(args.dataset_root, dataset),
            volume_size=args.volume_size
        )
        table = stats.get_table()

        print()
        print(f"### Dataset Statistics ({dataset})")
        print()
        print(table)
        print()

if __name__ == "__main__":
    main()
