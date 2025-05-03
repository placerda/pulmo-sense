#!/usr/bin/env python3
"""
Compute patient, study (volume), and slice statistics for related dataset directories.

Given a base dataset directory (e.g. data/ccccii_selected), this script
locates all sibling directories that start with the same base name:

    data/ccccii_selected
    data/ccccii_selected_train
    data/ccccii_selected_test
    data/ccccii_selected_fold_1_train
    ...

and for each generates a table of patient, volume (study), and slice counts
and percentages per class.

Usage:
    python dataset_statistics.py data/ccccii_selected
"""

import sys
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from tabulate import tabulate


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('dataset_statistics')


class DatasetStatistics:
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.logger = setup_logging()
        self.logger.info(f"Computing stats for {self.dataset_path}")

        # class-level counters
        self.slice_counts = defaultdict(int)
        self.volume_counts = defaultdict(int)
        self.patient_counts = defaultdict(int)

        # totals
        self.total_slices = 0
        self.total_volumes = 0
        self.total_patients = 0

        # percentages
        self.slice_percents = {}
        self.volume_percents = {}
        self.patient_percents = {}

        self._gather()

    def _gather(self):
        # map patient -> class to avoid double counting
        patient_classes = {}

        for class_dir in sorted(self.dataset_path.iterdir()):
            if not class_dir.is_dir():
                continue
            cls = class_dir.name
            for patient_dir in class_dir.iterdir():
                if not patient_dir.is_dir():
                    continue
                pid = patient_dir.name
                # record patient class if first seen
                if pid not in patient_classes:
                    patient_classes[pid] = cls
                # count volumes and slices per scan directory
                for scan_dir in patient_dir.iterdir():
                    if not scan_dir.is_dir():
                        continue
                    files = [f for f in scan_dir.iterdir() if f.is_file()]
                    count = len(files)
                    # count every scan as a volume
                    self.volume_counts[cls] += 1
                    self.slice_counts[cls] += count

        # compute patient counts per class
        for cls in patient_classes.values():
            self.patient_counts[cls] += 1

        # aggregate totals
        self.total_patients = sum(self.patient_counts.values())
        self.total_volumes = sum(self.volume_counts.values())
        self.total_slices = sum(self.slice_counts.values())

        # compute percentages
        if self.total_patients > 0:
            for cls, cnt in self.patient_counts.items():
                self.patient_percents[cls] = cnt / self.total_patients * 100
        if self.total_volumes > 0:
            for cls, cnt in self.volume_counts.items():
                self.volume_percents[cls] = cnt / self.total_volumes * 100
        if self.total_slices > 0:
            for cls, cnt in self.slice_counts.items():
                self.slice_percents[cls] = cnt / self.total_slices * 100

    def get_table(self) -> str:
        headers = [
            "Class",
            "Patients", "% Patients",
            "Volumes", "% Volumes",
            "Slices", "% Slices"
        ]
        rows = []
        classes = sorted(set(list(self.patient_counts) + list(self.volume_counts) + list(self.slice_counts)))
        for cls in classes:
            rows.append([
                cls,
                self.patient_counts.get(cls, 0), f"{self.patient_percents.get(cls,0):.2f}%",
                self.volume_counts.get(cls, 0), f"{self.volume_percents.get(cls,0):.2f}%",
                self.slice_counts.get(cls, 0), f"{self.slice_percents.get(cls,0):.2f}%"
            ])
        # total row
        rows.append([
            "Total",
            self.total_patients, "100.00%",
            self.total_volumes, "100.00%",
            self.total_slices, "100.00%"
        ])
        return tabulate(rows, headers=headers, tablefmt="pipe")


def main():
    parser = argparse.ArgumentParser(
        description="Compute stats for base dataset and its derived splits"
    )
    parser.add_argument(
        'base_dir',
        type=str,
        help="Path to base dataset directory (e.g. data/ccccii_selected)"
    )
    args = parser.parse_args()

    base_path = Path(args.base_dir)
    if not base_path.is_dir():
        sys.exit(f"Error: {base_path} not found or not a directory.")

    parent = base_path.parent
    prefix = base_path.name

    # find all related dataset dirs by prefix
    dataset_dirs = sorted([
        d for d in parent.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
    ])

    if not dataset_dirs:
        sys.exit(f"No directories starting with '{prefix}' found in {parent}.")

    for dataset_dir in dataset_dirs:
        stats = DatasetStatistics(dataset_path=dataset_dir)
        print(f"\n### Dataset Statistics: {dataset_dir.name}\n")
        print(stats.get_table())
        print("")

if __name__ == '__main__':
    main()
