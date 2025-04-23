#!/usr/bin/env python3
"""
Script: dataset_train_val_split.py

Simulate a stratified train/validation split by patient for a dataset organized as:

    dataset_root/Class/PatientID/ScanID/*.png|.jpg|.tif|.tiff

It splits patients into train and validation sets by class to preserve proportions,
and prints summary tables of patients, volumes and slices in each split.

Usage:
    python dataset_train_val_split.py <dataset_root> --train_ratio 80 [--volume_size 30] [--seed 42]

Example:
    python dataset_train_val_split.py ./data/ccccii_selected --train_ratio 90 --volume_size 30 --seed 123
"""

import sys
import os
import argparse
import random
from collections import defaultdict
from tabulate import tabulate

def parse_patients(dataset_root, classes):
    # build map class -> list of patient IDs
    patients = {}
    for cls in classes:
        cls_dir = os.path.join(dataset_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        pids = [pid for pid in os.listdir(cls_dir)
                if os.path.isdir(os.path.join(cls_dir, pid))]
        patients[cls] = pids
    return patients

def count_stats(dataset_root, subset_map, classes, volume_size):
    slice_counts = defaultdict(int)
    volume_counts = defaultdict(int)
    for cls in classes:
        for pid in subset_map.get(cls, []):
            patient_dir = os.path.join(dataset_root, cls, pid)
            for scan in os.listdir(patient_dir):
                scan_dir = os.path.join(patient_dir, scan)
                if not os.path.isdir(scan_dir):
                    continue
                files = [f for f in os.listdir(scan_dir)
                         if os.path.isfile(os.path.join(scan_dir, f))]
                count = len(files)
                if count >= volume_size:
                    slice_counts[cls] += count
                    volume_counts[cls] += 1

    patient_counts = {cls: len(subset_map.get(cls, [])) for cls in classes}
    total_patients = sum(patient_counts.values())
    total_volumes = sum(volume_counts.values())
    total_slices = sum(slice_counts.values())

    patient_percents = {
        cls: (patient_counts[cls] / total_patients * 100 if total_patients else 0.0)
        for cls in classes
    }
    volume_percents = {
        cls: (volume_counts[cls] / total_volumes * 100 if total_volumes else 0.0)
        for cls in classes
    }
    slice_percents = {
        cls: (slice_counts[cls] / total_slices * 100 if total_slices else 0.0)
        for cls in classes
    }

    return (
        patient_counts, patient_percents,
        volume_counts, volume_percents,
        slice_counts, slice_percents,
        total_patients, total_volumes, total_slices
    )

def print_table(title, dataset_name, classes, stats):
    (
        patient_counts, patient_percents,
        volume_counts, volume_percents,
        slice_counts, slice_percents,
        total_patients, total_volumes, total_slices
    ) = stats

    headers = [
        "Class",
        "Total Patients", "% Patients",
        "Total Volumes", "% Volume",
        "Total Slices", "% Slices"
    ]
    rows = []
    for cls in classes:
        rows.append([
            cls,
            patient_counts.get(cls, 0), f"{patient_percents.get(cls, 0):.2f}%",
            volume_counts.get(cls, 0), f"{volume_percents.get(cls, 0):.2f}%",
            slice_counts.get(cls, 0), f"{slice_percents.get(cls, 0):.2f}%"
        ])
    rows.append([
        "Total",
        total_patients, "100.00%",
        total_volumes, "100.00%",
        total_slices, "100.00%"
    ])

    print()
    print(f"### {title} ({dataset_name})")
    print()
    print(tabulate(rows, headers=headers, tablefmt="pipe"))
    print()

def main():
    parser = argparse.ArgumentParser(
        description="Simulate stratified train/validation split by patient"
    )
    parser.add_argument(
        "dataset_root",
        help="Path to the dataset root directory"
    )
    parser.add_argument(
        "--train_ratio",
        type=int,
        default=80,
        help="Percentage of patients in the train split (default: 80)"
    )
    parser.add_argument(
        "--volume_size",
        type=int,
        default=30,
        help="Minimum number of slices to count a volume (default: 30)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_root):
        sys.exit(f"Error: '{args.dataset_root}' not found or not a directory.")

    # detect class subdirectories
    classes = sorted([
        d for d in os.listdir(args.dataset_root)
        if os.path.isdir(os.path.join(args.dataset_root, d))
    ])

    # map class -> patient IDs
    patient_map = parse_patients(args.dataset_root, classes)

    # split by class
    random.seed(args.seed)
    train_map = {}
    val_map = {}
    for cls, pids in patient_map.items():
        n_total = len(pids)
        n_train = int(round(n_total * args.train_ratio / 100))
        pids_shuf = pids.copy()
        random.shuffle(pids_shuf)
        train_map[cls] = pids_shuf[:n_train]
        val_map[cls] = pids_shuf[n_train:]

    # count stats
    train_stats = count_stats(
        args.dataset_root, train_map, classes, args.volume_size
    )
    val_stats = count_stats(
        args.dataset_root, val_map, classes, args.volume_size
    )

    dataset_name = os.path.basename(os.path.normpath(args.dataset_root))
    print_table("Train set", dataset_name, classes, train_stats)
    print_table("Validation set", dataset_name, classes, val_stats)

if __name__ == "__main__":
    main()
