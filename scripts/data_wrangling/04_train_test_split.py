#!/usr/bin/env python3
"""
Split a CT-scan dataset into train/test and k-fold cross-validation

This script generates the following directories alongside the input data root:

    data/ccccii_selected_train
    data/ccccii_selected_test
    data/ccccii_selected_fold_1_train
    data/ccccii_selected_fold_1_val
    ...
    data/ccccii_selected_fold_5_train
    data/ccccii_selected_fold_5_val

Ensures stratification by class and patient-level splitting without data leakage.
"""

import argparse
import random
import shutil
from pathlib import Path
import numpy as np


def copytree(src: Path, dst: Path):
    """Recursively copy a folder, overwriting the destination if it exists."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description="Patient-level stratified train/test split and k-fold cross-validation"
    )
    parser.add_argument(
        '-i','--input-dir', required=True,
        help="Input dataset root (e.g. data/ccccii_selected)"
    )
    parser.add_argument(
        '--train-percentage', type=float, default=90.0,
        help="Percent of patients per class to put into the train split (default: 90)"
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        '--folds', type=int, default=5,
        help="Number of folds (default: 5)"
    )
    args = parser.parse_args()

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Resolve paths
    input_root = Path(args.input_dir).resolve()
    parent = input_root.parent
    name = input_root.name

    # Define output directories with input name prefix
    train_root = parent / f"{name}_train"
    test_root  = parent / f"{name}_test"

    # 1) Gather patient IDs by class
    patients_by_class = {}
    for class_dir in input_root.iterdir():
        if not class_dir.is_dir():
            continue
        cls = class_dir.name
        patients = [d.name for d in class_dir.iterdir() if d.is_dir()]
        patients_by_class[cls] = patients

    # 2) Stratified 90/10 train/test split of patients
    train_patients = {}
    test_patients  = {}
    for cls, pats in patients_by_class.items():
        pats_sorted = sorted(pats)
        random.shuffle(pats_sorted)
        n_train = int(len(pats_sorted) * args.train_percentage / 100.0)
        train_patients[cls] = set(pats_sorted[:n_train])
        test_patients[cls]  = set(pats_sorted[n_train:])
        print(f"Class '{cls}': {len(train_patients[cls])} train, {len(test_patients[cls])} test patients")

    # 3) Copy patients into train/test
    for cls, pats in patients_by_class.items():
        for pid in pats:
            src = input_root / cls / pid
            dst_root = train_root if pid in train_patients[cls] else test_root
            dst = dst_root / cls / pid
            print(f"Copying {src} → {dst}")
            copytree(src, dst)

    # 4) Create k-fold splits from the train set
    folds = args.folds
    folds_by_class = {}
    for cls, pats in train_patients.items():
        pats_list = list(pats)
        random.shuffle(pats_list)
        splits = np.array_split(pats_list, folds)
        folds_by_class[cls] = [list(split) for split in splits]

    # 5) Generate directories for each fold
    for fold_idx in range(folds):
        fold_num = fold_idx + 1
        fold_train_root = parent / f"{name}_fold_{fold_num}_train"
        fold_val_root   = parent / f"{name}_fold_{fold_num}_val"
        print(f"\nGenerating fold {fold_num}: train vs val")

        for cls, splits in folds_by_class.items():
            val_pats = splits[fold_idx]
            train_pats = [pid for i, split in enumerate(splits) if i != fold_idx for pid in split]

            # Copy train patients for this fold
            for pid in train_pats:
                src = train_root / cls / pid
                dst = fold_train_root / cls / pid
                print(f"  Train Copy {src} → {dst}")
                copytree(src, dst)

            # Copy validation patients for this fold
            for pid in val_pats:
                src = train_root / cls / pid
                dst = fold_val_root / cls / pid
                print(f"  Val   Copy {src} → {dst}")
                copytree(src, dst)

    print("\nFinished train/test split and cross-validation folds.")

if __name__ == '__main__':
    main()
