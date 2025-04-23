#!/usr/bin/env python3
"""
Split a CT-scan dataset into train/test by patient,
preserving class stratification and avoiding data leakage.

Example:
    python split_dataset_by_patient.py \
        --input-dir data/ccccii_selected \
        --train-percentage 70 \
        --seed 123

This produces:
    data/ccccii_selected_train/…
    data/ccccii_selected_test/…
where each patient's scans all live in one split.
"""

import argparse
import random
import shutil
from pathlib import Path

def copytree(src: Path, dst: Path):
    """Recursively copy a folder, overwriting the destination if it exists."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def main():
    p = argparse.ArgumentParser(
        description="Stratified patient-level train/test split of a folder dataset"
    )
    p.add_argument(
        '-i','--input-dir', required=True,
        help="Input dataset root (e.g. data/ccccii_selected)"
    )
    p.add_argument(
        '-t','--train-percentage', type=float, required=True,
        help="Percent of patients per class to put into the train split (0–100)"
    )
    p.add_argument(
        '--seed', type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = p.parse_args()

    random.seed(args.seed)
    input_root = Path(args.input_dir).resolve()
    parent, name = input_root.parent, input_root.name
    train_root = parent / f"{name}_train"
    test_root  = parent / f"{name}_test"

    # 1) Gather patient IDs by class
    patients_by_class = {}
    for class_dir in input_root.iterdir():
        if not class_dir.is_dir(): continue
        class_name = class_dir.name
        pats = [d.name for d in class_dir.iterdir() if d.is_dir()]
        patients_by_class[class_name] = pats

    # 2) Stratified split of patient IDs
    train_patients = {}
    test_patients  = {}
    for cls, pats in patients_by_class.items():
        shuffled = sorted(pats)
        random.shuffle(shuffled)
        n_train = int(len(shuffled) * args.train_percentage / 100.0)
        train_patients[cls] = set(shuffled[:n_train])
        test_patients[cls]  = set(shuffled[n_train:])
        print(f"Class '{cls}': {len(train_patients[cls])} train, {len(test_patients[cls])} test patients")

    # 3) Copy entire patient folders into each split
    for cls, pats in patients_by_class.items():
        for pid in pats:
            src = input_root / cls / pid
            # choose split based on patient ID
            dst_root = train_root if pid in train_patients[cls] else test_root
            dst = dst_root / cls / pid
            print(f"Copying {src} → {dst}")
            copytree(src, dst)

    print("\nFinished splitting dataset.")

if __name__ == '__main__':
    main()
