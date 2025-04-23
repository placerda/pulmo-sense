#!/usr/bin/env python3
"""
Prepare an experiment dataset by copying selected scan folders into a new dataset root.

Usage:
    python prepare_experiment_dataset.py \
        --list-file temp/selected-segmented.txt \
        --dataset-name ccccii_selected \
        [--source-root data/ccccii]

This will copy each scan path in the list file (e.g. data/ccccii/Normal/2248/703)
into data/ccccii_selected/Normal/2248/703.
"""

import argparse
import os
import shutil

def copy_folder(src: str, dst: str):
    """Recursively copy a folder, overwriting the destination if it exists."""
    if not os.path.isdir(src):
        print(f"[Warning] Source not found: {src}")
        return
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copytree(src, dst)
    print(f"Copied: {src} -> {dst}")

def main():
    parser = argparse.ArgumentParser(
        description="Copy selected CT scan folders into a new dataset directory"
    )
    parser.add_argument(
        '-l', '--list-file', default='temp/selected-studies.txt',
        help='File listing selected scan directories (one per line)'
    )
    parser.add_argument(
        '-s', '--source-root', default='data/ccccii',
        help='Root directory of the original dataset'
    )
    parser.add_argument(
        '-d', '--dataset-name', required=True,
        help='Name of the new dataset directory to create under the same parent as source-root'
    )
    args = parser.parse_args()

    # Resolve absolute paths
    abs_src_root = os.path.abspath(args.source_root)
    abs_dest_root = os.path.join(os.path.dirname(abs_src_root), args.dataset_name)

    # Read and normalize list of scan paths
    if not os.path.isfile(args.list_file):
        print(f"[Error] List file not found: {args.list_file}")
        return
    with open(args.list_file, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]

    for path in paths:
        abs_path = os.path.abspath(path)
        # If the listed path lives under source-root, preserve its relative path
        if abs_path.startswith(abs_src_root + os.sep):
            rel = os.path.relpath(abs_path, abs_src_root)
            src = abs_path
        else:
            # Otherwise assume it's relative to source-root
            rel = path
            src = os.path.join(abs_src_root, rel)

        dst = os.path.join(abs_dest_root, rel)
        copy_folder(src, dst)

if __name__ == '__main__':
    main()
