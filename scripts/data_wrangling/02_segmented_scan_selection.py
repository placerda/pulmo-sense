#!/usr/bin/env python3
"""
This script classifies CT scan studies as either segmented or non-segmented based on their overall mean pixel intensity.

Segmented studies typically contain a larger proportion of black background (lower intensity values) due to prior 
manual or automatic segmentation. To detect this, the script calculates the average pixel intensity across all slices 
in a study: if the mean is below a given threshold (default = 50.0), the study is classified as segmented; otherwise, 
it is classified as non-segmented.

The script processes a list of study folders (paths) provided in a file, handles reading errors gracefully, and 
saves the results into three output files under the `temp/` folder.

Key points:
- Pixel intensities are assumed to be in the 0–255 grayscale range.
- Only '.png', '.jpg', and '.jpeg' files are considered valid slices.
- Errors while loading images are logged separately without interrupting the execution.

Usage:
    python count_segmented_scans.py [--selected-file PATH] [--threshold THRESH]

Example:
    python count_segmented_scans.py --threshold 50.0
    python count_segmented_scans.py --selected-file temp/selected-studies.txt -t 45

Defaults:
    selected_file = temp/selected-studies.txt
    threshold     = 50.0 (mean pixel intensity)

Outputs (in temp/ folder):
    - selected-segmented.txt       : paths of studies classified as segmented
    - selected-nonsegmented.txt    : paths of studies classified as non-segmented
    - segmentation-errors.txt      : errors encountered during processing
"""	

import os
import argparse
from PIL import Image
import numpy as np


def process_study(scan_dir, valid_exts, threshold, seg_out, nonseg_out, err_out):
    """
    Process a single scan directory: compute mean intensity, classify, write to appropriate file.
    Returns 'seg' if segmented, 'non' if non-segmented, or None on error/skip.
    """
    try:
        files = [f for f in os.listdir(scan_dir)
                 if f.lower().endswith(valid_exts)]
        if not files:
            return None
        means = []
        for fname in files:
            path = os.path.join(scan_dir, fname)
            try:
                img = np.array(Image.open(path).convert('L'), dtype=np.float32)
                means.append(img.mean())
            except Exception as e:
                err_out.write(f"{scan_dir} -> {fname}: {type(e).__name__}: {e}\n")
                err_out.flush()
                return None
        avg_mean = float(np.mean(means))
        if avg_mean < threshold:
            seg_out.write(f"{scan_dir}\n")
            seg_out.flush()
            print(f"Segmented: {scan_dir} (mean={avg_mean:.2f})")
            return 'seg'
        else:
            nonseg_out.write(f"{scan_dir}\n")
            nonseg_out.flush()
            print(f"Non-segmented: {scan_dir} (mean={avg_mean:.2f})")
            return 'non'
    except Exception as e:
        err_out.write(f"{scan_dir}: {type(e).__name__}: {e}\n")
        err_out.flush()
        print(f"Error: {scan_dir}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Count segmented vs non-segmented CT scan studies from selected list"
    )
    parser.add_argument(
        '--selected-file', '-s', default='temp/selected-studies.txt',
        help='Path to selected studies list (default: temp/selected-studies.txt)'
    )
    parser.add_argument(
        '--threshold', '-t', type=float, default=50.0,
        help='Mean intensity threshold (default: 50.0)'
    )
    args = parser.parse_args()

    if not os.path.isfile(args.selected_file):
        print(f"Selected file not found: {args.selected_file}")
        return

    os.makedirs('temp', exist_ok=True)
    seg_path = os.path.join('temp', 'selected-segmented.txt')
    nonseg_path = os.path.join('temp', 'selected-nonsegmented.txt')
    err_path = os.path.join('temp', 'segmentation-errors.txt')

    VALID_EXTS = ('.png', '.jpg', '.jpeg')
    total_seg = total_non = 0

    # Read list of selected scan directories
    with open(args.selected_file, 'r') as f:
        scan_dirs = [line.strip() for line in f if line.strip()]

    # Open output files
    with open(seg_path, 'w', buffering=1) as seg_out, \
         open(nonseg_path, 'w', buffering=1) as nonseg_out, \
         open(err_path, 'w', buffering=1) as err_out:

        for scan_dir in scan_dirs:
            if not os.path.isdir(scan_dir):
                err_out.write(f"Not a directory: {scan_dir}\n")
                err_out.flush()
                continue
            result = process_study(scan_dir, VALID_EXTS, args.threshold,
                                   seg_out, nonseg_out, err_out)
            if result == 'seg':
                total_seg += 1
            elif result == 'non':
                total_non += 1

    # Print summary
    print("\nSummary:")
    print(f"Segmented studies:     {total_seg}")
    print(f"Non-segmented studies: {total_non}")


if __name__ == '__main__':
    main()
