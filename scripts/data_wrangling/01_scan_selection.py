#!/usr/bin/env python3
"""
This script analyzes CT scan folders to detect cases where multiple exams may be mixed within a single folder. 
It works by computing the pixel-wise absolute differences between consecutive slices, then calculating the 
global standard deviation of those differences. A sudden large standard deviation indicates a possible 
boundary between different exams.

Each scan is also categorized based on its number of slices: scans with at least 30 valid slices are selected 
for further use, while those with fewer than 30 are recorded separately. The script additionally logs 
processing errors (such as unreadable images) and unsupported file formats, and generates detailed reports 
of slice-to-slice intensity variations.

The standard deviation is computed across all pixel differences, and measures how widely pixel changes are 
spread out relative to their mean.

Usage:
    python scan_selection.py [root_dir] [--std-threshold STD]

Example:
    python scan_selection.py --std-threshold 50.0
    python scan_selection.py data/ccccii/CP/1620/4308 -s 50

Defaults:
    root_dir            = data/ccccii
    std_threshold       = 50.0 gray-levels

Outputs (saved in data/selection_logs/ folder):
    - multiple-scans.txt      : scans flagged for possible multiple exams (large intensity jumps)
    - selected-studies.txt    : scans with >=30 slices and no flagged jumps
    - few-slices.txt          : scans with <30 slices (listed with slice counts)
    - each-slice.txt          : all slice-to-slice standard deviation measurements
    - processing-error.txt    : scans where reading an image failed
    - other-formats.txt       : scans containing unsupported image formats
"""

import os
import argparse
from PIL import Image
import numpy as np

# Supported extensions (including TIFF)
VALID_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


def compute_std_diff(prev_img, img):
    diffs = np.abs(img - prev_img)
    return diffs.std()


def check_and_categorize(scan_dir, std_thresh, m_out, sel_out, few_out, each_out, err_out, other_out):
    """
    Process a single scan directory:
      - Count and list valid slices (.png, .jpg, .jpeg, .tif, .tiff)
      - If <30 slices, log in few_slices
      - Otherwise compute std diffs; flag first over-threshold jump
      - Log any processing errors
    """
    files = os.listdir(scan_dir)

    # List valid image slices
    slices = sorted(f for f in files if f.lower().endswith(VALID_EXTS))
    count = len(slices)

    # Short study
    if count < 30:
        last = slices[-1] if slices else ''
        msg = f"{scan_dir}: {last} -> {count} slices"
        few_out.write(msg + '\n')
        few_out.flush()
        print(msg)
        return

    prev_img = None
    prev_fname = None
    # Compare consecutive slices
    for fname in slices:
        path = os.path.join(scan_dir, fname)
        try:
            img = np.array(Image.open(path).convert('L'), dtype=np.float32)
        except Exception as e:
            err = f"{scan_dir}: {fname} -> {type(e).__name__}: {e}"
            err_out.write(err + '\n')
            err_out.flush()
            print(f"Error: {err}")
            return

        if prev_img is not None:
            try:
                std_diff = compute_std_diff(prev_img, img)
            except Exception as e:
                err = f"{scan_dir}: {fname} -> {type(e).__name__}: {e}"
                err_out.write(err + '\n')
                err_out.flush()
                print(f"Error: {err}")
                return

            msg = f"{scan_dir}: {prev_fname} -> {fname} std_diff = {std_diff:.2f}"
            each_out.write(msg + '\n')
            each_out.flush()
            print(msg)
            if std_diff > std_thresh:
                m_out.write(msg + '\n')
                m_out.flush()
                print(f"Flagged: {msg}")
                return

        prev_img = img
        prev_fname = fname

    # No flag: select this scan
    sel_out.write(f"{scan_dir}\n")
    sel_out.flush()
    print(f"Selected: {scan_dir} ({count} slices)")


def main():
    parser = argparse.ArgumentParser(
        description="Detect and categorize CT scan folders by std-dev jumps and slice count"
    )
    parser.add_argument(
        'root', nargs='?', default='data/ccccii',
        help='Root dataset directory or single scan directory'
    )
    parser.add_argument(
        '--std-threshold', '-s', type=float, default=50.0,
        help='Std-dev threshold (gray-levels)'
    )
    args = parser.parse_args()

    # Prepare logs outputs
    temp = 'data/selection_logs'
    os.makedirs(temp, exist_ok=True)
    paths = {
        'multiple': os.path.join(temp, 'multiple-scans.txt'),
        'selected': os.path.join(temp, 'selected-studies.txt'),
        'few':      os.path.join(temp, 'few-slices.txt'),
        'each':     os.path.join(temp, 'each-slice.txt'),
        'error':    os.path.join(temp, 'processing-error.txt'),
        'other':    os.path.join(temp, 'other-formats.txt'),
    }

    with open(paths['multiple'], 'w', buffering=1) as m_out, \
         open(paths['selected'], 'w', buffering=1) as sel_out, \
         open(paths['few'], 'w', buffering=1) as few_out, \
         open(paths['each'], 'w', buffering=1) as each_out, \
         open(paths['error'], 'w', buffering=1) as err_out, \
         open(paths['other'], 'w', buffering=1) as other_out:

        entries = os.listdir(args.root)
        # Single-scan if any valid image at root
        if any(f.lower().endswith(VALID_EXTS) for f in entries):
            check_and_categorize(
                args.root, args.std_threshold,
                m_out, sel_out, few_out, each_out, err_out, other_out
            )
        else:
            # Traverse class/patient/scan
            for cls in entries:
                cls_dir = os.path.join(args.root, cls)
                if not os.path.isdir(cls_dir): continue
                for patient in os.listdir(cls_dir):
                    pat_dir = os.path.join(cls_dir, patient)
                    if not os.path.isdir(pat_dir): continue
                    for scan in os.listdir(pat_dir):
                        scan_dir = os.path.join(pat_dir, scan)
                        if not os.path.isdir(scan_dir): continue
                        check_and_categorize(
                            scan_dir, args.std_threshold,
                            m_out, sel_out, few_out, each_out, err_out, other_out
                        )

    print("Done. Check data/selection_logs/ for outputs.")

if __name__ == '__main__':
    main()
