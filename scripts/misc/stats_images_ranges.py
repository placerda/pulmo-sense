#!/usr/bin/env python3
"""
Compute and compare pixel intensity ranges for TIFF, JPEG, and PNG files across CT scan studies,
writing per-file stats as it processes and a final global summary.

Usage:
    python stats_image_ranges.py [root_dir]

Defaults:
    root_dir = data/ccccii

Outputs in temp/:
    image-range-summary.txt   (per-file range + global summary)
    range-errors.txt          (files with processing errors)
"""
import os
import argparse
from collections import defaultdict
import numpy as np
from PIL import Image

# Supported formats
FORMATS = {
    '.png': 'PNG',
    '.jpg': 'JPEG',
    '.jpeg': 'JPEG',
    '.tif': 'TIFF',
    '.tiff': 'TIFF'
}

def get_intensity_range(path):
    img = Image.open(path)
    arr = np.array(img)
    return arr.dtype, float(arr.min()), float(arr.max())


def main():
    parser = argparse.ArgumentParser(
        description="Compute intensity ranges for different image formats"
    )
    parser.add_argument(
        'root', nargs='?', default='data/ccccii',
        help='Root dataset directory'
    )
    args = parser.parse_args()

    temp_dir = 'temp'
    os.makedirs(temp_dir, exist_ok=True)
    summary_file = os.path.join(temp_dir, 'image-range-summary.txt')
    error_file = os.path.join(temp_dir, 'range-errors.txt')

    # Initialize stats collector
    stats = defaultdict(lambda: defaultdict(list))

    # Open summary and error outputs
    with open(summary_file, 'w', buffering=1) as summary_out, \
         open(error_file, 'w', buffering=1) as err_out:

        # Write header for per-file stats
        summary_out.write("Image Intensity Range Summary per file\n")
        summary_out.write("================================================================\n")

        # Walk through files and process
        for root_dir, _, files in os.walk(args.root):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in FORMATS:
                    continue
                path = os.path.join(root_dir, fname)
                try:
                    dtype, vmin, vmax = get_intensity_range(path)
                    fmt = FORMATS[ext]
                    # Write per-file line
                    summary_out.write(
                        f"{path}: format={fmt}, dtype={dtype}, min={vmin:.2f}, max={vmax:.2f}\n"
                    )
                    summary_out.flush()
                    # Collect for global
                    stats[fmt][dtype].append((vmin, vmax))
                except Exception as e:
                    err_out.write(f"{path}: {type(e).__name__}: {e}\n")
                    err_out.flush()

        # After per-file listing, write global summary
        summary_out.write("\nGlobal Summary by Format and Dtype\n")
        summary_out.write("================================================================\n")
        for fmt, by_dtype in stats.items():
            summary_out.write(f"Format: {fmt}\n")
            for dtype, ranges in by_dtype.items():
                mins, maxs = zip(*ranges)
                global_min = min(mins)
                global_max = max(maxs)
                summary_out.write(
                    f"  Dtype {dtype}: global_min={global_min:.2f}, global_max={global_max:.2f}\n"
                )
            summary_out.write("\n")

    print(f"Processing done. See '{summary_file}' and '{error_file}'.")

if __name__ == '__main__':
    main()
