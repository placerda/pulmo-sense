#!/usr/bin/env python3
"""
compare_images.py

This script compares a PNG file from the training dataset and a PNG file from the test dataset.
It computes several metrics (dimensions, mean intensity, standard deviation, contrast, and histogram similarity)
and prints a terminal report using the rich library.
"""

import argparse
import sys
from PIL import Image
import numpy as np
from rich.console import Console
from rich.table import Table

def analyze_image(image_path):
    """
    Analyze the image by converting it to grayscale and computing key metrics.
    """
    try:
        img = Image.open(image_path)
    except Exception as e:
        raise IOError(f"Could not open image {image_path}: {e}")
    
    # Convert image to grayscale (if not already)
    img_gray = img.convert("L")
    arr = np.array(img_gray)
    
    # Get image dimensions
    width, height = img.size
    
    # Compute basic statistics
    mean_intensity = np.mean(arr)
    std_intensity = np.std(arr)
    min_intensity = int(np.min(arr))
    max_intensity = int(np.max(arr))
    contrast = max_intensity - min_intensity  # dynamic range

    
    # Get normalized histogram (256 bins for grayscale)
    histogram = np.array(img_gray.histogram())
    normalized_histogram = histogram / histogram.sum()
    
    return {
        "path": image_path,
        "width": width,
        "height": height,
        "mean": mean_intensity,
        "std": std_intensity,
        "min": min_intensity,
        "max": max_intensity,
        "contrast": contrast,
        "histogram": normalized_histogram
    }

def compare_histograms(hist1, hist2):
    """
    Compare two normalized histograms using the L2 norm difference.
    """
    return np.linalg.norm(hist1 - hist2)

def generate_report(train_metrics, test_metrics):
    """
    Generate and print a terminal report comparing the two images.
    """
    console = Console()
    table = Table(title="CT Image Comparison Report")

    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Training Dataset", style="magenta")
    table.add_column("Test Dataset", style="green")
    table.add_column("Difference", style="red")

    # Compare dimensions: width and height
    width_diff = abs(train_metrics["width"] - test_metrics["width"])
    table.add_row("Width (pixels)",
                  str(train_metrics["width"]),
                  str(test_metrics["width"]),
                  str(width_diff))
    
    height_diff = abs(train_metrics["height"] - test_metrics["height"])
    table.add_row("Height (pixels)",
                  str(train_metrics["height"]),
                  str(test_metrics["height"]),
                  str(height_diff))

    # Mean intensity
    mean_diff = abs(train_metrics["mean"] - test_metrics["mean"])
    table.add_row("Mean Intensity",
                  f"{train_metrics['mean']:.2f}",
                  f"{test_metrics['mean']:.2f}",
                  f"{mean_diff:.2f}")

    # Standard deviation (contrast variability)
    std_diff = abs(train_metrics["std"] - test_metrics["std"])
    table.add_row("Std Dev Intensity",
                  f"{train_metrics['std']:.2f}",
                  f"{test_metrics['std']:.2f}",
                  f"{std_diff:.2f}")

    # Contrast (max-min difference)
    contrast_diff = abs(train_metrics["contrast"] - test_metrics["contrast"])
    table.add_row("Contrast (Max-Min)",
                  f"{train_metrics['contrast']}",
                  f"{test_metrics['contrast']}",
                  f"{contrast_diff}")

    # Histogram difference (L2 norm)
    hist_diff = compare_histograms(train_metrics["histogram"], test_metrics["histogram"])
    table.add_row("Histogram Difference (L2 norm)",
                  "Reference",
                  f"{hist_diff:.4f}",
                  f"{hist_diff:.4f}")

    console.print(table)

def main():
    parser = argparse.ArgumentParser(
        description="Compare a training CT image with a test CT image and generate a terminal report."
    )
    # Add positional arguments with default values if not provided
    parser.add_argument("train_image", nargs="?", default="data/ccccii/NCP/16/1164/0052.png",
                        help="Path to the training dataset PNG file (default: data/ccccii/NCP/16/1164/0052.png)")
    parser.add_argument("test_image", nargs="?", default="data/mosmed_png/NCP/study_0256/study_0256/0010.png",
                        help="Path to the test dataset PNG file (default: data/mosmed_png/NCP/study_0256/study_0256/0000.png)")
    args = parser.parse_args()

    try:
        train_metrics = analyze_image(args.train_image)
    except Exception as e:
        print(f"Error loading training image: {e}")
        sys.exit(1)

    try:
        test_metrics = analyze_image(args.test_image)
    except Exception as e:
        print(f"Error loading test image: {e}")
        sys.exit(1)

    generate_report(train_metrics, test_metrics)

if __name__ == "__main__":
    main()
