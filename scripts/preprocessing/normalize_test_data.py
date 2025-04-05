#!/usr/bin/env python3
"""
normalize_test_images.py

This script computes the global pixel intensity mean and standard deviation from
the training dataset (data/ccccii) and then normalizes all test images in the
test dataset (data/mosmed_png) to match the training distribution.
Normalized images are saved to data/mosmed_png_normal preserving the folder structure.
"""

import os
import argparse
import numpy as np
from PIL import Image
from rich.console import Console

console = Console()

def compute_training_stats(train_root):
    """
    Iterates through all PNG files in train_root and computes the global mean and std.
    Assumes images are grayscale or converts them to grayscale.
    """
    total_sum = 0.0
    total_sq_sum = 0.0
    total_pixels = 0

    # Walk through the training dataset folder
    for root, _, files in os.walk(train_root):
        for file in files:
            if file.lower().endswith(".png"):
                img_path = os.path.join(root, file)
                try:
                    img = Image.open(img_path).convert("L")  # force grayscale
                except Exception as e:
                    console.print(f"[red]Error loading {img_path}: {e}[/red]")
                    continue

                arr = np.array(img, dtype=np.float64)
                total_sum += arr.sum()
                total_sq_sum += np.square(arr).sum()
                total_pixels += arr.size

    if total_pixels == 0:
        raise ValueError("No training images found.")

    train_mean = total_sum / total_pixels
    train_variance = (total_sq_sum / total_pixels) - train_mean**2
    train_std = np.sqrt(train_variance)

    return train_mean, train_std

def normalize_image(test_img, train_mean, train_std):
    """
    Normalizes a test image so that its pixel intensity distribution matches the training dataset.
    The normalization is done per-image:
    
      new_pixel = (pixel - test_mean) * (train_std / test_std) + train_mean
      
    The result is clipped to [0, 255] and converted to uint8.
    """
    arr = np.array(test_img, dtype=np.float64)
    test_mean = arr.mean()
    test_std = arr.std()

    if test_std == 0:
        console.print("[yellow]Warning: Test image has zero standard deviation. Skipping normalization.[/yellow]")
        return test_img

    # Apply the normalization transformation
    normalized = (arr - test_mean) * (train_std / test_std) + train_mean

    # Clip the values to valid range and convert to uint8
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    return Image.fromarray(normalized)

def process_test_images(test_root, output_root, train_mean, train_std):
    """
    Processes all PNG images in the test dataset folder, normalizes them,
    and saves the results under output_root preserving the directory structure.
    """
    for root, _, files in os.walk(test_root):
        for file in files:
            if file.lower().endswith(".png"):
                input_path = os.path.join(root, file)
                # Determine the relative path and the corresponding output path
                rel_path = os.path.relpath(input_path, test_root)
                output_path = os.path.join(output_root, rel_path)

                # Ensure the output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                try:
                    img = Image.open(input_path).convert("L")
                except Exception as e:
                    console.print(f"[red]Error loading {input_path}: {e}[/red]")
                    continue

                normalized_img = normalize_image(img, train_mean, train_std)
                try:
                    normalized_img.save(output_path)
                    console.print(f"[green]Saved normalized image:[/green] {output_path}")
                except Exception as e:
                    console.print(f"[red]Error saving {output_path}: {e}[/red]")

def main():
    parser = argparse.ArgumentParser(
        description="Normalize test dataset images based on training dataset intensity statistics."
    )
    parser.add_argument(
        "--train_root",
        type=str,
        default="data/ccccii",
        help="Root folder of the training dataset (default: data/ccccii)"
    )
    parser.add_argument(
        "--test_root",
        type=str,
        default="data/mosmed_png",
        help="Root folder of the test dataset (default: data/mosmed_png)"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/mosmed_png_normal",
        help="Output folder for normalized test images (default: data/mosmed_png_normal)"
    )
    args = parser.parse_args()

    console.print("[bold cyan]Computing training dataset statistics...[/bold cyan]")
    try:
        train_mean, train_std = compute_training_stats(args.train_root)
        console.print(f"Training Mean: [green]{train_mean:.2f}[/green], Training Std: [green]{train_std:.2f}[/green]")
    except Exception as e:
        console.print(f"[red]Error computing training statistics: {e}[/red]")
        return

    console.print(f"[bold cyan]Processing test images from {args.test_root}...[/bold cyan]")
    process_test_images(args.test_root, args.output_root, train_mean, train_std)
    console.print(f"[bold cyan]Normalization complete. Normalized images are saved in {args.output_root}[/bold cyan]")

if __name__ == "__main__":
    main()
