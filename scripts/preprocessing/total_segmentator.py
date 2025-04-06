import os
import numpy as np
from PIL import Image
from skimage import exposure, filters, morphology, io

# Define input and output base directories
# input_base = "data/ccccii"
# output_base = "data/ccccii_segmented"
input_base = "data/mosmed_png"
output_base = "data/mosmed_segmented"


# Walk through the directory structure of the input dataset
for root, dirs, files in os.walk(input_base):
    for file in files:
        if file.lower().endswith('.png'):
            # Construct full input file path
            input_path = os.path.join(root, file)
            # Create corresponding output directory path (mirroring input structure)
            rel_path = os.path.relpath(root, input_base)
            output_dir = os.path.join(output_base, rel_path)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, file)

            try:
                # Load image using PIL and convert to grayscale
                img = Image.open(input_path).convert('L')
                img_np = np.array(img)

                # Apply histogram equalization to improve contrast
                img_eq = exposure.equalize_hist(img_np)

                # Compute Otsu threshold on the equalized image
                thresh = filters.threshold_otsu(img_eq)
                binary_mask = img_eq > thresh

                # Morphological closing to fill holes (adjust radius if necessary)
                selem = morphology.disk(5)
                binary_closed = morphology.closing(binary_mask, selem)

                # Remove small objects to clean up noise (adjust min_size if needed)
                clean_mask = morphology.remove_small_objects(binary_closed, min_size=64)

                # Create segmented image by applying the mask
                segmented = img_np * clean_mask.astype(np.uint8)

                # Save segmented image as PNG
                io.imsave(output_path, segmented.astype(np.uint8))
                print(f"Segmented: {input_path} -> {output_path}")

            except Exception as e:
                print(f"Failed to process {input_path}: {e}")