import os
import numpy as np
from PIL import Image
from unet_lungs_segmentation import LungsPredict

# Define input and output base directories.
input_base = "data/ccccii"
output_base = "data/ccccii_segmented"

# Initialize the lung segmentation predictor.
predictor = LungsPredict()

# Walk through the directory structure.
for root, dirs, files in os.walk(input_base):
    for file in files:
        if file.lower().endswith(".png"):
            input_path = os.path.join(root, file)
            
            # Load the image and convert to grayscale.
            img = Image.open(input_path).convert("L")
            img_np = np.array(img)

            # Run lung segmentation on the image.
            mask = predictor.segment_lungs(img_np)

            # If the output mask has an extra dimension, select the central slice.
            if mask.ndim == 3:
                central_index = mask.shape[-1] // 2
                mask = mask[:, :, central_index]

            # If the mask values are probabilities, threshold them to get a binary mask.
            if mask.dtype != np.uint8:
                mask = (mask > 0.5).astype(np.uint8)

            # Create a segmented image: keep only the lung regions.
            segmented_np = img_np * mask

            # Convert the NumPy array back to a PIL image.
            segmented_img = Image.fromarray(segmented_np.astype(np.uint8))

            # Build the corresponding output directory and file path.
            rel_path = os.path.relpath(root, input_base)
            output_dir = os.path.join(output_base, rel_path)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, file)

            # Save the segmented image.
            segmented_img.save(output_path)
            print(f"Segmented and saved: {output_path}")

print("All images processed!")
