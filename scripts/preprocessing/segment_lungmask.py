import os
import numpy as np
from PIL import Image
import SimpleITK as sitk
from lungmask import LMInferer

# Initialize Lungmask inferer (using model "R231" by default)
inferer = LMInferer(modelname="R231")

# Define input and output base directories
input_base = "data/ccccii"
output_base = "data/ccccii_segmented"

# Walk through the directory structure of the input dataset
for root, dirs, files in os.walk(input_base):
    for file in files:
        if file.lower().endswith(".png"):
            # Construct full input file path
            input_path = os.path.join(root, file)
            
            # Create corresponding output directory path (mirroring input structure)
            rel_path = os.path.relpath(root, input_base)
            output_dir = os.path.join(output_base, rel_path)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, file)
            
            try:
                # Load the PNG image using PIL and convert to grayscale
                img_pil = Image.open(input_path).convert('L')
                img_np = np.array(img_pil)
                
                # For a 2D image, add a slice dimension to create a 3D volume
                if img_np.ndim == 2:
                    img_np_3d = img_np[np.newaxis, ...]  # shape becomes (1, height, width)
                else:
                    img_np_3d = img_np
                
                # Create a SimpleITK image from the numpy array (3D image)
                img = sitk.GetImageFromArray(img_np_3d)
                
                # Apply lung segmentation using Lungmask.
                # The inferer returns a segmentation with labels 1 (right lung) and 2 (left lung)
                seg = inferer.apply(img)
                
                # If seg is a numpy array, use it as is; otherwise, convert from SimpleITK image to array.
                if isinstance(seg, np.ndarray):
                    seg_array = seg
                else:
                    seg_array = sitk.GetArrayFromImage(seg)
                    
                # Convert segmentation to a binary mask (lungs=1, background=0)
                binary_mask = (seg_array > 0).astype(np.uint8)
                
                # Get the original image array (should be 2D for PNG)
                img_array = sitk.GetArrayFromImage(img)
                if img_array.ndim > 2:
                    img_array = img_array[0]  # use first slice if necessary
                    binary_mask = binary_mask[0]  # select corresponding segmentation slice

                # Debug: print unique values of the binary mask
                unique_vals = np.unique(binary_mask)
                print(f"Unique values in binary mask for {input_path}: {unique_vals}")

                # Save the binary mask for diagnostic purposes (optional)
                diagnostic_mask = sitk.GetImageFromArray(binary_mask)
                if diagnostic_mask.GetDimension() != img.GetDimension():
                    origin = img.GetOrigin()
                    spacing = img.GetSpacing()
                    direction = img.GetDirection()
                    # Extract metadata for the 2D slice
                    slice_origin = origin[1:]
                    slice_spacing = spacing[1:]
                    slice_direction = (direction[4], direction[5], direction[7], direction[8])
                    diagnostic_mask.SetOrigin(slice_origin)
                    diagnostic_mask.SetSpacing(slice_spacing)
                    diagnostic_mask.SetDirection(slice_direction)
                else:
                    diagnostic_mask.CopyInformation(img)
                diagnostic_mask_path = output_path.replace('.png', '_mask.png')
                sitk.WriteImage(diagnostic_mask, diagnostic_mask_path)
                print(f"Saved diagnostic mask to {diagnostic_mask_path}")

                # Apply the mask: keep original intensities inside lungs, set others to zero
                lung_only = img_array * binary_mask

                # Convert the masked array back to a SimpleITK image
                out_img = sitk.GetImageFromArray(lung_only)

                # If dimensions match, copy metadata directly; otherwise, manually set 2D metadata.
                if img.GetDimension() == out_img.GetDimension():
                    out_img.CopyInformation(img)
                else:
                    origin = img.GetOrigin()
                    spacing = img.GetSpacing()
                    direction = img.GetDirection()
                    slice_origin = origin[1:]
                    slice_spacing = spacing[1:]
                    slice_direction = (direction[4], direction[5], direction[7], direction[8])
                    out_img.SetOrigin(slice_origin)
                    out_img.SetSpacing(slice_spacing)
                    out_img.SetDirection(slice_direction)

                sitk.WriteImage(out_img, output_path)
                print(f"Segmented: {input_path} -> {output_path}")

                # NOTE:
                # If the binary mask printed unique value [0] for many images, it indicates no lung tissue was detected.
                # Since lungmask is trained on CT images, consider preprocessing (e.g., intensity rescaling) your PNG images
                # to resemble CT intensity ranges.
            except Exception as e:
                print(f"Failed to process {input_path}: {e}")