import os
import json
import numpy as np
import cv2
from scipy.ndimage import zoom

# Constants for resampling
NEW_SPACING_XY = 0.6
NEW_SPACING_Z = 8

def normalize(image):
    """Normalize the image to zero mean and unit std."""
    return (image - image.mean()) / image.std()

def resample(image, old_spacing, new_spacing=np.array([NEW_SPACING_XY, NEW_SPACING_XY, NEW_SPACING_Z])):
    """
    Resample the 3D image to new spacing.
    
    Parameters:
        image: numpy array of shape (slices, height, width)
        old_spacing: array-like with 3 values [x, y, z]
        new_spacing: desired spacing [NEW_SPACING_XY, NEW_SPACING_XY, NEW_SPACING_Z]
    Returns:
        resampled_image: numpy array after zooming
    """
    # Calculate the resize factor: old_spacing/new_spacing
    resize_factor = old_spacing / new_spacing
    new_shape = image.shape * resize_factor
    rounded_new_shape = np.round(new_shape).astype(int)
    resize_factor_actual = rounded_new_shape / image.shape
    # resample using zoom
    resampled_image = zoom(image, resize_factor_actual, mode='nearest')
    return resampled_image

def process_study(study_name, images_root, labels_root, output_root):
    """
    Process a single study:
     - Load the npy image and spacing.
     - Read the covid_label.json to decide if it's NCP (True) or Normal (False).
     - Resample, normalize, select 30 central slices, and resize each slice to 512x512.
     - Convert slices to uint8 by scaling intensity per slice.
     - Save each slice as a PNG file into the output folder with the ccccii folder structure.
    """
    # Define input paths
    study_img_dir = os.path.join(images_root, study_name)
    image_path = os.path.join(study_img_dir, "image.npy")
    spacing_path = os.path.join(study_img_dir, "spacing.json")
    
    label_dir = os.path.join(labels_root, study_name)
    label_path = os.path.join(label_dir, "covid_label.json")
    
    # Load image and spacing data
    image = np.load(image_path)
    print("Original image shape:", image.shape)

    if image.ndim == 3 and image.shape[2] < image.shape[0]:
        # Transpose so that the depth (D) becomes the first dimension
        image = np.transpose(image, (2, 0, 1))
        print("Transposed image shape:", image.shape)

    with open(spacing_path, 'r') as f:
        spacing = np.array(json.load(f))
    
    # Load label (expecting True for NCP, False for Normal)
    with open(label_path, 'r') as f:
        label_data = json.load(f)
    # In mosmed, label is stored as a boolean in the json file.
    is_covid = label_data if isinstance(label_data, bool) else label_data.get('covid', False)
    
    # Determine output label folder (mosmed has only NCP and Normal)
    label_folder = "NCP" if is_covid else "Normal"
    
    # Resample image
    image = resample(image, spacing)
    # Normalize image (resulting values may not be in [0, 255])
    image = normalize(image)
    
    # Select 30 central slices (if available)
    num_slices = image.shape[0]
    if num_slices < 30:
        # If fewer than 30 slices, use them all
        selected_image = image
    else:
        z_center = num_slices // 2
        start = z_center - 15
        end = z_center + 15
        selected_image = image[start:end, :, :]
    
    # Resize each slice to 512x512
    num_selected = selected_image.shape[0]
    resized_slices = np.zeros((num_selected, 512, 512), dtype=np.float32)
    # Compute zoom factors for height and width based on current slice shape
    current_h, current_w = selected_image.shape[1], selected_image.shape[2]
    zoom_factors = (512 / current_h, 512 / current_w)
    for i in range(num_selected):
        resized_slices[i] = zoom(selected_image[i], zoom_factors, mode='nearest')
    
    # Convert each slice to uint8 by scaling intensities to [0, 255]
    slices_uint8 = []
    for i in range(num_selected):
        slice_img = resized_slices[i]
        # Scale each slice individually
        min_val = slice_img.min()
        max_val = slice_img.max()
        if max_val - min_val > 0:
            slice_scaled = (slice_img - min_val) / (max_val - min_val)
        else:
            slice_scaled = slice_img - min_val
        slice_uint8 = (slice_scaled * 255).astype(np.uint8)
        slices_uint8.append(slice_uint8)
    slices_uint8 = np.array(slices_uint8)
    
    # Build output folder structure:
    # mosmed_png/<label_folder>/<study_name>/<study_name>/ 
    # (using the study name as both patient id and scan id)
    out_dir = os.path.join(output_root, label_folder, study_name, study_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Save each slice as a PNG file with sequential names (e.g., 0000.png)
    for i, slice_img in enumerate(slices_uint8):
        out_path = os.path.join(out_dir, f"{i:04d}.png")
        if os.path.exists(out_path):
            print(f"Skipping {out_path} as it already exists.")
            continue
        cv2.imwrite(out_path, slice_img)
    print(f"Processed {study_name}: {num_selected} slices processed and saved in {out_dir}")

def process_mosmed_dataset(input_root, output_root):
    """
    Process the entire mosmed dataset.
    Expects the following structure in input_root:
       - images/
             study_XXXX/ (each contains image.npy and spacing.json)
       - covid_labels/ (or labels/) 
             study_XXXX/ (each contains covid_label.json)
    The output_root (e.g., "mosmed_png") will contain the new structure.
    """
    # Define folders (adjust if your labels folder is named differently)
    images_folder = os.path.join(input_root, "images")
    labels_folder = os.path.join(input_root, "covid_labels")  # or "labels"
    
    # Get list of studies
    studies = sorted(os.listdir(images_folder))
    
    for study in studies:
        process_study(study, images_folder, labels_folder, output_root)

if __name__ == "__main__":
    # Example usage:
    # Set the path to the mosmed dataset root folder (which contains 'images' and 'covid_labels')
    input_dataset = "data/mosmed/"   # <-- change this to your mosmed dataset folder
    # Set the output folder for PNG dataset in ccccii format
    output_dataset = "data/mosmed_png"
    
    process_mosmed_dataset(input_dataset, output_dataset)
