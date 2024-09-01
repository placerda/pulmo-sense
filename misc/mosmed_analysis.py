from dotenv import load_dotenv
import numpy as np
import os
from utils.log_config import get_custom_logger

my_logger = get_custom_logger('mosmed_analysis')

def get_stats_and_slice(path, counter=1):
    # Load the .npy file
    image = np.load(path)

    # Get the dimensions
    width, length, depth = image.shape

    # Check if the width and height are not 512
    if width != 512 or length != 512:
        my_logger.info(f"{counter}. Image at {path} has dimensions different from 512x512: {width}x{length}")
        is_not_512 = True
    else:
        is_not_512 = False

    # Get voxel statistics
    median = np.median(image)
    average = np.mean(image)

    # Prepare the output string
    output = f"{counter}. {path}: Width: {width}, Length: {length}, Depth: {depth}, Median: {median}, Average: {average}"

    my_logger.info(output)

    # Return the dimensions and whether the image is not 512x512
    return width, length, depth, is_not_512

def main(folder_path):

    # Lists to store the dimensions of all images
    widths = []
    lengths = []
    depths = []
    not_512_images = []

    counter = 1
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == "image.npy":
                file_path = os.path.join(root, file)
                width, length, depth, is_not_512 = get_stats_and_slice(file_path, counter)
                widths.append(width)
                lengths.append(length)
                depths.append(depth)
                if is_not_512:
                    not_512_images.append(file_path)
                counter += 1
                break

    # Calculate and print the overall statistics
    my_logger.info(f"Overall Width - Mean: {np.mean(widths)}, Median: {np.median(widths)}, Min: {np.min(widths)}, Max: {np.max(widths)}")
    my_logger.info(f"Overall Length - Mean: {np.mean(lengths)}, Median: {np.median(lengths)}, Min: {np.min(lengths)}, Max: {np.max(lengths)}")
    my_logger.info(f"Overall Depth - Mean: {np.mean(depths)}, Median: {np.median(depths)}, Min: {np.min(depths)}, Max: {np.max(depths)}")

    # Print the images that are not 512x512
    my_logger.info("\nImages with dimensions different from 512x512:")
    for i, path in enumerate(not_512_images, 1):
        my_logger.info(f"{i}. {path}")

if __name__ == '__main__':
    load_dotenv()
    root_dir = os.getenv('PATH_TO_MOSMED_DATASET')    
    folder_path = f'{root_dir}/images'
    main(folder_path)

# To run this script, use the following command:
# python -m misc.mosmed_analysis

# Overall Width - Mean: 512.0, Median: 512.0, Min: 512, Max: 512
# Overall Length - Mean: 512.0, Median: 512.0, Min: 512, Max: 512
# Overall Depth - Mean: 41.81171171171171, Median: 41.0, Min: 31, Max: 72