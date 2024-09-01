from dotenv import load_dotenv
import os
import numpy as np
import matplotlib.pyplot as plt

def print_slice(image_path, depth):
    # Load the .npy file
    image = np.load(image_path)

    # Check if the depth is valid
    if depth < 0 or depth >= image.shape[2]:
        print(f"Invalid depth. Please choose a depth between 0 and {image.shape[2]-1}")
        return

    # Get the slice
    slice = image[:, :, depth]

    # Display the slice
    plt.imshow(slice, cmap='gray')
    plt.show()

def main(image_path, depth):    
    print_slice(image_path, depth)

if __name__ == '__main__':
    load_dotenv()
    root_dir = os.getenv('PATH_TO_MOSMED_DATASET')    
    image_path = f'{root_dir}/images/study_0001/image.npy'
    depth = 21
    
    main(image_path, depth)

# To run this script, use the following command:
# python -m misc.mosmed_print