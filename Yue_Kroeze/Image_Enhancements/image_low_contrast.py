import cv2
import numpy as np
import os

def reduce_contrast(input_path, scale, output_path):
    """
    Reduces contrast of an image.

    Parameters:
    - image: Input image (numpy array).
    - scale: Contrast reduction factor (1.0 = original, 0.0 = gray).

    Returns:
    - Image with reduced contrast.
    """
    input_image = cv2.imread(input_path)

    scale = np.clip(scale, 0.0, 1.0)  # Ensure scale is in range [0,1]

    # Convert image to float and normalize
    gray = np.mean(input_image, axis=(0, 1))  # Compute global average color
    low_contrast = gray + scale * (input_image - gray)  # Interpolate between gray and original

    output = np.uint8(low_contrast)
    cv2.imwrite(output_path, output)


def reduce_contrast_all_images(folder_path, folder_output_path, factor):
    """
    Applies brightness reduction to all images in a folder.

    :param folder_path: Path to the input folder containing images.
    :param folder_output_path: Base path for output folder.
    :param factor: Brightness scaling factor.
    :return: Nothing
    """

    # Ensure the output directory exists
    output_folder = os.path.join(folder_output_path, f"low_{factor}")
    os.makedirs(output_folder, exist_ok=True)

    # Process each image in the folder
    for filename in os.listdir(folder_path):
        input_path = os.path.join(folder_path, filename)
        output_path = os.path.join(output_folder, filename)

        # Check if the file is an image
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            reduce_contrast(input_path, factor, output_path)
        else:
            print(f"Skipping non-image file: {filename}")


input_images = "../Images"
output_images = "../Enhanced_Images/Low_Contrast_Images"
factors = [0.1, 0.3, 0.5, 0.7, 0.9]

for factor in factors:
    reduce_contrast_all_images(input_images, output_images, factor)
    print(f"Done with factor: {factor}")


