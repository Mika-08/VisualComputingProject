import cv2
import numpy as np
import os
from PIL import Image



def apply_motion_blur(input_image, intensity, angle, output_path):
    """
    Applies motion blur to an image.

    Parameters:
    - image: Input image (numpy array).
    - intensity: Blur strength (higher = more blur).
    - angle: Direction of motion blur in degrees (0 = horizontal, 90 = vertical).

    Returns:
    - Blurred image.
    """
    # Ensure intensity is at most 1
    # img = cv2.imread(input_image)
    intensity = max(1, intensity)

    # Create motion blur kernel
    size = intensity
    kernel = np.zeros((size, size))
    kernel[int((size - 1) / 2), :] = np.ones(size)
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0), (size, size))
    kernel = kernel / np.sum(kernel)  # Normalize

    # Apply the kernel to the image
    blurred = cv2.filter2D(input_image, -1, kernel)

    cv2.imwrite(output_path, blurred)

    # return blurred



def blur_all_images(folder_path, folder_output_path, factor, horizontal):
    """
    Applies blur reduction to all images in a folder.

    :param folder_path: Path to the input folder containing images.
    :param folder_output_path: Base path for output folder.
    :param factor: Blur scaling factor.
    :return: Nothing
    """

    # Ensure the output directory exists
    output_folder = os.path.join(folder_output_path, f"Blur_{factor}")
    os.makedirs(output_folder, exist_ok=True)

    # Process each image in the folder
    for filename in os.listdir(folder_path):
        input_path = os.path.join(folder_path, filename)
        output_path = os.path.join(output_folder, filename)
        image = cv2.imread(input_path)

        # Check if the file is an image
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            if horizontal:
                apply_motion_blur(image, factor, 0, output_path)
            else:
                apply_motion_blur(image, factor, 90, output_path)
        else:
            print(f"Skipping non-image file: {filename}")


input_images = "../Images"
output_images_horizontal = "../Enhanced_Images/Blurred_Images/Horizontal"
factors = [10, 30, 50, 70, 90]

for factor in factors:
    blur_all_images(input_images, output_images_horizontal, factor, True)
    print(f"Done with factor: {factor}")

output_images_vertical = "../Enhanced_Images/Blurred_Images/Vertical"
for factor in factors:
    blur_all_images(input_images, output_images_vertical, factor, False)
    print(f"Done with factor: {factor}")