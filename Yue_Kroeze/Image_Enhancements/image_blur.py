import cv2
import numpy as np


def apply_motion_blur(image, intensity, angle, output_path):
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
    intensity = max(1, intensity)

    # Create motion blur kernel
    size = intensity
    kernel = np.zeros((size, size))
    kernel[int((size - 1) / 2), :] = np.ones(size)
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0), (size, size))
    kernel = kernel / np.sum(kernel)  # Normalize

    # Apply the kernel to the image
    blurred = cv2.filter2D(image, -1, kernel)

    cv2.imwrite(output_path, blurred)


# Example usage
image = cv2.imread("../Images/099548361.jpg")
output_path = "../Enhanced_Images/Blurred_Images/099548361.jpg"
apply_motion_blur(image, 15, 45, output_path)
