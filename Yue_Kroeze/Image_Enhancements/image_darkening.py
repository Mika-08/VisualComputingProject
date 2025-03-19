from PIL import Image, ImageEnhance
import os


def reduce_brightness(input_image, output_path, factor):
    """
    Applies brightness reduction to image

    :param input_image: input image
    :param output_path: path to output image
    :param factor: brightness factor
    :return: Nothing
    """
    # Open an image file
    with Image.open(input_image) as img:
        # Enhance the image's brightness
        enhancer = ImageEnhance.Brightness(img)
        img_enhanced = enhancer.enhance(factor)
        # Save the result
        img_enhanced.save(output_path)
        # print(f"Image saved as {output_path}")


def darken_all_images(folder_path, folder_output_path, factor):
    """
    Applies brightness reduction to all images in a folder.

    :param folder_path: Path to the input folder containing images.
    :param folder_output_path: Base path for output folder.
    :param factor: Brightness scaling factor.
    :return: Nothing
    """

    # Ensure the output directory exists
    output_folder = os.path.join(folder_output_path, f"Darkened_{factor}")
    os.makedirs(output_folder, exist_ok=True)

    # Process each image in the folder
    for filename in os.listdir(folder_path):
        input_path = os.path.join(folder_path, filename)
        output_path = os.path.join(output_folder, filename)

        # Check if the file is an image
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            reduce_brightness(input_path, output_path, factor)
        else:
            print(f"Skipping non-image file: {filename}")



input_images = "../Images"
output_images = "../Enhanced_Images/Darker_Images"
factors = [0.1, 0.3, 0.5, 0.7, 0.9]

for factor in factors:
    darken_all_images(input_images, output_images, factor)
    print(f"Done with factor: {factor}")


