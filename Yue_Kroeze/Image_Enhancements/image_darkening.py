from PIL import Image, ImageEnhance

def reduce_brightness(image_path, output_path, factor):
    # Open an image file
    with Image.open(image_path) as img:
        # Enhance the image's brightness
        enhancer = ImageEnhance.Brightness(img)
        img_enhanced = enhancer.enhance(factor)
        # Save the result
        img_enhanced.save(output_path)
        print(f"Image saved as {output_path}")

# Example usage:
image_path = '../Images/099548361.jpg'
output_path = '../Enhanced_Images/Darker_Images/099548361.jpg.jpg'
brightness_factor = 0.5

reduce_brightness(image_path, output_path, brightness_factor)
