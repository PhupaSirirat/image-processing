import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Define the input and output folders
input_folder = 'images'
output_folder = 'grayscale'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define a list of image filenames
image_filenames = ["flower.jpg", "fractal.jpeg", "fruit.jpg"]

# Function to quantize an image to a specific gray level and save it
def quantize_and_save_image(input_path, output_folder, gray_level):
    # Load the image
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Quantize the image to the specified gray level
    quantized_image = (gray_level / 256.0 * gray_image).astype(np.uint8)

    # Generate the output path
    output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_path))[0]}_{gray_level}.jpg")

    # Save the quantized image
    cv2.imwrite(output_path, quantized_image)

# Define the list of gray levels to test
gray_levels = [8, 64, 128, 256]

# Process and save each image for each gray level
for filename in image_filenames:
    input_path = os.path.join(input_folder, filename)
    fig, axes = plt.subplots(1, len(gray_levels) + 1, figsize=(15, 5))

    for i, gray_level in enumerate(gray_levels):
        quantize_and_save_image(input_path, output_folder, gray_level)
        quantized_image = cv2.imread(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_{gray_level}.jpg"), cv2.IMREAD_GRAYSCALE)
        axes[i].imshow(quantized_image, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"Gray Level {gray_level}")

    # Add the original image to the figure
    original_image = cv2.imread(input_path)
    axes[len(gray_levels)].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[len(gray_levels)].axis('off')
    axes[len(gray_levels)].set_title("Original Image")

    # Save the individual figure for this image
    plt.tight_layout()
    figure_filename = os.path.splitext(filename)[0] + "_figure.jpg"
    figure_path = os.path.join(output_folder, figure_filename)
    plt.savefig(figure_path)
    plt.close()

# Display the combined figure for all images
plt.figure(figsize=(15, 5))
for i, image_filename in enumerate(image_filenames):
    image = cv2.imread(os.path.join(output_folder, f"{os.path.splitext(image_filename)[0]}_{gray_levels[-1]}.jpg"), cv2.IMREAD_GRAYSCALE)
    plt.subplot(1, len(image_filenames), i + 1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(f"Gray Levels for {image_filename}")

plt.tight_layout()
combined_figure_filename = "combined_figure.jpg"
combined_figure_path = os.path.join(output_folder, combined_figure_filename)
plt.savefig(combined_figure_path)
plt.show()
