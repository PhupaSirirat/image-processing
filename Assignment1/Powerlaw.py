import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Define image processing functions

def read_and_convert_to_gray(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def power_law_transform(image, c, gamma):
    image_float = image.astype(np.float32)
    image_transformed = c * (image_float ** gamma)
    image_normalize = (image_transformed - np.min(image_transformed)) / \
        (np.max(image_transformed) - np.min(image_transformed)) * 255
    gamma_corrected_img = image_normalize.astype(np.uint8)
    return gamma_corrected_img

# Define a function to save and display images in a grid

def save_and_display_images(image_list, c_values, gamma_values, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    rows, cols = len(image_list), len(c_values) * len(gamma_values) + 1
    plt.figure(figsize=(cols * 3, rows * 3))

    for i, image_path in enumerate(image_list):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image = read_and_convert_to_gray(image_path)

        plt.subplot(rows, cols, i * cols + 1)
        original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        plt.imshow(original_image)
        plt.axis('off')
        plt.title(f"{image_name}\nOriginal")

        for j, c in enumerate(c_values):
            for k, gamma in enumerate(gamma_values):
                subplot_index = i * cols + j * len(gamma_values) + k + 2
                transformed_image = power_law_transform(image, c, gamma)
                plt.subplot(rows, cols, subplot_index)
                plt.imshow(transformed_image, cmap='gray')
                plt.axis('off')
                plt.title(f"c={c}, gamma={gamma}")

                # Save the transformed image
                output_path = os.path.join(output_folder, f"{image_name}-{c}-{gamma}.jpg")
                cv2.imwrite(output_path, transformed_image)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'output_figure.png'), bbox_inches='tight')
    plt.show()

# Define image list and processing parameters

image_list = ["images/cartoon.jpg", "images/scenery1.jpg", "images/scenery2.jpg"]
c_values = [0.5, 1, 2]
gamma_values = [0.4, 2.5]

# Define output folder
output_folder = 'powerlaw'

# Process and save images

save_and_display_images(image_list, c_values, gamma_values, output_folder)
