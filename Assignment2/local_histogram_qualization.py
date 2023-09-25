import cv2
import numpy as np


def enhance_image(input_image, output_image, k_values):
    # Load the input image
    img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)

    # Initialize a list to store enhanced images and corresponding labels
    enhanced_images = []
    labels = []

    # Apply local histogram equalization with different neighborhood sizes and k values
    for k in k_values:
        for tile_size in [3, 7, 11]:
            clahe = cv2.createCLAHE(
                clipLimit=k, tileGridSize=(tile_size, tile_size))
            img_eq = clahe.apply(img)
            enhanced_images.append(img_eq)

            # Create label
            label = f'k={k}, Tile Size={tile_size}'
            labels.append(label)

    # Add labels to the images
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    color = (255, 255, 255)  # White text

    for i, label in enumerate(labels):
        img_with_label = np.copy(enhanced_images[i])
        cv2.putText(img_with_label, label, (10, 30), font,
                    font_scale, color, font_thickness)
        enhanced_images[i] = img_with_label

    # Arrange the images in 3 columns
    num_columns = 3
    combined_img = np.vstack([np.hstack(enhanced_images[i:i+num_columns])
                             for i in range(0, len(enhanced_images), num_columns)])

    # Save the combined image
    cv2.imwrite(output_image + '_combined.jpg', combined_img)


if __name__ == '__main__':
    input_image = 'assignment2_image1.jpg'
    output_image = 'enhanced_image'
    k_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                8.0, 9.0, 10.0]  # List of different k values

    enhance_image(input_image, output_image, k_values)
