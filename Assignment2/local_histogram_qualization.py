import cv2
import numpy as np


def enhance_image(input_image_path):
    # Load the image
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    # Define neighborhood sizes
    neighborhood_sizes = [3, 7, 11]

    # Apply local histogram equalization for each neighborhood size
    for size in neighborhood_sizes:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(size, size))
        enhanced_img = clahe.apply(img)

        # Concatenate the original and enhanced images for comparison
        result = np.hstack((img, enhanced_img))

        # Show and save the result image
        cv2.imwrite(f'result_image_{size}x{size}.jpg', enhanced_img)



# Specify the input image path
input_image_path = 'assignment2_image1.jpg'

# Call the function to enhance the image
enhance_image(input_image_path)
