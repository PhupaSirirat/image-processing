import cv2
import numpy as np
from matplotlib import pyplot as plt

def segment_logo(image_path):
    # Load the image
    img = cv2.imread('./images/' + image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply edge detection or any other feature detection method
    edges = cv2.Canny(gray, 100, 200)

    # Assuming the logo is the most prominent feature, find contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour which might be the logo
    # This is a simplification, in real scenarios you might need a more robust method
    logo_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the logo
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [logo_contour], -1, 255, thickness=cv2.FILLED)

    # Segment the logo
    # 255 for white background
    segmented_logo = np.where(mask[..., None] == 255, img, 255)

    # Save the segmented logo
    output_path = 'Canny_' + image_path.split('.')[0] + 'Segmented.jpg'
    cv2.imwrite(output_path, segmented_logo)
    print(f'Segmented logo saved as: {output_path}')
    
    plt.subplot(121), plt.imshow(segmented_logo, cmap='gray')
    plt.title('Segmented Logo'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    
    plt.show()


# List of images
imagePath = ['handbag1.jpg', 'handbag2.jpeg', 'handbag3.jpg', 'handbag4.jpg', 'handbag7.jpg', 'handbag8.jpeg']

# Process each image in the list
for path in imagePath:
    segment_logo(path)
