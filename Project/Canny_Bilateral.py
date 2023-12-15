import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


def segment_logo(image_path):
    # Load the image
    img = cv2.imread(os.path.join('images', image_path))

    # Apply bilateral filtering for smoothing while preserving edges
    smoothed_img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # Convert the smoothed image to grayscale
    gray = cv2.cvtColor(smoothed_img, cv2.COLOR_BGR2GRAY)

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

    # Plotting
    plt.figure(figsize=(10, 8))

    plt.subplot(231), plt.imshow(img[:, :, ::-1])
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(232), plt.imshow(smoothed_img[:, :, ::-1])
    plt.title('Smoothed Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(233), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(234), plt.imshow(cv2.drawContours(
        np.zeros_like(img), contours, -1, (0, 255, 0), 2))
    plt.title('Contours'), plt.xticks([]), plt.yticks([])

    plt.subplot(235), plt.imshow(mask, cmap='gray')
    plt.title('Mask'), plt.xticks([]), plt.yticks([])

    plt.subplot(236), plt.imshow(segmented_logo[:, :, ::-1])
    plt.title('Segmented Logo'), plt.xticks([]), plt.yticks([])

    # Save the plotted image
    output_path = f'Canny_Bilateral_{os.path.splitext(image_path)[0]}_Plots.jpg'
    plt.savefig(output_path, bbox_inches='tight')

    # Save the segmented logo
    output_path_logo = f'Canny_Bilateral_{os.path.splitext(image_path)[0]}_Segmented.jpg'
    # cv2.imwrite(output_path_logo, segmented_logo)

    # Show the plotted image
    # plt.show()


# List of images
image_paths = ['handbag1.jpg', 'handbag2.jpeg', 'handbag3.jpg',
               'handbag4.jpg', 'handbag7.jpg', 'handbag8.jpeg']

# Process each image in the list
for path in image_paths:
    segment_logo(path)
