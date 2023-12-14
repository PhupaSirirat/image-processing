import cv2
import numpy as np
from matplotlib import pyplot as plt


def segment_logo(image_path):
    # Load the image
    img = cv2.imread('./images/' + image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    # Adjust parameters as needed
    _, edges = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Assuming the logo is the most prominent feature, find contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour which might be the logo
    # This is a simplification, in real scenarios you might need a more robust method
    logo_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the logo
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, [logo_contour], 255)

    # Segment the logo
    # 255 for white background
    segmented_logo = np.where(mask[..., None] == 255, img, 255)

    # Plotting
    plt.figure(figsize=(12, 8))

    plt.subplot(231), plt.imshow(img[:, :, ::-1])
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(232), plt.imshow(gray, cmap='gray')
    plt.title('Grayscale Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(233), plt.imshow(edges, cmap='gray')
    plt.title('Adaptive Thresholding'), plt.xticks([]), plt.yticks([])

    plt.subplot(234), plt.imshow(cv2.drawContours(
        np.zeros_like(img), contours, -1, (0, 255, 0), 2))
    plt.title('Contours'), plt.xticks([]), plt.yticks([])

    plt.subplot(235), plt.imshow(mask, cmap='gray')
    plt.title('Mask'), plt.xticks([]), plt.yticks([])

    plt.subplot(236), plt.imshow(segmented_logo[:, :, ::-1])
    plt.title('Segmented Logo'), plt.xticks([]), plt.yticks([])

    # Save the plotted image
    output_path = 'Threshold_' + image_path.split('.')[0] + 'Plots.jpg'
    plt.savefig(output_path, bbox_inches='tight')

    # Show the plotted image
    # plt.show()

    # Save the segmented logo
    output_path_logo = 'Threshold_' + \
        image_path.split('.')[0] + 'Segmented.jpg'
    # cv2.imwrite(output_path_logo, segmented_logo)


# List of images
imagePath = ['handbag1.jpg', 'handbag2.jpeg', 'handbag3.jpg',
             'handbag4.jpg', 'handbag7.jpg', 'handbag8.jpeg']

# Process each image in the list
for path in imagePath:
    segment_logo(path)
