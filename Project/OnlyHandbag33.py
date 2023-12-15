import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def segment_logo(image_path):
    # Load the image
    img = cv2.imread(os.path.join('images', image_path))

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours using external retrieval mode
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set minimum and maximum logo area thresholds
    min_logo_area = 500  # Adjust as needed
    max_logo_area = 5000  # Adjust as needed

    # Extract contours within the area range
    logo_contours = [contour for contour in contours if min_logo_area <
                     cv2.contourArea(contour) < max_logo_area]

    # Find the contour with the maximum area
    if logo_contours:
        logo_contour = max(logo_contours, key=cv2.contourArea)

        # Apply convex hull to the logo contour
        hull = cv2.convexHull(logo_contour)

        # Create a mask for the logo
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [hull], -1, 255, thickness=cv2.FILLED)

        # Segment the logo
        segmented_logo = cv2.bitwise_and(img, img, mask=mask)

        # Plotting (optional)
        plt.figure(figsize=(10, 8))
        plt.subplot(231), plt.imshow(img[:, :, ::-1])
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(232), plt.imshow(blurred, cmap='gray')
        plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(233), plt.imshow(edges, cmap='gray')
        plt.title('Canny Edge Detection'), plt.xticks([]), plt.yticks([])
        plt.subplot(234), plt.imshow(cv2.drawContours(
            np.zeros_like(img), contours, -1, (0, 255, 0), 2))
        plt.title('Contours'), plt.xticks([]), plt.yticks([])
        plt.subplot(235), plt.imshow(mask, cmap='gray')
        plt.title('Mask'), plt.xticks([]), plt.yticks([])
        plt.subplot(236), plt.imshow(segmented_logo[:, :, ::-1])
        plt.title('Segmented Logo'), plt.xticks([]), plt.yticks([])

        # Save the plotted image (optional)
        output_path = f'ADJUSTED_{os.path.splitext(image_path)[0]}_Plots.jpg'
        plt.savefig(output_path, bbox_inches='tight')

        # Save the segmented logo
        output_path_logo = f'Canny_Contours_{os.path.splitext(image_path)[0]}_Segmented.jpg'
        cv2.imwrite(output_path_logo, segmented_logo)

        # Show the plotted image (optional)
        plt.show()

# List of images
image_paths = ['handbag3.jpg']

# Process each image in the list
for path in image_paths:
    segment_logo(path)
