import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def is_logo_candidate(contour, img):
    min_area = 0.01 * img.shape[0] * img.shape[1]
    max_area = 0.02 * img.shape[0] * img.shape[1]
    area = cv2.contourArea(contour)
    return min_area < area < max_area


def plot_image(image, title, position):
    plt.subplot(2, 4, position)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')


def segment_logo(image_path):
    img = cv2.imread(image_path)

    plt.figure(figsize=(16, 8))

    # Plot original image
    plot_image(img, 'Original Image', 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    alpha = 1.1  # Contrast control
    beta = 35   # Brightness control
    contrasted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # Plot contrasted image
    plot_image(contrasted, 'Contrasted Image', 2)

    # Apply Gaussian blur and find edges
    blurred = cv2.GaussianBlur(contrasted, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 450)

    # Plot blurred image and edges
    plot_image(blurred, 'Blurred Image', 3)
    plot_image(edges, 'Edges', 4)

    # Dilate the edges to make them more connected
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Optional: Apply morphological closing to close gaps
    closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel)

    # Plot dilated edges and closed edges
    plot_image(dilated_edges, 'Dilated Edges', 5)
    plot_image(closed_edges, 'Closed Edges', 6)

    # Find contours on the processed edge image
    contours, _ = cv2.findContours(
        closed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        if is_logo_candidate(contour, img):
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, 255, -1)  # Fill the contour

            plot_image(mask, 'Mask', 7)
            
            inverted_mask = cv2.bitwise_not(mask)
            segmented_image = cv2.bitwise_and(img, img, mask=mask)
            outside_area = np.ones_like(img) * 0
            outside_image = cv2.bitwise_and(
                outside_area, outside_area, mask=inverted_mask)
            final_image = cv2.add(segmented_image, outside_image)

            # Save or display the segmented image
            cv2.imwrite(f'Segmented_handbag7.jpg', final_image)

            # Plot segmented image
            plot_image(final_image, f'Segmented Image', 8)

    # Save the plotted image (optional)
    output_path = f'Process_7_Plots.jpg'
    plt.savefig(output_path, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()


# Example usage
imagePath = ['.//images/handbag7.jpg']
for path in imagePath:
    segment_logo(path)
