import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def is_logo_candidate(contour, img):
    min_area = 0.01 * img.shape[0] * img.shape[1]
    max_area = 0.03 * img.shape[0] * img.shape[1]
    area = cv2.contourArea(contour)
    return min_area < area < max_area


def plot_image(image, title, position):
    plt.subplot(1, 6, position)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')


def segment_logo(image_path):
    img = cv2.imread(os.path.join('images', image_path))

    plt.figure(figsize=(15, 10))

    # Plot original image
    plot_image(img, 'Original Image', 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 0    # Brightness control (0-100)
    contrasted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # Plot contrasted image
    plot_image(contrasted, 'Contrasted Image', 2)

    # Apply thresholding
    _, edges = cv2.threshold(
        contrasted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Plot edges
    plot_image(edges, 'Threshold', 3)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    plot_image(cv2.drawContours(
        np.zeros_like(img), contours, -1, (0, 255, 0), 2), 'Contours', 4)

    for i, contour in enumerate(contours):
        if is_logo_candidate(contour, img):
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [contour], 255)

            plot_image(mask, 'Mask', 5)

            segmented_image = np.where(mask[..., None] == 255, img, 0)

            # Save segmented image
            cv2.imwrite(f'CANNY_segmented_{i}.jpg', segmented_image)

            # Plot segmented image
            plot_image(segmented_image, f'Segmented Image', 6)

    # Save the plotted image
    output_path = f'Canny_{os.path.splitext(image_path)[0]}_Plots.jpg'
    plt.savefig(output_path, bbox_inches='tight')

    plt.tight_layout()
    plt.show()


# Example usage
imagePath = ['handbag1.jpg', 'handbag2.jpeg', 'handbag4.jpg']
for path in imagePath:
    segment_logo(path)
