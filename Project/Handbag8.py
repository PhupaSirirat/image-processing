import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def is_logo_candidate(contour, img):
    min_area = 0.02 * img.shape[0] * img.shape[1]
    max_area = 0.05 * img.shape[0] * img.shape[1]
    area = cv2.contourArea(contour)
    return min_area < area < max_area


def plot_image(image, title, position):
    plt.subplot(1, 6, position)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')

def segment_logo(image_path):
    img = cv2.imread(os.path.join('images', image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    plt.figure(figsize=(16, 8))
    
    plot_image(img, 'Original Image', 1)

    # Increase contrast
    alpha = 1  # Contrast control (1.0-3.0)
    beta = 0    # Brightness control (0-100)
    contrasted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    
    plot_image(contrasted, 'Contrasted Image', 2)

    # Then apply thresholding
    _, edges = cv2.threshold(
        contrasted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    plot_image(edges, 'Threshold', 3)
    plot_image(cv2.drawContours(
        np.zeros_like(img), contours, -1, (0, 255, 0), 2), 'Contours', 4)

    for i, contour in enumerate(contours):
        if is_logo_candidate(contour, img):
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [contour], 255)
            plot_image(mask, 'Mask', 5)
            
            segmented_image = np.where(mask[..., None] == 255, img, 0)
            plot_image(segmented_image, 'Segmented image', 6)

            # Optionally, save each segmented image
            image_name = image_path.split('/')[-1].split('.')[0]
            image_file_type = image_path.split('/')[-1].split('.')[1]
            cv2.imwrite(
                f'IMG8_segmented_{i}.{image_file_type}', segmented_image)
            # Save the plotted image (optional)
            output_path = f'Process_8_Plots_{i}.jpg'
            plt.savefig(output_path, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()


# Example usage
imagePath = ['handbag8.jpeg']
for path in imagePath:
    segment_logo(path)
