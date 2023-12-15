import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def is_logo_candidate(contour, img):
    min_area = 0.1 * img.shape[0] * img.shape[1]
    max_area = 0.4 * img.shape[0] * img.shape[1]
    area = cv2.contourArea(contour)
    return min_area < area < max_area


def plot_image(image, title, position):
    plt.subplot(2, 4, position)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    
def segment_logo(image_path):
    img = cv2.imread(os.path.join('images', image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    plt.figure(figsize=(16, 8))
    plot_image(img, 'Original Image', 1)

    # Increase contrast
    alpha = 1.0  # Contrast control
    beta = 50    # Brightness control
    contrasted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    plot_image(contrasted, 'Contrasted Image', 2)

    # Apply Gaussian blur and find edges
    blurred = cv2.GaussianBlur(contrasted, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 200)
    
    plot_image(blurred, 'GaussianBlur', 3)
    plot_image(edges, 'Canny Edges', 4)

    # Dilate the edges to make them more connected
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Optional: Apply morphological closing to close gaps
    closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel)
    
    plot_image(dilated_edges, 'Dilated Edges', 5)
    plot_image(closed_edges, 'Closed Edges', 6)

    # Find contours on the processed edge image
    contours, _ = cv2.findContours(
        closed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    

    for i, contour in enumerate(contours):
        # Create a mask from the contour
        if is_logo_candidate(contour, img):
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, 255, -1)  # Fill the contour
            
            plot_image(mask, 'Mask', 7)

            # Create an inverted mask for the area outside the contour
            inverted_mask = cv2.bitwise_not(mask)

            # Apply the original colors inside the contour and set outside to white
            segmented_image = cv2.bitwise_and(img, img, mask=mask)
            
            plot_image(segmented_image, f'Segmented Image', 8)
            
            outside_area = np.ones_like(img) * 0  # Create a white image
            outside_image = cv2.bitwise_and(
                outside_area, outside_area, mask=inverted_mask)
            final_image = cv2.add(segmented_image, outside_image)

            # Save or display the segmented image
            image_name = image_path.split('/')[-1].split('.')[0]
            image_file_type = image_path.split('/')[-1].split('.')[1]
            cv2.imwrite(
                f'Handbag3_segmented.{image_file_type}', final_image)
    # Save the plotted image
    output_path = f'Process_3_{os.path.splitext(image_path)[0]}_Plots.jpg'
    plt.savefig(output_path, bbox_inches='tight')

    plt.tight_layout()
    plt.show()

# Example usage
imagePath = ['handbag3.jpg']
for path in imagePath:
    segment_logo(path)
