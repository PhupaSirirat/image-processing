import cv2
import numpy as np

# Load the original image
image = cv2.imread('assignment2_image1.jpg', cv2.IMREAD_GRAYSCALE)

# Apply global histogram equalization
enhanced_image = cv2.equalizeHist(image)

# Save the enhanced image as 'result_image.jpg'
cv2.imwrite('result_image_global_equalization.jpg', enhanced_image)
