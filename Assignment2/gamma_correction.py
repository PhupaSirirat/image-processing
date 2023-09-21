import cv2
import numpy as np

def enhance_image(image_path, gamma):
    # Load the image
    image = cv2.imread(image_path)

    # Apply gamma correction to the entire image
    enhanced_image = np.power(image.astype(np.float32) / 255.0, gamma)
    enhanced_image = (enhanced_image * 255).astype(np.uint8)

    return enhanced_image

# Example usage
input_image_path = 'assignment2_image1.jpg'
result_image = enhance_image(input_image_path, gamma=0.5)

# Save the result
cv2.imwrite('result_image_gamma_correcting.jpg', result_image)
