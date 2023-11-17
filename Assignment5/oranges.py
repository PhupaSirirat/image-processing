from skimage import io
import numpy as np

# Load the image
image = io.imread('oranges.jpg')

# Define the function to segment the orange color with a fine-tuned range
def segment_orange_finetuned(image):
    # Convert the image to float and normalize the color values to be between 0 and 1
    image = image / 255.0
    
    # Fine-tuning the lower and upper bounds for the orange color to include a narrower range
    # The bounds are in (R,G,B) format
    lower = np.array([0.7, 0.3, 0.0]) # fine-tuned lower bound of orange color
    upper = np.array([1.0, 0.7, 0.3]) # fine-tuned upper bound of orange color
    
    # Create a mask for the orange color
    mask = np.all(image >= lower, axis=-1) & np.all(image <= upper, axis=-1)
    
    # Copy the original image
    segmented_image = image.copy()
    
    # Change non-orange fruits to blue color
    # Blue color in (R,G,B) format is (0, 0, 0.5)
    blue_color = np.array([0, 0, 0.5])
    segmented_image[~mask] = blue_color
    
    # Convert back to original scale
    segmented_image *= 255
    segmented_image = segmented_image.astype(np.uint8)
    
    return segmented_image

# Apply the fine-tuned segmentation
segmented_image_finetuned = segment_orange_finetuned(image)

# Save the segmented image
output_path = 'segmented_oranges.jpg'
io.imsave(output_path, segmented_image_finetuned)
