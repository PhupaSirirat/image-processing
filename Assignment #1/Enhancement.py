import cv2
import matplotlib.pyplot as plt
import os

# Function to apply enhancement to the image and save it
def apply_enhancement_and_save(input_folder, image_filename, output_folder):
    # Load the image
    image = cv2.imread(os.path.join(input_folder, image_filename))
    
    # Convert the image to grayscale
    grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    L = 256

    # Enhance the image
    for i in range(len(grayscale_img)):
        for j in range(len(grayscale_img[0])):
            if grayscale_img[i][j] <= L / 3:
                grayscale_img[i][j] = 5 * L / 6
            elif grayscale_img[i][j] <= 2 * L / 3:
                grayscale_img[i][j] = (-2 * grayscale_img[i][j]) + 384
            else:
                grayscale_img[i][j] = L / 6

    # Generate the output path
    output_path = os.path.join(output_folder, image_filename)

    # Save the enhanced image
    cv2.imwrite(output_path, grayscale_img)

    return grayscale_img

# Define the input and output folders
input_folder = 'images'
output_folder = 'enhancement'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List of image filenames to process
image_filenames = ["flower.jpg", "traffic.jpg", "tram.jpg"]

# Process and display each image
for image_filename in image_filenames:
    # Apply enhancement and save the image
    enhanced_image = apply_enhancement_and_save(input_folder, image_filename, output_folder)

    # Create a subplot for original and enhanced images
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot the original image
    original_image = cv2.imread(os.path.join(input_folder, image_filename))
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(original_image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Plot the enhanced image
    enhanced_image = cv2.imread(os.path.join(output_folder, image_filename), cv2.IMREAD_GRAYSCALE)
    axes[1].imshow(enhanced_image, cmap='gray')
    axes[1].set_title("Enhanced Image")
    axes[1].axis('off')

    plt.show()
