import numpy as np

# Define a manual function to convert an image to its complementary colors
def convert_to_complementary(image):
    # Convert the image to a numpy array
    data = np.array(image)
    # Calculate the complementary color (255 - color value)
    complementary_data = 255 - data
    # Convert back to an image
    complementary_image = Image.fromarray(complementary_data.astype('uint8'))
    return complementary_image

# Apply the manual complementary color conversion function
complementary_image_manual = convert_to_complementary("fruit.jpg")

# Save the manually converted complementary image
complementary_image_manual_path = 'complementary_fruit.jpg'
complementary_image_manual.save(complementary_image_manual_path)

# Display the manually converted complementary image
complementary_image_manual.show()
