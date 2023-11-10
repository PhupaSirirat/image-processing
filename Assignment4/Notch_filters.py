import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to generate a low-pass or high-pass filter mask


def create_filter_mask(rows, cols, radius, filter_type):
    crow, ccol = rows // 2, cols // 2  # center
    mask = np.zeros((rows, cols), np.uint8)

    for u in range(rows):
        for v in range(cols):
            d = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)
            if filter_type == 'low' and d <= radius:
                mask[u, v] = 1
            elif filter_type == 'high' and d >= radius:
                mask[u, v] = 1
    return mask

# Function to apply notch filter and save the output image


def apply_notch_filter(image_path, filter_type, radius):
    # Read the image in grayscale
    image = cv2.imread(image_path, 0)

    # Fourier transform
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # Create mask and apply it
    rows, cols = image.shape
    mask = create_filter_mask(rows, cols, radius, filter_type)
    fshift_filtered = fshift * mask

    # Inverse FFT to get the image back
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Normalize the image for display
    img_back_normalized = cv2.normalize(
        img_back, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    # Plot the mask, the frequency domain, and the resulting image
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    axs[0].imshow(mask, cmap='gray')
    axs[0].set_title('Filter Mask')
    axs[0].axis('off')

    axs[1].imshow(20 * np.log(np.abs(fshift_filtered)), cmap='gray')
    axs[1].set_title('Filtered Frequency Domain')
    axs[1].axis('off')

    axs[2].imshow(img_back_normalized, cmap='gray')
    axs[2].set_title(
        f'Resulting Image\n{filter_type.capitalize()} pass filter r={radius}')
    axs[2].axis('off')

    plt.tight_layout(pad=1.0)

    # Save the plot as an image file
    output_file_name = f"{image_path.rstrip('.jpg')}_{filter_type}_r{radius}.png"
    plt.savefig(output_file_name, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)  # Close the figure to free memory

    print(f"Saved {output_file_name}")


# Run the filtering process on uploaded images
image_paths = ['flower1.jpg', 'fruit.jpg']  # Replace with actual paths
filter_types = ['low', 'high']
radii = [10, 50, 100]

for image_path in image_paths:
    for radius in radii:
        for filter_type in filter_types:
            apply_notch_filter(image_path, filter_type, radius)
