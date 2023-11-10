import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, ifftshift, fftn, ifftn

# Adjusting the previous functions to remove vertical noise

# Function to apply a Butterworth low pass filter for vertical noise
def butterworth_lowpass_filter_vertical(img, cutoff, order=2):
    rows, cols = img.shape
    x, y = np.ogrid[:rows, :cols]
    center_row, center_col = rows / 2, cols / 2
    distance = np.sqrt((x - center_row)**2 + (y - center_col)**2)
    mask = 1 / (1 + (distance / cutoff)**(2*order))
    return mask

# Function to apply a Gaussian low pass filter for vertical noise


def gaussian_lowpass_filter_vertical(img, cutoff):
    rows, cols = img.shape
    x, y = np.ogrid[:rows, :cols]
    center_row, center_col = rows / 2, cols / 2
    distance = np.sqrt((x - center_row)**2 + (y - center_col)**2)
    mask = np.exp(-(distance**2) / (2*(cutoff**2)))
    return mask

# Correct the Notch filter to act as a lowpass filter for vertical noise


def notch_lowpass_filter_vertical(img, cutoff):
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    r = cutoff
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 1
    return mask


# Load the new image with vertical noise
vertical_noise_image_path = 'Noisy_flower1_vertical.jpg'
vertical_noise_image = cv2.imread(vertical_noise_image_path, 0)

# Apply filters with cutoffs 10, 20, ..., 100
cutoffs = list(range(10, 110, 10))
titles = ['Butterworth', 'Gaussian', 'Notch']
butterworth_images = []
gaussian_images = []
notch_filtered_images = []

# Apply filters with cutoffs 10, 20, ..., 100 for the new image with vertical noise
butterworth_images_vertical = [apply_filter(
    vertical_noise_image, butterworth_lowpass_filter_vertical, cutoff) for cutoff in cutoffs]
gaussian_images_vertical = [apply_filter(
    vertical_noise_image, gaussian_lowpass_filter_vertical, cutoff) for cutoff in cutoffs]
notch_images_vertical = [apply_filter(
    vertical_noise_image, notch_lowpass_filter_vertical, cutoff) for cutoff in cutoffs]

# Combine all the filtered images into one figure
fig, axes = plt.subplots(len(cutoffs), len(titles), figsize=(15, 20))

for i, cutoff in enumerate(cutoffs):
    axes[i, 0].imshow(butterworth_images_vertical[i], cmap='gray')
    axes[i, 0].axis('off')
    axes[i, 0].set_title(f'Butterworth cutoff {cutoff}')

    axes[i, 1].imshow(gaussian_images_vertical[i], cmap='gray')
    axes[i, 1].axis('off')
    axes[i, 1].set_title(f'Gaussian cutoff {cutoff}')

    axes[i, 2].imshow(notch_images_vertical[i], cmap='gray')
    axes[i, 2].axis('off')
    axes[i, 2].set_title(f'Notch cutoff {cutoff}')

plt.tight_layout()
# plt.show() is not needed as we are directly saving the file

# Save the combined image
final_vertical_output_path = 'filtered_vertical_combined.jpg'
plt.savefig(final_vertical_output_path, bbox_inches='tight')
plt.close(fig)  # Close the figure to free memory

final_vertical_output_path
