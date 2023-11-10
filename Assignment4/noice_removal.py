import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, ifftshift, fftn, ifftn

# Function to apply a Butterworth low pass filter


def butterworth_lowpass_filter(img, cutoff, order=2):
    rows, cols = img.shape
    x, y = np.ogrid[:rows, :cols]
    center_row, center_col = rows / 2, cols / 2
    distance = np.sqrt((x - center_row)**2 + (y - center_col)**2)
    mask = 1 / (1 + (distance / cutoff)**(2*order))
    return mask

# Function to apply a Gaussian low pass filter


def gaussian_lowpass_filter(img, cutoff):
    rows, cols = img.shape
    x, y = np.ogrid[:rows, :cols]
    center_row, center_col = rows / 2, cols / 2
    distance = np.sqrt((x - center_row)**2 + (y - center_col)**2)
    mask = np.exp(-(distance**2) / (2*(cutoff**2)))
    return mask

# Correct the Notch filter to act as a lowpass filter


def notch_lowpass_filter(img, cutoff):
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    r = cutoff
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 1
    return mask

# Function to apply filter and inverse FFT


def apply_filter(image, filter_func, cutoff):
    dft = fftn(image)
    dft_shift = fftshift(dft)

    # Apply filter
    filter_mask = filter_func(image, cutoff)
    filtered_dft = dft_shift * filter_mask

    # Inverse FFT
    idft_shift = ifftshift(filtered_dft)
    img_back = ifftn(idft_shift)
    img_back = np.abs(img_back)
    return img_back


# Load the image
image_path = 'Noisy_flower1_vertical.jpg'
image = cv2.imread(image_path, 0)

# Apply filters with cutoffs 10, 20, ..., 100
cutoffs = list(range(10, 110, 10))
titles = ['Butterworth', 'Gaussian', 'Notch']
butterworth_images = []
gaussian_images = []
notch_filtered_images = []

# Apply Butterworth and Gaussian filters for each cutoff
for cutoff in cutoffs:
    b_img = apply_filter(image, butterworth_lowpass_filter, cutoff)
    g_img = apply_filter(image, gaussian_lowpass_filter, cutoff)
    butterworth_images.append(b_img)
    gaussian_images.append(g_img)

# Apply Notch filter for each cutoff
for cutoff in cutoffs:
    n_img = apply_filter(image, notch_lowpass_filter, cutoff)
    notch_filtered_images.append((n_img, f'Notch lowpass cutoff {cutoff}'))

# Combine all the filtered images into one figure
fig, axes = plt.subplots(len(cutoffs), len(titles), figsize=(15, 20))

for i, cutoff in enumerate(cutoffs):
    axes[i, 0].imshow(butterworth_images[i], cmap='gray')
    axes[i, 0].axis('off')
    axes[i, 0].set_title(f'Butterworth cutoff {cutoff}')

    axes[i, 1].imshow(gaussian_images[i], cmap='gray')
    axes[i, 1].axis('off')
    axes[i, 1].set_title(f'Gaussian cutoff {cutoff}')

    axes[i, 2].imshow(notch_filtered_images[i][0], cmap='gray')
    axes[i, 2].axis('off')
    axes[i, 2].set_title(notch_filtered_images[i][1])

plt.tight_layout()

# Save the combined image
final_output_path = 'filtered_vertical_combined.jpg'
plt.savefig(final_output_path)
