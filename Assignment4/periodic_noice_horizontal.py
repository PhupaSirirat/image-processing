import cv2
import numpy as np


def gaussian_notch_reject(shape, d0=30, width=10):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    d2_from_d0_above = (x - ccol)**2 + (y - (crow - d0))**2
    d2_from_d0_below = (x - ccol)**2 + (y - (crow + d0))**2
    mask_above = np.exp(-d2_from_d0_above / (2 * width**2))
    mask_below = np.exp(-d2_from_d0_below / (2 * width**2))
    mask_gaussian = 1 - (mask_above + mask_below)
    mask_gaussian = mask_gaussian[:, :, np.newaxis]
    return mask_gaussian


def add_title(img, text, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.5, color=255, thickness=1, bgcolor=0):
    # Get the width and height of the text box
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, scale, thickness)
    # Create new image with space for text
    new_img = np.full(
        (img.shape[0] + text_height + baseline + 10, img.shape[1]), bgcolor, dtype=np.uint8)
    # Place the original image onto the new image
    new_img[text_height + baseline + 10:] = img
    # Set the text start position
    text_x = (new_img.shape[1] - text_width) // 2
    text_y = text_height + 5
    # Put text on new image
    cv2.putText(new_img, text, (text_x, text_y), font, scale, color, thickness)
    return new_img


img_filename = 'Noisy_flower1_horizontal.jpg'
img = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shifted = np.fft.fftshift(dft)

# Calculate magnitude spectrum of the original image
magnitude_spectrum = 20 * \
    np.log(cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1]))
magnitude_spectrum_normalized = cv2.normalize(
    magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Apply Gaussian notch filters
notch_filters = [gaussian_notch_reject(img.shape, d0, 25) for d0 in [70, 140]]
combined_notch = np.prod(notch_filters, axis=0)

fshift = dft_shifted * combined_notch
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift, flags=cv2.DFT_REAL_OUTPUT)
img_back = cv2.normalize(img_back, None, 0, 255,
                         cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite("Filtered_image_horizontal.jpg", img_back)

# Compute the magnitude spectrum for the filtered frequency domain
magnitude_spectrum_filtered = 20 * \
    np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]) + 1)
magnitude_spectrum_filtered_normalized = cv2.normalize(
    magnitude_spectrum_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

titles = ['Original Image', 'Original Magnitude Spectrum',
          'Processed Image', 'Filtered Magnitude Spectrum']

# Create titled images
titled_original = add_title(
    img, titles[0], scale=1, thickness=2, color=255, bgcolor=0)
titled_magnitude = add_title(magnitude_spectrum_normalized,
                             titles[1], scale=1, thickness=2, color=255, bgcolor=0)
titled_processed = add_title(
    img_back, titles[2], scale=1, thickness=2, color=255, bgcolor=0)
titled_filtered_magnitude = add_title(
    magnitude_spectrum_filtered_normalized, titles[3], scale=1, thickness=2, color=255, bgcolor=0)

# Stack images horizontally
combined_image = np.hstack(
    (titled_original, titled_magnitude, titled_processed, titled_filtered_magnitude))

# Save the combined image
combined_image_filename = 'combined_image_horizontal_noice.jpg'
cv2.imwrite(combined_image_filename, combined_image)
