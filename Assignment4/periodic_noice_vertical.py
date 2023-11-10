import cv2
import numpy as np


def gaussian_notch_reject(shape, d0=30, width=10):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    y, x = np.ogrid[:rows, :cols]
    d2_from_center = (x - ccol)**2 + (y - crow)**2
    d2_from_d0_right = (x - (ccol-d0))**2 + (y - crow)**2
    d2_from_d0_left = (x - (ccol+d0))**2 + (y - crow)**2

    mask_right = np.exp(-d2_from_d0_right / (2*width**2))
    mask_left = np.exp(-d2_from_d0_left / (2*width**2))

    mask_gaussian = 1 - (mask_right + mask_left)
    mask_gaussian = mask_gaussian[:, :, np.newaxis]

    return mask_gaussian


def add_title(img, title_text):
    # Define the font and place the title at the top center of the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_size = cv2.getTextSize(title_text, font, 1, 2)[0]
    title_x = (img.shape[1] - title_size[0]) // 2
    title_y = title_size[1] + 20  # 20 pixels from the top edge
    titled_img = cv2.putText(img.copy(), title_text,
                             (title_x, title_y), font, 1, (255, 255, 255), 2)
    return titled_img


img = cv2.imread('noisy_flower1_vertical.jpg', cv2.IMREAD_GRAYSCALE)

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shifted = np.fft.fftshift(dft)

notch = gaussian_notch_reject(img.shape, 95, 25) * \
    gaussian_notch_reject(img.shape, 190, 25)

fshift = dft_shifted * notch
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift, flags=cv2.DFT_REAL_OUTPUT)
img_back = cv2.normalize(img_back, None, 0, 255,
                         cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite("Filtered_image_vertical.jpg", img_back)

magnitude_spectrum = 20 * \
    np.log(cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1]) + 1)
magnitude_spectrum_normalized = cv2.normalize(
    magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

magnitude_spectrum_output = 20 * \
    np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]) + 1)
magnitude_spectrum_output_normalized = cv2.normalize(
    magnitude_spectrum_output, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Add titles to each image
img_with_title = add_title(img, 'Original Image')
img_back_with_title = add_title(img_back, 'Processed Image')
magnitude_spectrum_with_title = add_title(
    magnitude_spectrum_normalized, 'Magnitude Spectrum')
magnitude_spectrum_output_with_title = add_title(
    magnitude_spectrum_output_normalized, 'Filtered Magnitude Spectrum')

# Combine the images with titles into a single large image
top_row = np.hstack((img_with_title, magnitude_spectrum_with_title))
bottom_row = np.hstack(
    (img_back_with_title, magnitude_spectrum_output_with_title))
combined_image_with_titles = np.vstack((top_row, bottom_row))

# Save the image to a file
cv2.imwrite('combined_image_vertical_noice.jpg', combined_image_with_titles)
