import cv2
import numpy as np


def fftshift(inputImg):
    outputImg = np.fft.fftshift(inputImg)
    return outputImg


def filter2DFreq(inputImg, H):
    dft = cv2.dft(np.float32(inputImg), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * \
        np.log(cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1]))

    # Create a mask first, center square is 1, remaining all zeros
    rows, cols = inputImg.shape
    crow, ccol = rows//2, cols//2     # center
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 1

    # Apply mask and inverse DFT
    fshift = dft_shifted * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return img_back


def synthesizeFilterH(inputOutput_H, center, radius):
    rows, cols = inputOutput_H.shape
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, center, radius, 1, thickness=-1)
    return mask


def calcPSD(inputImg, flag=0):
    f = np.fft.fft2(inputImg)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    return magnitude_spectrum if flag == 0 else np.log(1 + magnitude_spectrum)


# Load image in grayscale
imgIn = cv2.imread("Noisy_flower1_horizontal.jpg", cv2.IMREAD_GRAYSCALE)
if imgIn is None:
    print("ERROR : Image cannot be loaded..!!")
    exit(-1)

imgIn = np.float32(imgIn)
rows, cols = imgIn.shape
imgIn = imgIn[:rows & -2, :cols & -2]

# PSD calculation
imgPSD = calcPSD(imgIn)
imgPSD_shifted = fftshift(imgPSD)
imgPSD_normalized = cv2.normalize(
    imgPSD_shifted, None, 0, 255, cv2.NORM_MINMAX)

# H calculation
H = np.ones(imgIn.shape, np.float32)
radius = 60
H = synthesizeFilterH(H, (705, 458), radius)
H = synthesizeFilterH(H, (850, 391), radius)
H = synthesizeFilterH(H, (993, 325), radius)

# Filtering
H_shifted = fftshift(H)
imgOut = filter2DFreq(imgIn, H_shifted)
imgOut_normalized = cv2.normalize(imgOut, None, 0, 255, cv2.NORM_MINMAX)

# Save images
cv2.imwrite("Out_result.jpg", imgOut_normalized)
cv2.imwrite("Out_PSD.jpg", imgPSD_normalized)
cv2.imwrite("Out_filter.jpg", H_shifted)
