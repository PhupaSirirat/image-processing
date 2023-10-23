import cv2
import numpy as np

# Sobel filtering
img = cv2.imread('blurred_image.jpg')

kernel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

kernel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])
kernel_size = len(kernel_x)

sobel_x = np.zeros_like(img)
sobel_y = np.zeros_like(img)
magitube = np.zeros_like(img)
sharpen_img = np.zeros_like(img)

for i in range(kernel_size):
    sobel_x[:, :, i] = cv2.filter2D(img[:, :, i], -1, kernel_x)
    sobel_y[:, :, i] = cv2.filter2D(img[:, :, i], -1, kernel_y)
    magitube[:, :, i] = np.sqrt(sobel_x[:, :, i]**2 + sobel_y[:, :, i]**2)
    sharpen_img[:, :, i] = img[:, :, i] - magitube[:, :, i]

cv2.imwrite("grad_sobel_x.jpg", sobel_x)
cv2.imwrite("grad_sobel_y.jpg", sobel_y)
cv2.imwrite("grad_magitube.jpg", magitube)
cv2.imwrite("grad_sharpened_img.jpg", sharpen_img)
