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

sobel_x, sobel_y = cv2.filter2D(img, -1, kernel_x), cv2.filter2D(img, -1, kernel_y)
magitube = np.sqrt(sobel_x**2 + sobel_y**2)
sharpen_img = img - magitube

cv2.imwrite("grad_sobel_x.jpg", sobel_x)
cv2.imwrite("grad_sobel_y.jpg", sobel_y)
cv2.imwrite("grad_magitube.jpg", magitube)
cv2.imwrite("grad_sharpened_img.jpg", sharpen_img)
