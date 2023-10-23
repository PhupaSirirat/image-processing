import cv2
import numpy as np

img = cv2.imread('blurred_image.jpg', 0)

# kernel
kernel = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]])

LaplacianImage = cv2.filter2D(src=img, 
                              ddepth=-1, 
                              kernel=kernel)
c = -1
g = img + c*LaplacianImage
gClip = np.clip(g, 0, 255)

cv2.imwrite("gray_img.jpg", img)
cv2.imwrite("laplacian_filter_img.jpg", LaplacianImage)
cv2.imwrite("sharpened_img.jpg", gClip)
