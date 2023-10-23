import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('blurred_image.jpg', 0)

plt.figure(figsize=(8,5), dpi=150)
plt.imshow(img, cmap='gray')
plt.axis('off')

# kernel
kernel = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]])

LaplacianImage = cv2.filter2D(src=img, 
                              ddepth=-1, 
                              kernel=kernel)

plt.figure(figsize=(8,5), dpi=150)
plt.imshow(LaplacianImage, cmap='gray')
plt.axis('off')

c = -1
g = img + c*LaplacianImage

plt.figure(figsize=(8,5), dpi=150)
plt.imshow(g, cmap='gray')
plt.axis('off')

gClip = np.clip(g, 0, 255)
plt.figure(figsize=(8,5), dpi=150)
plt.imshow(gClip, cmap='gray')
plt.axis('off')

cv2.imwrite("plc1.jpg", img)
cv2.imwrite("plc2.jpg", LaplacianImage)
cv2.imwrite("plc3.jpg", g)
cv2.imwrite("plc4.jpg", gClip)
