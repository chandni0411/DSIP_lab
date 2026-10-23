import cv2
import numpy as np

# Load the image in grayscale
image = cv2.imread(r'D:\dsip labs\EXP_5\images.jpg', 0)

# Check if image is loaded properly
if image is None:
    raise FileNotFoundError("Image not found. Check the image path.")

# Define the kernel (structuring element)
kernel = np.ones((8, 8 ), np.uint8)

# Perform Erosion
erosion = cv2.erode(image, kernel, iterations=1)

# Perform Dilation
dilation = cv2.dilate(image, kernel, iterations=1)

# Perform Opening (Erosion followed by Dilation)
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# Perform Closing (Dilation followed by Erosion)
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# Display the original image
cv2.imshow('Original Image', image)

# Display Erosion result
cv2.imshow('Erosion', erosion)

# Display Dilation result
cv2.imshow('Dilation', dilation)

# Display Opening result
cv2.imshow('Opening', opening)

# Display Closing result
cv2.imshow('Closing', closing)

# Wait for a key press
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()
