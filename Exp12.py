import cv2
import numpy as np

# Load the image in grayscale
image = cv2.imread(r'D:\dsip labs\EXP_5\images.jpg', 0)

# Check if image is loaded properly
if image is None:
    raise FileNotFoundError("Image not found. Check the image path.")

# Convert the image into binary image
_, binary = cv2.threshold(
    image, 127, 255, cv2.THRESH_BINARY
)

# Define the structuring element for the pattern
# 1  = foreground
# -1 = background
# 0  = don't care

pattern = np.array([
    [-1,  1, -1],
    [ 1,  1,  1],
    [-1,  1, -1]
], dtype=np.int8)

# Apply the Hit or Miss Transformation
result = cv2.morphologyEx(
    binary,
    cv2.MORPH_HITMISS,
    pattern
)

# Display the binary image
cv2.imshow('Binary Image', binary)

# Display the Hit or Miss result
cv2.imshow('Hit or Miss Result', result)

# Wait for a key press
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()
