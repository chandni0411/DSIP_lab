import cv2
import numpy as np
import matplotlib.pyplot as plt

# PART 1: LOAD IMAGE

image_path = 'openEnded08/images184.jpg'

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found!")
    exit()

print("Image loaded successfully!")
# PART 2: ORIGINAL HISTOGRAM

histogram = cv2.calcHist(
    [image], [0], None, [256], [0, 256]
)

plt.figure(figsize=(8, 6))
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.plot(histogram)
plt.xlim([0, 256])
plt.grid(True)
plt.show()

# PART 3: HISTOGRAM EQUALIZATION

equalized_image = cv2.equalizeHist(image)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Equalized Image')
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# PART 4: EQUALIZED HISTOGRAM

equalized_histogram = cv2.calcHist(
    [equalized_image], [0], None, [256], [0, 256]
)

plt.figure(figsize=(8, 6))
plt.title('Equalized Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.plot(equalized_histogram)
plt.xlim([0, 256])
plt.grid(True)
plt.show()

# PART 5: HISTOGRAM MATCHING

# Source image
source_path = 'openEnded08/images184.jpg'

# Reference image
reference_path = 'openEnded08/image02.jpg'

source_image = cv2.imread(
    source_path,
    cv2.IMREAD_GRAYSCALE
)

reference_image = cv2.imread(
    reference_path,
    cv2.IMREAD_GRAYSCALE
)

# Check images
if source_image is None:
    print("Source image not found!")
    exit()

if reference_image is None:
    print("Reference image not found!")
    exit()

print("Source and reference images loaded successfully!")

# Calculate source histogram
source_hist = cv2.calcHist(
    [source_image], [0], None, [256], [0, 256]
)

# Calculate reference histogram
reference_hist = cv2.calcHist(
    [reference_image], [0], None, [256], [0, 256]
)

# Normalize histograms
source_hist /= source_hist.sum()
reference_hist /= reference_hist.sum()

# Calculate CDF
source_cdf = source_hist.cumsum()
reference_cdf = reference_hist.cumsum()

# Create mapping
mapping = np.interp(
    source_cdf,
    reference_cdf,
    range(256)
)

# Apply mapping
matched_image = mapping[source_image]

matched_image = matched_image.astype(np.uint8)

# PART 6: DISPLAY HISTOGRAM MATCHING RESULT

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title('Source Image')
plt.imshow(source_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Reference Image')
plt.imshow(reference_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Matched Image')
plt.imshow(matched_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
