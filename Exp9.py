import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# LOAD IMAGE
# ============================================================

image_path = "/content/untitled.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
    print("Make sure 'Untitled.jpg' is in the same folder as this Python file.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ============================================================
# 1. ORIGINAL IMAGE
# ============================================================

plt.figure(figsize=(8, 6))
plt.imshow(rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()


# ============================================================
# 2. MEAN FILTER
# ============================================================

mean_5x5 = cv2.blur(image, (5, 5))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(mean_5x5, cv2.COLOR_BGR2RGB))
plt.title("Mean Filter (5×5)")
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 3. WEIGHTED MEAN FILTER
# ============================================================

weighted_kernel = np.array([
    [1, 1, 1],
    [1, 2, 1],
    [1, 1, 1]
], dtype=np.float32)

weighted_kernel = weighted_kernel / 10

weighted_mean = cv2.filter2D(image, -1, weighted_kernel)

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(weighted_mean, cv2.COLOR_BGR2RGB))
plt.title("Weighted Mean Filter")
plt.axis("off")
plt.show()


# ============================================================
# 4. GAUSSIAN FILTER
# ============================================================

# sigmaX = 0
# OpenCV automatically calculates sigma

g0 = cv2.GaussianBlur(image, (5, 5), sigmaX=0)

# sigmaX = 1
g1 = cv2.GaussianBlur(image, (5, 5), sigmaX=1)

# sigmaX = 2
g2 = cv2.GaussianBlur(image, (5, 5), sigmaX=2)


plt.figure(figsize=(16, 5))

plt.subplot(1, 4, 1)
plt.imshow(rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(cv2.cvtColor(g0, cv2.COLOR_BGR2RGB))
plt.title("Gaussian (σ=0)")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(cv2.cvtColor(g1, cv2.COLOR_BGR2RGB))
plt.title("Gaussian (σ=1)")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(cv2.cvtColor(g2, cv2.COLOR_BGR2RGB))
plt.title("Gaussian (σ=2)")
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 5. LAPLACIAN EDGE DETECTION
# ============================================================

laplacian = cv2.Laplacian(gray, cv2.CV_64F)

# Convert signed values into positive 8-bit values
laplacian_abs = cv2.convertScaleAbs(laplacian)

plt.figure(figsize=(8, 6))
plt.imshow(laplacian_abs, cmap="gray")
plt.title("Laplacian Edge Detection")
plt.axis("off")
plt.show()


# ============================================================
# 6. ROBERTS EDGE DETECTION
# ============================================================

roberts_x = np.array([
    [1, 0],
    [0, -1]
], dtype=np.float32)

roberts_y = np.array([
    [0, 1],
    [-1, 0]
], dtype=np.float32)


gx = cv2.filter2D(
    gray.astype(np.float32),
    cv2.CV_32F,
    roberts_x
)

gy = cv2.filter2D(
    gray.astype(np.float32),
    cv2.CV_32F,
    roberts_y
)

roberts = cv2.magnitude(gx, gy)

roberts = cv2.normalize(
    roberts,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)


plt.figure(figsize=(8, 6))
plt.imshow(roberts, cmap="gray")
plt.title("Roberts Edge Detection")
plt.axis("off")
plt.show()


# ============================================================
# 7. PREWITT EDGE DETECTION
# ============================================================

# Vertical Kernel
prewitt_x = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)


# Horizontal Kernel
prewitt_y = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
], dtype=np.float32)


px = cv2.filter2D(
    gray.astype(np.float32),
    cv2.CV_32F,
    prewitt_x
)

py = cv2.filter2D(
    gray.astype(np.float32),
    cv2.CV_32F,
    prewitt_y
)

prewitt = cv2.magnitude(px, py)

prewitt = cv2.normalize(
    prewitt,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)


plt.figure(figsize=(8, 6))
plt.imshow(prewitt, cmap="gray")
plt.title("Prewitt Edge Detection")
plt.axis("off")
plt.show()


# ============================================================
# 8. SOBEL EDGE DETECTION
# ============================================================

sobel_x = cv2.Sobel(
    gray,
    cv2.CV_64F,
    1,
    0,
    ksize=3
)

sobel_y = cv2.Sobel(
    gray,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)

sobel_magnitude = cv2.magnitude(
    sobel_x.astype(np.float32),
    sobel_y.astype(np.float32)
)

sobel_magnitude = cv2.normalize(
    sobel_magnitude,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)


plt.figure(figsize=(8, 6))
plt.imshow(sobel_magnitude, cmap="gray")
plt.title("Sobel Edge Detection")
plt.axis("off")
plt.show()


# ============================================================
# 9. CUSTOM SHARPENING FILTER
# ============================================================

custom_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)

custom_filtered = cv2.filter2D(
    image,
    -1,
    custom_kernel
)


print("Custom Sharpening Kernel:")
print(custom_kernel)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(
    custom_filtered,
    cv2.COLOR_BGR2RGB
))
plt.title("Custom Sharpening Filter")
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 10. USER-DEFINED CUSTOM FILTER
# ============================================================

my_kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=np.float32)


# Normalize the filter
my_kernel = my_kernel / my_kernel.sum()


custom_result = cv2.filter2D(
    image,
    -1,
    my_kernel
)


print("User-Defined Custom Kernel:")
print(my_kernel)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(
    custom_result,
    cv2.COLOR_BGR2RGB
))
plt.title("User-Defined Custom Filter")
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 11. SAVE ALL RESULTS
# ============================================================

cv2.imwrite("mean_filter.jpg", mean_5x5)

cv2.imwrite(
    "weighted_mean_filter.jpg",
    weighted_mean
)

cv2.imwrite(
    "gaussian_filter.jpg",
    g0
)

cv2.imwrite(
    "laplacian.jpg",
    laplacian_abs
)

cv2.imwrite(
    "roberts.jpg",
    roberts
)

cv2.imwrite(
    "prewitt.jpg",
    prewitt
)

cv2.imwrite(
    "sobel.jpg",
    sobel_magnitude
)

cv2.imwrite(
    "custom_filter.jpg",
    custom_filtered
)

cv2.imwrite(
    "user_defined_filter.jpg",
    custom_result
)


print()
print("========================================")
print("All filter results have been saved.")
print("========================================")
