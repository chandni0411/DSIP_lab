
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv

# Folder path
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output10"
OUTPUT_DIR.mkdir(exist_ok=True)

# Your image names
image_files = [
    "Picture1.png",
    "Picture2.png",
    "Picture3.png",
    "Picture4.png",
    "Picture5.png"
]

# Gaussian Low Pass Filter
def gaussian_lowpass(rows, cols, cutoff=35):
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((y - rows//2)**2 + (x - cols//2)**2)
    return np.exp(-(distance**2) / (2 * cutoff**2))

# Butterworth Low Pass Filter
def butterworth_lowpass(rows, cols, cutoff=35, order=2):
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((y - rows//2)**2 + (x - cols//2)**2)
    return 1 / (1 + (distance / cutoff)**(2 * order))

# Apply frequency filter
def frequency_filter(image, H):
    F = np.fft.fft2(image)
    F_shift = np.fft.fftshift(F)
    filtered = F_shift * H
    F_ishift = np.fft.ifftshift(filtered)
    result = np.fft.ifft2(F_ishift)
    return np.clip(np.real(result), 0, 255).astype(np.uint8)

# Metrics
def mse(img1, img2):
    return np.mean((img1.astype(float) - img2.astype(float))**2)

def psnr(img1, img2):
    error = mse(img1, img2)
    if error == 0:
        return float("inf")
    return 10 * np.log10(255**2 / error)

def sharpness(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

def edge_density(edges):
    return np.count_nonzero(edges) / edges.size * 100

# Store comparison results
results = []

# Process images
for name in image_files:

    img_path = BASE_DIR / name
    img = cv2.imread(str(img_path))

    if img is None:
        print("Image not found:", name)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape

    # Create filters
    H_gaussian = gaussian_lowpass(rows, cols, 35)
    H_butter = butterworth_lowpass(rows, cols, 35, 2)

    # Smoothing
    gaussian_smooth = frequency_filter(gray, H_gaussian)
    butter_smooth = frequency_filter(gray, H_butter)

    # High boost sharpening
    A = 1.5
    H_highboost = A - H_gaussian
    sharpened = frequency_filter(gray, H_highboost)

    # Boundary detection
    edges_original = cv2.Canny(gray, 80, 160)
    edges_gaussian = cv2.Canny(gaussian_smooth, 80, 160)
    edges_butter = cv2.Canny(butter_smooth, 80, 160)

    # Save outputs
    cv2.imwrite(str(OUTPUT_DIR / f"{name}_gaussian.png"), gaussian_smooth)
    cv2.imwrite(str(OUTPUT_DIR / f"{name}_butterworth.png"), butter_smooth)
    cv2.imwrite(str(OUTPUT_DIR / f"{name}_sharpened.png"), sharpened)
    cv2.imwrite(str(OUTPUT_DIR / f"{name}_edges.png"), edges_butter)

    # Calculate metrics
    results.append([
        name,
        round(mse(gray, gaussian_smooth), 2),
        round(psnr(gray, gaussian_smooth), 2),
        round(sharpness(gaussian_smooth), 2),
        round(mse(gray, butter_smooth), 2),
        round(psnr(gray, butter_smooth), 2),
        round(sharpness(butter_smooth), 2),
        round(edge_density(edges_original), 2),
        round(edge_density(edges_butter), 2)
    ])

    # Combined display
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(gaussian_smooth, cmap="gray")
    plt.title("Gaussian Low Pass")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(butter_smooth, cmap="gray")
    plt.title("Butterworth Low Pass")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(sharpened, cmap="gray")
    plt.title("High Boost Sharpening")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(edges_original, cmap="gray")
    plt.title("Original Boundary")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(edges_butter, cmap="gray")
    plt.title("Filtered Boundary")
    plt.axis("off")

    plt.suptitle(name)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / f"{name}_combined.png", dpi=150)
    plt.show()

# Save CSV report
with open(OUTPUT_DIR / "comparison_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Image", "Gaussian MSE", "Gaussian PSNR",
        "Gaussian Sharpness", "Butterworth MSE",
        "Butterworth PSNR", "Butterworth Sharpness",
        "Original Edge Density", "Filtered Edge Density"
    ])
    writer.writerows(results)

print("\nAll 5 images processed successfully!")
print("Output folder:", OUTPUT_DIR)
print("Metrics saved in comparison_metrics.csv")
