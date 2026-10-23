import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read image in grayscale
img = cv2.imread(r'D:\dsip labs\EXP_5\images.jpg', 0)

if img is None:
    raise FileNotFoundError("Image not found. Check the image path.")

# 2. Fourier Transform
F = np.fft.fftshift(np.fft.fft2(img))

# Get image dimensions
M, N = img.shape

# 3. Create frequency distance matrix
u, v = np.meshgrid(
    np.arange(N) - N // 2,
    np.arange(M) - M // 2
)

D = np.sqrt(u**2 + v**2)

# 4. Create Low-Pass and High-Pass Filters
cutoff = 47

LPF = D < cutoff
HPF = D >= cutoff

# 5. Apply Low-Pass Filter for smoothing
F_LPF = F * LPF
smooth = np.abs(
    np.fft.ifft2(np.fft.ifftshift(F_LPF))
)

# 6. Apply High-Pass Filter for sharpening
F_HPF = F * HPF
high_pass = np.abs(
    np.fft.ifft2(np.fft.ifftshift(F_HPF))
)

# Add high-frequency component to original image
sharp = np.clip(
    img.astype(float) + high_pass,
    0,
    255
)

# 7. Display results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(smooth, cmap='gray')
plt.title('LPF - Smoothening')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sharp, cmap='gray')
plt.title('HPF - Sharpening')
plt.axis('off')

plt.tight_layout()
plt.show()
