
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Image names
image_files = ["image01.png", "image02.jpg", "image03.png", ]

# Kernels
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

for name in image_files:

    img = cv2.imread(name)

    if img is None:
        print("Image not found:", name)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Remove small bright noise
    opening = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, kernel_open
    )

    # Fill small dark cracks
    closing = cv2.morphologyEx(
        opening, cv2.MORPH_CLOSE, kernel_close
    )

    # Display combined output
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(binary, cmap="gray")
    plt.title("Binary")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(opening, cmap="gray")
    plt.title("Opening")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(closing, cmap="gray")
    plt.title("Opening + Closing")
    plt.axis("off")

    plt.suptitle(name)
    plt.tight_layout()
    plt.show()

print("All images processed successfully.")
