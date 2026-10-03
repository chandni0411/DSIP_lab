import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "/content/image05.png"
image = cv2.imread(image_path)

# Check if image is loaded
if image is None:
    print("Error: Image not found. Check the image path.")
else:

    # Convert BGR to RGB for displaying
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    # --------------------------------------------------
    # 1. Gaussian Smoothing
    # --------------------------------------------------

    kernel_size = (5, 5)
    sigma = 1.5

    gaussian_kernel = cv2.getGaussianKernel(5, sigma)
    gaussian_kernel = np.outer(
        gaussian_kernel,
        gaussian_kernel
    )

    smoothed_gaussian = cv2.filter2D(
        image,
        -1,
        gaussian_kernel
    )


    # --------------------------------------------------
    # 2. Averaging / Mean Filter
    # --------------------------------------------------

    mean_kernel = np.ones(
        (5, 5),
        dtype=np.float32
    ) / 25

    smoothed_mean = cv2.filter2D(
        image,
        -1,
        mean_kernel
    )


    # --------------------------------------------------
    # 3. Median Filter
    # --------------------------------------------------

    smoothed_median = cv2.medianBlur(
        image,
        5
    )


    # --------------------------------------------------
    # 4. Sharpening using Spatial Filter
    # --------------------------------------------------

    sharpening_kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ], dtype=np.float32)

    sharpened = cv2.filter2D(
        image,
        -1,
        sharpening_kernel
    )


    # --------------------------------------------------
    # 5. High-Pass Sharpening
    # --------------------------------------------------

    blurred_image = cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )

    high_pass_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    high_pass_sharpened = cv2.filter2D(
        blurred_image,
        -1,
        high_pass_kernel
    )


    # --------------------------------------------------
    # Display Results
    # --------------------------------------------------

    plt.figure(figsize=(15, 10))

    # Original
    plt.subplot(2, 3, 1)
    plt.imshow(rgb)
    plt.title("Original Image")
    plt.axis("off")

    # Gaussian
    plt.subplot(2, 3, 2)
    plt.imshow(
        cv2.cvtColor(
            smoothed_gaussian,
            cv2.COLOR_BGR2RGB
        )
    )
    plt.title("Gaussian Smoothing")
    plt.axis("off")

    # Mean
    plt.subplot(2, 3, 3)
    plt.imshow(
        cv2.cvtColor(
            smoothed_mean,
            cv2.COLOR_BGR2RGB
        )
    )
    plt.title("Averaging / Mean Filter")
    plt.axis("off")

    # Median
    plt.subplot(2, 3, 4)
    plt.imshow(
        cv2.cvtColor(
            smoothed_median,
            cv2.COLOR_BGR2RGB
        )
    )
    plt.title("Median Smoothing")
    plt.axis("off")

    # Sharpening
    plt.subplot(2, 3, 5)
    plt.imshow(
        cv2.cvtColor(
            sharpened,
            cv2.COLOR_BGR2RGB
        )
    )
    plt.title("Sharpening Filter")
    plt.axis("off")

    # High-pass sharpening
    plt.subplot(2, 3, 6)
    plt.imshow(
        cv2.cvtColor(
            high_pass_sharpened,
            cv2.COLOR_BGR2RGB
        )
    )
    plt.title("High-Pass Sharpening")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


    # --------------------------------------------------
    # Save Results
    # --------------------------------------------------

    cv2.imwrite(
        "/content/gaussian_smoothing.jpg",
        smoothed_gaussian
    )

    cv2.imwrite(
        "/content/mean_filter.jpg",
        smoothed_mean
    )

    cv2.imwrite(
        "/content/median_filter.jpg",
        smoothed_median
    )

    cv2.imwrite(
        "/content/sharpened.jpg",
        sharpened
    )

    cv2.imwrite(
        "/content/high_pass_sharpened.jpg",
        high_pass_sharpened
    )

    print("All filter results have been saved successfully.")
  
