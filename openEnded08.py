import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SOURCE IMAGES - 5 DEGRADED IMAGES
# ============================================================

source_paths = [
    'openEnded08/image01.png',
    'openEnded08/imagee02.png',
    'openEnded08/image03.png',
    'openEnded08/image04.png',
    'openEnded08/image05.png'
]


# ============================================================
# REFERENCE IMAGE
# ============================================================

reference_path = 'openEnded08/Reference.jpg'


# ============================================================
# LOAD REFERENCE IMAGE
# ============================================================

reference_image = cv2.imread(
    reference_path,
    cv2.IMREAD_GRAYSCALE
)

if reference_image is None:
    print("Reference image not found!")
    exit()

print("Reference image loaded successfully!")


# ============================================================
# PROCESS ALL 5 SOURCE IMAGES
# ============================================================

for i, source_path in enumerate(source_paths):

    print("\nProcessing Image", i + 1)

    # --------------------------------------------------------
    # Load source image
    # --------------------------------------------------------

    source_image = cv2.imread(
        source_path,
        cv2.IMREAD_GRAYSCALE
    )

    if source_image is None:
        print("Image not found:", source_path)
        continue

    print("Image loaded successfully!")


    # ========================================================
    # PART 1: ORIGINAL HISTOGRAM
    # ========================================================

    histogram = cv2.calcHist(
        [source_image],
        [0],
        None,
        [256],
        [0, 256]
    )

    plt.figure(figsize=(8, 6))
    plt.title('Histogram - Image ' + str(i + 1))
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.plot(histogram)
    plt.xlim([0, 256])
    plt.grid(True)
    plt.show()


    # ========================================================
    # PART 2: HISTOGRAM EQUALIZATION
    # ========================================================

    equalized_image = cv2.equalizeHist(source_image)


    # Display Original and Equalized Image
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(source_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Equalized Image')
    plt.imshow(equalized_image, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


    # ========================================================
    # PART 3: EQUALIZED HISTOGRAM
    # ========================================================

    equalized_histogram = cv2.calcHist(
        [equalized_image],
        [0],
        None,
        [256],
        [0, 256]
    )

    plt.figure(figsize=(8, 6))
    plt.title('Equalized Histogram - Image ' + str(i + 1))
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.plot(equalized_histogram)
    plt.xlim([0, 256])
    plt.grid(True)
    plt.show()


    # ========================================================
    # PART 4: HISTOGRAM MATCHING
    # SAME METHOD AS SIR'S CODE
    # ========================================================

    # Calculate histogram of source image
    source_hist = cv2.calcHist(
        [source_image],
        [0],
        None,
        [256],
        [0, 256]
    )

    # Calculate histogram of reference image
    reference_hist = cv2.calcHist(
        [reference_image],
        [0],
        None,
        [256],
        [0, 256]
    )


    # Normalize histograms
    source_hist /= source_hist.sum()
    reference_hist /= reference_hist.sum()


    # Calculate CDF
    source_cdf = source_hist.cumsum()
    reference_cdf = reference_hist.cumsum()


    # Create mapping from source CDF to reference CDF
    mapping = np.interp(
        source_cdf,
        reference_cdf,
        range(256)
    )


    # Apply mapping
    matched_image = mapping[source_image]


    # Convert to uint8
    matched_image = matched_image.astype(np.uint8)


    # ========================================================
    # PART 5: DISPLAY SOURCE, REFERENCE AND MATCHED IMAGE
    # SAME AS SIR'S CODE
    # ========================================================

    plt.figure(figsize=(12, 6))

    plt.subplot(131)
    plt.title('Source Image')
    plt.imshow(source_image, cmap='gray')
    plt.axis('off')

    plt.subplot(132)
    plt.title('Reference Image')
    plt.imshow(reference_image, cmap='gray')
    plt.axis('off')

    plt.subplot(133)
    plt.title('Matched Image')
    plt.imshow(matched_image, cmap='gray')
    plt.axis('off')

    plt.suptitle(
        'Histogram Matching - Image ' + str(i + 1)
    )

    plt.tight_layout()
    plt.show()


    # ========================================================
    # PART 6: MATCHED IMAGE HISTOGRAM
    # ========================================================

    matched_histogram = cv2.calcHist(
        [matched_image],
        [0],
        None,
        [256],
        [0, 256]
    )

    plt.figure(figsize=(8, 6))
    plt.title(
        'Matched Histogram - Image ' + str(i + 1)
    )
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.plot(matched_histogram)
    plt.xlim([0, 256])
    plt.grid(True)
    plt.show()


print("\nAll 5 images processed successfully!")
# Change contrast
contrast_image = cv2.convertScaleAbs(
    source_image,
    alpha=1.5,
    beta=0
)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(source_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Contrast Changed")
plt.imshow(contrast_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
