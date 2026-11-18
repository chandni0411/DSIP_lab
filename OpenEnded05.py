Program: 
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path 
from scipy.ndimage import generic_filter import csv 
# 1. FOLDER PATHS 
BASE_DIR = Path(__file__).resolve().parent INPUT_DIR = BASE_DIR / "images" 
OUTPUT_DIR = BASE_DIR / "output" 
OUTPUT_DIR.mkdir(exist_ok=True)
extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp",  "*.tif", "*.tiff"] 
image_files = [] 
for ext in extensions: 
image_files.extend(INPUT_DIR.glob(ext)) 
image_files = sorted(image_files) 
if len(image_files) != 4: 
print("Expected 4 images, found:", len(image_files)) 
if len(image_files) == 0: 
raise FileNotFoundError("No images found in the images  folder.") 
# 2. ALPHA-TRIMMED MEAN FILTER 
def alpha_trimmed_filter(image, kernel_size=3,  trim_percent=20): 
def trim_mean(values): 
values = np.sort(values) 
trim_count = int(len(values) * trim_percent / 200) 
if trim_count > 0: 
values = values[trim_count:-trim_count]
return np.mean(values) 
channels = [] 
for channel in range(3): 
filtered = generic_filter( 
image[:, :, channel], 
trim_mean, 
size=kernel_size 
) 
channels.append(filtered) 
return np.stack(channels, axis=2).astype(np.uint8) 
# 3. IMAGE QUALITY MEASUREMENTS 
def calculate_metrics(image): 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) 
edge_strength = np.mean( 
np.sqrt(sobel_x ** 2 + sobel_y ** 2) 
) 
sharpness = cv2.Laplacian( 
gray, cv2.CV_64F 
).var()
return edge_strength, sharpness 
# 4. PROCESS EACH IMAGE 
results = [] 
for image_number, image_path in enumerate(image_files,  start=1): 
print("\nProcessing Image", image_number, ":",  image_path.name) 
original = cv2.imread(str(image_path)) 
if original is None: 
print("Unable to read image:", image_path.name) continue 
# Apply median filters 
median_3 = cv2.medianBlur(original, 3) 
median_5 = cv2.medianBlur(original, 5) 
median_7 = cv2.medianBlur(original, 7) 
# Apply alpha-trimmed mean filter 
alpha_trimmed = alpha_trimmed_filter( 
original, 3, 20 
) 
# Apply bilateral filter 
bilateral = cv2.bilateralFilter(
original, 9, 75, 75 
) 
# All outputs of this image together filtered_images = [ 
("Original", original), 
("Median 3x3", median_3), 
("Median 5x5", median_5), 
("Median 7x7", median_7), 
("Alpha Trimmed Mean", alpha_trimmed), ("Bilateral Filter", bilateral) 
] 
# 5. CREATE ONE COMBINED FIGURE 
fig, axes = plt.subplots(2, 3, figsize=(15, 9)) 
fig.suptitle( 
f"Image {image_number} - Non-Linear Filtering  Comparison", 
fontsize=17 
) 
for ax, (title, filtered) in zip( 
axes.ravel(), filtered_images 
): 
display_image = cv2.cvtColor( 
filtered, cv2.COLOR_BGR2RGB 
)
ax.imshow(display_image) 
ax.set_title(title, fontsize=12) 
ax.axis("off") 
plt.tight_layout(rect=[0, 0, 1, 0.94]) 
# Save all six results in ONE image comparison_path = OUTPUT_DIR /  
f"Image_{image_number}_Comparison.png" 
plt.savefig( 
comparison_path, 
dpi=200, 
bbox_inches="tight" 
) 
plt.show() 
plt.close() 
# 6. CALCULATE QUALITY MEASUREMENTS for filter_name, filtered in filtered_images: 
edge, sharpness = calculate_metrics(filtered) 
results.append([ 
image_path.name, 
filter_name, 
round(edge, 2),
round(sharpness, 2) 
]) 
print("Saved:", comparison_path.name) 
# 7. SAVE ANALYSIS CSV 
csv_path = OUTPUT_DIR / "filter_analysis.csv" with open(csv_path, "w", newline="") as file: writer = csv.writer(file) 
writer.writerow([ 
"Image", 
"Filter", 
"Average Edge Strength", 
"Laplacian Variance" 
]) 
writer.writerows(results) 
print("\n================================") print("ALL FOUR IMAGES PROCESSED!") 
print("================================") 
print("Check the output folder:", OUTPUT_DIR) print("Four combined comparison images and one CSV file created.")
