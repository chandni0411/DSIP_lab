import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import correlate


# =========================================
# 1. FILE PATHS
# =========================================

base_dir = os.path.dirname(os.path.abspath(__file__))

original_file = os.path.join(base_dir, "original.mp3")
karaoke_file = os.path.join(base_dir, "karaoke.mp3")
different_file = os.path.join(base_dir, "different.mp3")


# =========================================
# 2. CHECK FILES
# =========================================

for file in [original_file, karaoke_file, different_file]:

    if not os.path.exists(file):
        print("File not found:", file)
        raise FileNotFoundError(file)


# =========================================
# 3. LOAD AUDIO FILES
# =========================================

sr = 22050

original, _ = librosa.load(
    original_file,
    sr=sr,
    mono=True
)

karaoke, _ = librosa.load(
    karaoke_file,
    sr=sr,
    mono=True
)

different, _ = librosa.load(
    different_file,
    sr=sr,
    mono=True
)


# =========================================
# 4. USE FIRST 15 SECONDS
# =========================================

duration = 15
samples = sr * duration

original = original[:samples]
karaoke = karaoke[:samples]
different = different[:samples]


# =========================================
# 5. MAKE SAME LENGTH
# =========================================

length = min(
    len(original),
    len(karaoke),
    len(different)
)

original = original[:length]
karaoke = karaoke[:length]
different = different[:length]


# =========================================
# 6. NORMALIZE AUDIO
# =========================================

original = original / np.max(np.abs(original))
karaoke = karaoke / np.max(np.abs(karaoke))
different = different / np.max(np.abs(different))


# =========================================
# 7. NORMALIZED CROSS-CORRELATION
# =========================================

def normalized_correlation(x, y):

    corr = correlate(
        x,
        y,
        mode="full"
    )

    value = np.max(np.abs(corr)) / (
        np.linalg.norm(x) *
        np.linalg.norm(y)
    )

    return corr, value


# Three comparisons

corr_original_karaoke, value_original_karaoke = \
    normalized_correlation(original, karaoke)

corr_original_different, value_original_different = \
    normalized_correlation(original, different)

corr_karaoke_different, value_karaoke_different = \
    normalized_correlation(karaoke, different)


# =========================================
# 8. AUTO-CORRELATION
# =========================================

auto_original = correlate(
    original,
    original,
    mode="full"
)

auto_karaoke = correlate(
    karaoke,
    karaoke,
    mode="full"
)

auto_different = correlate(
    different,
    different,
    mode="full"
)


# =================================================
# PLOT 1
# AUTO-CORRELATION
# =================================================

plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(auto_original)
plt.title("Autocorrelation - Original")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(auto_karaoke)
plt.title("Autocorrelation - Karaoke")
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(auto_different)
plt.title("Autocorrelation - Different")
plt.grid()

plt.tight_layout()
plt.show()


# =================================================
# PLOT 2
# CROSS-CORRELATION
# =================================================

plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(corr_original_karaoke)
plt.title("Cross Correlation: Original vs Karaoke")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(corr_original_different)
plt.title("Cross Correlation: Original vs Different")
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(corr_karaoke_different)
plt.title("Cross Correlation: Karaoke vs Different")
plt.grid()

plt.tight_layout()
plt.show()


# =================================================
# PLOT 3
# NORMALIZED CORRELATION VALUES
# =================================================

print("\n==========================================")
print("NORMALIZED CORRELATION VALUES")
print("==========================================")

print(
    "Original vs Karaoke   :",
    round(value_original_karaoke, 4)
)

print(
    "Original vs Different :",
    round(value_original_different, 4)
)

print(
    "Karaoke vs Different  :",
    round(value_karaoke_different, 4)
)

print("==========================================")


# =================================================
# PLOT 4
# NORMALIZED CORRELATION COMPARISON
# =================================================

pairs = [
    "Original\nKaraoke",
    "Original\nDifferent",
    "Karaoke\nDifferent"
]

values = [
    value_original_karaoke,
    value_original_different,
    value_karaoke_different
]

plt.figure(figsize=(7, 5))

plt.bar(
    pairs,
    values
)

plt.title("Normalized Correlation Comparison")

plt.ylabel("Correlation Value")

plt.ylim(0, 1)

plt.grid(axis="y")

plt.tight_layout()
plt.show()
