import numpy as np
import matplotlib.pyplot as plt

# Original Signal
signal = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# FFT
fft_result = np.fft.fft(signal)

# Magnitude & Phase
magnitude_spectrum = np.abs(fft_result)
phase_spectrum = np.angle(fft_result)

# IFFT
reconstructed_signal = np.fft.ifft(fft_result)

# ------------------ Plot Functions ------------------

def plot_original_signal(signal):
    plt.figure(figsize=(8,4))
    plt.stem(signal)
    plt.title("Original Signal")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

def plot_magnitude_spectrum(magnitude_spectrum):
    plt.figure(figsize=(8,4))
    plt.stem(magnitude_spectrum)
    plt.title("Magnitude Spectrum")
    plt.xlabel("Frequency Index")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.show()

def plot_phase_spectrum(phase_spectrum):
    plt.figure(figsize=(8,4))
    plt.stem(phase_spectrum)
    plt.title("Phase Spectrum")
    plt.xlabel("Frequency Index")
    plt.ylabel("Phase (Radians)")
    plt.grid()
    plt.show()

# Real + Imaginary in ONE plot with data values
def plot_complex_signal(complex_signal):
    plt.figure(figsize=(10,5))
    x = np.arange(len(complex_signal))

    # Real Part
    plt.stem(x, np.real(complex_signal),
             linefmt='b-', markerfmt='bo', basefmt=" ")

    # Imaginary Part
    plt.stem(x, np.imag(complex_signal),
             linefmt='r-', markerfmt='ro', basefmt=" ")

    # Data Labels
    for i in range(len(complex_signal)):
        plt.text(x[i], np.real(complex_signal[i]) + 0.8,
                 f"{np.real(complex_signal[i]):.1f}",
                 color='blue', ha='center', fontsize=8)

        plt.text(x[i], np.imag(complex_signal[i]) - 1.2,
                 f"{np.imag(complex_signal[i]):.1f}",
                 color='red', ha='center', fontsize=8)

    plt.title("Complex FFT Signal (Real & Imaginary)")
    plt.xlabel("Frequency Index")
    plt.ylabel("Amplitude")
    plt.legend(["Real", "Imaginary"])
    plt.grid()
    plt.show()

def plot_reconstructed_signal(reconstructed_signal):
    plt.figure(figsize=(8,4))
    plt.stem(np.real(reconstructed_signal))
    plt.title("Reconstructed Signal using IFFT")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

# ------------------ Function Calls ------------------

plot_original_signal(signal)
plot_magnitude_spectrum(magnitude_spectrum)
plot_phase_spectrum(phase_spectrum)
plot_complex_signal(fft_result)
plot_reconstructed_signal(reconstructed_signal)
