import numpy as np
import matplotlib.pyplot as plt

def simulate_continuous_sine_wave(time, amplitude, frequency, phase):
    sine_wave = amplitude * np.sin(
        2 * np.pi * frequency * time + phase
    )
    return sine_wave


def simulate_discrete_sine_wave(
        num_samples,
        sampling_frequency,
        amplitude,
        frequency,
        phase):

    time = np.arange(num_samples) / sampling_frequency

    sine_wave = amplitude * np.sin(
        2 * np.pi * frequency * time + phase
    )

    return sine_wave


# Define the time range for the continuous sine wave signal
time = np.linspace(0, 1, 1000)

# Define parameters
num_samples = 100
sampling_frequency = 50      # Hz
amplitude = 1
frequency = 2                # Hz
phase = 0                    # radians

# Simulate the signals
continuous_sine_wave = simulate_continuous_sine_wave(
    time,
    amplitude,
    frequency,
    phase
)

discrete_sine_wave = simulate_discrete_sine_wave(
    num_samples,
    sampling_frequency,
    amplitude,
    frequency,
    phase
)

# Plot and display the signals
plt.figure(figsize=(10, 6))

# Continuous Sine Wave
plt.subplot(2, 1, 1)
plt.plot(time, continuous_sine_wave)
plt.title('Continuous Sine Wave Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# Discrete Sine Wave
plt.subplot(2, 1, 2)
plt.stem(discrete_sine_wave)
plt.title('Discrete Sine Wave Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()