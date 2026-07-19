import numpy as np
import matplotlib.pyplot as plt

def simulate_continuous_exponential(time, amplitude, coefficient):
    exponential_signal = amplitude * np.exp(coefficient * time)
    return exponential_signal


def simulate_discrete_exponential(num_samples, amplitude, coefficient):
    exponential_signal = amplitude * np.exp(
        coefficient * np.arange(num_samples)
    )
    return exponential_signal


# Define the time range for the continuous exponential signal
time = np.linspace(0, 5, 1000)

# Define the parameters
num_samples = 20      # Number of samples
amplitude = 2         # Initial amplitude
coefficient = -0.5    # Exponential coefficient

# Simulate the signals
continuous_exponential = simulate_continuous_exponential(
    time, amplitude, coefficient
)

discrete_exponential = simulate_discrete_exponential(
    num_samples, amplitude, coefficient
)

# Plot and display the signals
plt.figure(figsize=(10, 6))

# Continuous Exponential Signal
plt.subplot(2, 1, 1)
plt.plot(time, continuous_exponential)
plt.title('Continuous Exponential Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

# Discrete Exponential Signal
plt.subplot(2, 1, 2)
plt.stem(discrete_exponential)
plt.title('Discrete Exponential Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()