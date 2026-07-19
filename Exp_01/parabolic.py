import numpy as np
import matplotlib.pyplot as plt

def simulate_continuous_parabolic(time, coefficients):
    parabolic_signal = np.polyval(coefficients, time)
    return parabolic_signal


def simulate_discrete_parabolic(num_samples, coefficients):
    parabolic_signal = np.polyval(
        coefficients,
        np.arange(num_samples)
    )
    return parabolic_signal


# Define the time range for the continuous parabolic signal
time = np.linspace(-5, 5, 1000)

# Define the number of samples and coefficients
num_samples = 20
coefficients = [1, 2, 1]   # a, b, c

# Simulate the signals
continuous_parabolic = simulate_continuous_parabolic(
    time, coefficients
)

discrete_parabolic = simulate_discrete_parabolic(
    num_samples, coefficients
)

# Plot and display the signals
plt.figure(figsize=(10, 6))

# Continuous Parabolic Signal
plt.subplot(2, 1, 1)
plt.plot(time, continuous_parabolic)
plt.title('Continuous Parabolic Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

# Discrete Parabolic Signal
plt.subplot(2, 1, 2)
plt.stem(discrete_parabolic)
plt.title('Discrete Parabolic Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()