import numpy as np
import matplotlib.pyplot as plt

def simulate_continuous_ramp(time, slope):
    ramp = np.zeros_like(time)
    ramp[time >= 0] = slope * time[time >= 0]
    return ramp


def simulate_discrete_ramp(num_samples, slope):
    ramp = np.zeros(num_samples)

    ramp[num_samples // 2:] = (
        slope * np.arange(num_samples // 2, num_samples)
    )

    return ramp


# Define the time range for the continuous ramp signal
time = np.linspace(-5, 5, 1000)

# Define the number of samples and slope
num_samples = 20
slope = 2

# Simulate the signals
continuous_ramp = simulate_continuous_ramp(time, slope)
discrete_ramp = simulate_discrete_ramp(num_samples, slope)

# Plot and display the signals
plt.figure(figsize=(10, 6))

# Continuous Ramp Signal
plt.subplot(2, 1, 1)
plt.plot(time, continuous_ramp)
plt.title('Continuous Ramp Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

# Discrete Ramp Signal
plt.subplot(2, 1, 2)
plt.stem(discrete_ramp)
plt.title('Discrete Ramp Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()