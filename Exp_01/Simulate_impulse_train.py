import numpy as np
import matplotlib.pyplot as plt

def simulate_impulse_train(signal_length, period):
    impulse_train = np.zeros(signal_length)

    for n in range(signal_length):
        if n % period == 0:
            impulse_train[n] = 1

    return impulse_train


# Define the parameters for the impulse train
signal_length = 100   # Length of the impulse train
period = 10           # Period of the impulse train

# Simulate the impulse train
impulse_train = simulate_impulse_train(signal_length, period)

# Plot and display the impulse train
plt.stem(impulse_train)
plt.title('Impulse Train')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()