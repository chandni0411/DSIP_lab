import numpy as np
import matplotlib.pyplot as plt

signal1=np.array([1,3,8,9,12,15,18,20])
signal2=np.array([2,4,6,8,10,12,14,16])

def linear_convolution(signal1, signal2):
    linear_conv=np.convolve(signal1, signal2, mode='full')
    return linear_conv

def circular_convolution(signal1, signal2):
    fft_length = max(len(signal1), len(signal2))
    fft_signal1 = np.fft.fft(signal1, fft_length)
    fft_signal2 = np.fft.fft(signal2, fft_length)
    circular_conv = np.fft.ifft(fft_signal1 * fft_signal2)

    return circular_conv
linear_conv = linear_convolution(signal1, signal2)
circular_conv = circular_convolution(signal1, signal2)

plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.stem(linear_conv)
plt.title('Linear Convolution')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
np.savetxt("linear.txt", linear_conv)


plt.subplot(2,1,2)
plt.stem(circular_conv.real)
plt.title('Circular Convolution')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
np.savetxt("circular.txt", circular_conv) 
plt.show()
plt.tight_layout()
