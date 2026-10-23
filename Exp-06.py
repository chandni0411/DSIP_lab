import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz

Fs = 10000 # Sampling frequency = 10 kHz Fc = 3000	# Cutoff frequency = 3 kHz
N = 31	# Number of filter taps

b = firwin(numtaps=N, cutoff=Fc, window='hann', pass_zero='lowpass', fs=Fs)

f, H = freqz(b, worN=2048, fs=Fs)
magnitude_db = 20 * np.log10(np.maximum(np.abs(H), 1e-10)) fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.stem(range(N), b)
ax1.set_title(f'Filter Impulse Response ({N} Taps)') ax1.set_xlabel('Coefficient Index') ax1.set_ylabel('Amplitude')
ax1.grid(True)

ax2.plot(f, magnitude_db, 'b-', label='Magnitude Response') 
ax2.axvline(Fc, color='r', linestyle='--', label=f'Cutoff Frequency ({Fc/1000} kHz)')
ax2.axhline(-6, color='g', linestyle=':', label='-6 dB Point')
ax2.set_title('Filter Frequency Response') ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Magnitude (dB)') ax2.set_ylim(-80, 5) ax2.grid(True)
ax2.legend()


  import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz

# Shared Filter Parameters
Fs = 20000	# Increased to 20 kHz to properly visualize the 6 kHz Bandpass/stop parameters
N = 31 # Number of filter taps (Must be odd for Highpass / Bandstop)
window_type = 'hann'
# --- 1. Low-Pass Filter ---
fc_low = 3000
b_lp = firwin(numtaps=N, cutoff=fc_low, window=window_type,
pass_zero='lowpass', fs=Fs)
# --- 2. High-Pass Filter ---
fc_hp = 3000
b_hp = firwin(numtaps=N, cutoff=fc_hp, window=window_type,
pass_zero='highpass', fs=Fs)
# --- 3. Band-Pass Filter ---
fc_bp = [3000, 6000]
b_bp = firwin(numtaps=N, cutoff=fc_bp, window=window_type,
pass_zero='bandpass', fs=Fs)
# --- 4. Band-Stop Filter ---
fc_bs = [3000, 6000]
b_bs = firwin(numtaps=N, cutoff=fc_bs, window=window_type,
pass_zero='bandstop', fs=Fs)
# --- Compute Frequency Responses ---
filters = {
'Low-Pass (3 kHz)': b_lp,
'High-Pass (3 kHz)': b_hp,
'Band-Pass (3-6 kHz)': b_bp,
'Band-Stop (3-6 kHz)': b_bs
}

Marwadi University
Faculty of Engineering & Technology
Department of Information and Communication Technology

Subject: Digital Signal and
Image Processing (01CT1513) Aim: Design FIR filter with windowing method.
Experiment No: 6 Date: 16-10-2026 Enrollment No: 92400133020

plt.figure(figsize=(12, 8))
for label, coefficients in filters.items():
f, H = freqz(coefficients, worN=2048, fs=Fs)
magnitude_db = 20 * np.log10(np.maximum(np.abs(H), 1e-10))
plt.plot(f, magnitude_db, label=label, linewidth=2)
# Formatting the Comparison Plot
plt.title(f'FIR Filter Frequency Response Comparison (N={N}, Fs={Fs/1000}
kHz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.axvline(3000, color='k', linestyle='--', alpha=0.5, label='3 kHz
Edge')
plt.axvline(6000, color='k', linestyle=':', alpha=0.5, label='6 kHz Edge')
plt.ylim(-80, 5)
plt.xlim(0, Fs/2)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


