import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirfilter


# Function to read the data from a file
def read_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    return data[:, 0], data[:, 1]


def low_pass_filter_iir(samples, cutoff_frequency, sample_rate):
  # Design an IIR filter (Butterworth filter used here)
  nyquist_rate = 0.5 * sample_rate
  Wn = cutoff_frequency / nyquist_rate  # Normalize for Nyquist
  b, a = iirfilter(Wn, 2, btype='lowpass')  # Design filter (order=2)

  # Apply the IIR filter to the samples
  filtered_samples = np.filtfilt(b, a, samples)
  return filtered_samples


def low_pass_filter_fft(fft_coefficients, cutoff_frequency, sample_rate):
  num_samples = len(fft_coefficients)
  normalized_cutoff_freq = cutoff_frequency / (0.5 * sample_rate)  # Normalize for Nyquist

  # Create a low-pass filter mask (ideal filter in this example)
  filter_mask = np.ones(num_samples)
  filter_mask[int(normalized_cutoff_freq * num_samples):] = 0.0

  # Apply the filter mask to the FFT coefficients
  filtered_coefficients = fft_coefficients * filter_mask
  return filtered_coefficients


def remove_offset(column):
    min_value = min(column)
    return [value - min_value for value in column]



#######################
#                     #
#  Load the CSV file  #
#                     #
#######################
file_path = '/home/dlaroche/DualAntenna/ADC Samples (Date_5-21-2024 Time-0_54_18).csv'  # Replace with your file path
positive_real_values, negative_real_values = read_data(file_path)


# Convert to 12-bit real-valued numbers
# Assuming the values are integers between 0 and 4095 (12-bit range) this will output range [1,-1]
max_12bit_value = 2**12 - 1
#positive_real_values = positive_real_values.astype(np.float32) / max_12bit_value
#negative_real_values = negative_real_values.astype(np.float32) / max_12bit_value

#Convert the negative polarity, positive only values, into negative values
negative_real_values = -negative_real_values

# Remove DC offset
positive_real_values -= np.mean(positive_real_values)
negative_real_values -= np.mean(negative_real_values)

# Define the sampling rate
sampling_rate = 1000000  # 1 MHz
gain = 12

# Generate both polarity FFT results
positive_fft = np.fft.rfft(positive_real_values)
positive_fft = low_pass_filter_fft(positive_fft, 1000000, 1000000)
positive_fft_frequencies = np.fft.rfftfreq(len(positive_real_values), d=1/sampling_rate)
# positive_fft *= gain

negative_fft = np.fft.rfft(negative_real_values)
negative_fft = low_pass_filter_fft(negative_fft, 1000000, 1000000)
negative_fft_frequencies = np.fft.rfftfreq(len(negative_real_values), d=1/sampling_rate) # Place a - in front of np.fft.rfftfreq() to get negative frequency numbers in legend
# negative_fft *= gain

# Calculate the positive magnitude of the FFT in dBi
# positive_magnitude_dbi = -20 * np.log10(np.abs(positive_fft) / (2**12 - 1 * 3.3))
positive_magnitude_dbi = np.abs(positive_fft)

# Calculate the negative magnitude of the FFT in dBi
# negative_magnitude_dbi = -20 * np.log10(np.abs(negative_fft) / (2**12 - 1 * 3.3))
negative_magnitude_dbi = np.abs(negative_fft)



# Plot the positive and negative components
plt.figure(figsize=(12, 8))

# Plot the positive polarity FFT
plt.subplot(2, 1, 1)
plt.plot(positive_fft_frequencies, positive_magnitude_dbi, label='Positive Polarity')
plt.title('Positive Polarity FFT')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.grid(True)
plt.legend()

# Plot the negative polarity FFT
plt.subplot(2, 1, 2)
plt.plot(negative_fft_frequencies, negative_magnitude_dbi, label='Negative Polarity')
plt.title('Negative Polarity FFT')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
