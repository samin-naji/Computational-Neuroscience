from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

data = loadmat('D:\\computational_neuroscience\\assignment6\\ssvep_data_assignment.mat')

print(data.keys())
# Task 1:
# get channel col from our .mat file
channel_arr = data.get('channel')

# flatten the array to a one-dimensional array
channel_arr = channel_arr.flatten()

# find the index of Oz in the flattened array
index_Oz = np.where(channel_arr == 'OZ')[0][0]

# get data col from our .mat file
_data = data.get('data')

# get freq col from our .mat file
freq_arr = data.get('freq')

# flatten the array to a one-dimensional array
freq_arr = freq_arr.flatten()

# find the index of 8 & 12 (Hz) in the flattened array
index_8Hz = np.where(freq_arr == 8)[0][0]
index_12Hz = np.where(freq_arr == 12)[0][0]

# calculate mean of data in channel Oz with 8 & 12 frequency
data_segment_8Hz = np.mean(_data[index_Oz, :, :, index_8Hz], axis=1)
data_segment_12Hz = np.mean(_data[index_Oz, :, :, index_12Hz], axis=1)

data_segment_8Hz_list = _data[index_Oz, :, :, index_8Hz]
data_segment_12Hz_list = _data[index_Oz, :, :, index_12Hz]

print(f'The Average data in channel Oz with 8 (Hz): {data_segment_8Hz}')
print(f'The Average data in channel Oz with 12 (Hz): {data_segment_12Hz}')

# Task2:

# get the time point from the 4D array
time_point_8Hz = _data[index_Oz, 0:750, : , index_8Hz]
time_point_12Hz = _data[index_Oz, 0:750, : , index_12Hz]

plt.subplot(3,2,1)
plt.plot(time_point_8Hz[:750], np.repeat(data_segment_8Hz, 750), label="8Hz")
plt.xlabel("Time") 
plt.ylabel("Average") 
plt.title("Average of 8Hz vs Time")

plt.subplot(3,2,2)
plt.plot(time_point_12Hz[:750], np.repeat(data_segment_12Hz, 750), label="12Hz")
plt.xlabel("Time") 
plt.ylabel("Average") 
plt.title("Average of 12Hz vs Time")
plt.show()

# Task3:

# Your existing code to calculate the mean of 8(Hz)
data_segment_8Hz = np.mean(_data[index_Oz, :, :, index_8Hz], axis=0)
plt.subplot(1,2,1)
# Calculate the spectrogram using plt.specgram
plt.specgram(data_segment_8Hz.flatten(), Fs=8)

# Plot the spectrogram
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('Spectrogram of Mean Data 8(Hz)')
plt.colorbar(label='Intensity (dB)')


# Your existing code to calculate the mean of 12(Hz)
data_segment_12Hz = np.mean(_data[index_Oz, :, :, index_12Hz], axis=0)
plt.subplot(1,2,2)
# Calculate the spectrogram using plt.specgram
plt.specgram(data_segment_12Hz.flatten(), Fs=8)

# Plot the spectrogram
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('Spectrogram of Mean Data 12(Hz)')
plt.colorbar(label='Intensity (dB)')
plt.show()

# Task4:

f1, power_spectrum_8Hz = signal.welch(data_segment_8Hz, fs=8, nperseg=250)
plt.subplot(1,2,1)
plt.plot(f1,power_spectrum_8Hz)
plt.title('psd of average_8Hz')
plt.xlabel('freq (Hz)')


f1, power_spectrum_12Hz = signal.welch(data_segment_12Hz, fs=8, nperseg=250)
plt.subplot(1,2,2)
plt.plot(f1,power_spectrum_12Hz)
plt.title('psd of average_12Hz')
plt.xlabel('freq (Hz)')
plt.show()

# Task5:

'''
Harmonics are currents or voltages with frequencies that are integer multiples of the fundamental frequency 
of the system. For example, if the fundamental frequency is 60 Hz, then the harmonics are 120 Hz, 180 Hz, 240 Hz, 
and so on. Harmonics are produced by non-linear loads, such as rectifiers, discharge lighting, or saturated electric 
machines, that distort the shape of the voltage or current waveform from a pure sine wave. Harmonics can cause 
various problems in the power system, such as overheating, interference, resonance, and reduced power factor.
The power spectrum is a representation of the frequency content of a signal, which shows how much power is present 
at each frequency. The power spectrum can be calculated by applying the discrete Fourier transform (DFT) to the signal, 
which decomposes the signal into a sum of sinusoidal waves with different frequencies and amplitudes. 
The power spectrum can be plotted as a line graph, where the x-axis is the frequency and the y-axis is the 
power spectral density (PSD), which is the power per unit frequency. The power spectrum can reveal the presence of 
harmonics in the signal, as they appear as peaks at the harmonic frequencies. in these two plots, we can see only 
one peak in each plot. so we have just one harmonic.
'''