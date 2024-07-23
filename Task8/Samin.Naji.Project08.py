import mne
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram, stft


# Load the EDF file
EEG_data_file_path = 'D:\\computational_neuroscience\\assignment8\\dataverse_files\\h01.edf'

raw = mne.io.read_raw_edf(EEG_data_file_path, preload=True)

raw = raw.pick_types(meg=False, eeg=True, eog=False, exclude='bads')

# Apply preprocessing steps
# 1. High-pass filter
raw.filter(l_freq=1.0, h_freq=None)

# 2. Low-pass filter
raw.filter(l_freq=None, h_freq=40.0)

# 3. Notch filter to remove power line noise (50 Hz and 60 Hz)
raw.notch_filter(freqs=[50, 60], picks='all')

# 4. Remove bad channels (if any)
# Remove bad channels
raw.interpolate_bads()

# 5. Independent Component Analysis (ICA) for removing artifacts
ica = mne.preprocessing.ICA(n_components=19, random_state=97, max_iter=800)
ica.fit(raw)
raw_corrected = raw.copy()
ica.apply(raw_corrected)

# Plot the data after preprocessing
# raw_corrected.plot()
# plt.show()

# Save the preprocessed data

# preprocessed_file_path = './pre_data.fif'
# raw_corrected.save(preprocessed_file_path, overwrite=True)

# get sfreq from EDF file
sfreq = raw_corrected.info.get('sfreq')

# convert data into numpy array
raw_corrected_data = raw_corrected.get_data()

# get number of channels
num_channels = raw_corrected_data.shape[0]

# filter data between 1 - 40 Hz ( base on data info )
freqs , times, Zxx = stft(raw_corrected_data, fs=sfreq, nperseg=256, noverlap=128)
fmin, fmax = 1, 40
freq_mask = (freqs >= fmin) & (freqs <= fmax)
Zxx = Zxx[:, freq_mask, :]


# num_rows = 4
# num_cols = 5

# fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 10))


# # freqs, times, Sxx = mne.time_frequency.(raw_corrected_data, fmin=1, fmax=40, n_fft=256, n_overlap=128, sfreq=sfreq)
# # plt.figure(figsize=(10, 6))

# for i in range(num_rows):
#     for j in range(num_cols):
#         chn_index = i * num_cols + j
#         if chn_index < num_channels:
#             ax = axes[i, j]
#             pcm = ax.pcolormesh(times, freqs[freq_mask], 10 * np.log10(np.abs(Zxx[chn_index])), cmap='viridis', shading='auto')
#             ax.set_ylabel('Frequency (Hz)')
#             ax.set_xlabel('Time (s)')
#             ax.set_title(f"Spectrogram of EEG Channel {chn_index + 1}")
#             fig.colorbar(pcm, ax=ax, label="Power (dB)")
#             plt.tight_layout()
#             fig.savefig(f'./spectrogrum_data_of_h0/spectrogram_channel_{i + 1}.png')
#             plt.close()
#     # ax._colorbars(label="Power (dB)")

# plt.tight_layout()
# plt.show()

# Create subplots in a rectangular grid
num_cols = 4  # Choose the number of columns for the grid
num_rows = int(np.ceil(num_channels / num_cols))
channel_list_names = raw_corrected.info.ch_names

# Plot spectrograms and save as PNG files

for i in range(num_channels):
    row_idx = i // num_cols
    col_idx = i % num_cols
    fig, ax = plt.subplots(figsize=(7, 3))
    pcm = ax.pcolormesh(times, freqs[freq_mask], 10 * np.log10(np.abs(Zxx[i])), cmap='viridis', shading='auto')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f"Spectrogram of EEG Channel {channel_list_names[i]}")
    fig.colorbar(pcm, ax=ax, label="Power (dB)")
    plt.tight_layout()
    fig.savefig(f'./spectrogrum_data_of_h0/spectrogram_channel_{channel_list_names[i]}.png')
    plt.close()  # Close the figure to release resources
