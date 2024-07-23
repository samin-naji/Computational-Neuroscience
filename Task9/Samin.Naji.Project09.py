import numpy as np
import mne
from scipy.io import loadmat
from scipy.signal import welch
from scipy.stats import ttest_ind
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load the EEG data from .mat files
healthy_data1 = loadmat('EEG DATA/EEG DATA/H1.mat')
healthy_data2 = loadmat('EEG DATA/EEG DATA/H2.mat')
healthy_data3 = loadmat('EEG DATA/EEG DATA/H3.mat')

schizophrenia_data1 = loadmat('EEG DATA/EEG DATA/Sch1.mat')
schizophrenia_data2 = loadmat('EEG DATA/EEG DATA/Sch2.mat')
schizophrenia_data3 = loadmat('EEG DATA/EEG DATA/Sch3.mat')


print(healthy_data1)

# Create MNE Raw objects
sfreq = 250
CH_NUMBER = 19

healthy_raw1 = mne.io.RawArray(healthy_data1.get(
    'H1'), mne.create_info(CH_NUMBER, sfreq, ch_types='eeg'))
healthy_raw2 = mne.io.RawArray(healthy_data2.get(
    'H2'), mne.create_info(CH_NUMBER, sfreq, ch_types='eeg'))
healthy_raw3 = mne.io.RawArray(healthy_data3.get(
    'H3'), mne.create_info(CH_NUMBER, sfreq, ch_types='eeg'))

schizophrenia_raw1 = mne.io.RawArray(schizophrenia_data1.get(
    'Sch1'), mne.create_info(CH_NUMBER, sfreq, ch_types='eeg'))
schizophrenia_raw2 = mne.io.RawArray(schizophrenia_data2.get(
    'Sch2'), mne.create_info(CH_NUMBER, sfreq, ch_types='eeg'))
schizophrenia_raw3 = mne.io.RawArray(schizophrenia_data3.get(
    'Sch3'), mne.create_info(CH_NUMBER, sfreq, ch_types='eeg'))

# Concatenate the datasets
healthy_raw = mne.concatenate_raws([healthy_raw1, healthy_raw2, healthy_raw3])
schizophrenia_raw = mne.concatenate_raws(
    [schizophrenia_raw1, schizophrenia_raw2, schizophrenia_raw3])

# Apply bandpass filter to remove high-frequency noise
healthy_raw.filter(l_freq=1, h_freq=50)  # Adjust the frequency range as needed
schizophrenia_raw.filter(l_freq=1, h_freq=50)

# Remove power line interference (50 Hz)
healthy_raw.notch_filter(freqs=50)
schizophrenia_raw.notch_filter(freqs=50)


# Divide into 2-second segments
duration = 2  # in seconds
healthy_epochs = mne.make_fixed_length_epochs(healthy_raw, duration=duration)
schizophrenia_epochs = mne.make_fixed_length_epochs(
    schizophrenia_raw, duration=duration)


# Save preprocessed data to new files if needed

# healthy_raw.save('preprocessed_healthy_raw.fif', overwrite=True)
# schizophrenia_raw.save('preprocessed_schizophrenia_raw.fif', overwrite=True)


# Define frequency bands
freq_bands = {'theta': (4, 8),
              'alpha': (8, 13),
              'beta': (13, 30)}

# Function to calculate band power


def calculate_band_power(data, sfreq, freq_band):
    psd, freqs = mne.time_frequency.psd_array_welch(
        data, sfreq=sfreq, fmin=freq_band[0], fmax=freq_band[1])
    return np.sum(psd, axis=-1)

# Function to extract features from each segment


def extract_features(epochs):
    global sfreq
    features = []

    for epoch in epochs:
        data = epoch  # Get the EEG data for the epoch
        sfreq = sfreq

        # Calculate mean and standard deviation
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)

        # Calculate total signal power
        total_power = np.sum(data ** 2, axis=-1, keepdims=True)

        # Calculate band powers
        alpha_power = calculate_band_power(data, sfreq, freq_bands['alpha'])
        beta_power = calculate_band_power(data, sfreq, freq_bands['beta'])
        theta_power = calculate_band_power(data, sfreq, freq_bands['theta'])

        # Flatten the band powers
        alpha_power = alpha_power.reshape(alpha_power.shape[0], -1)
        beta_power = beta_power.reshape(beta_power.shape[0], -1)
        theta_power = theta_power.reshape(theta_power.shape[0], -1)

        # Calculate Fourier series coefficients and flatten them
        fourier_coeffs = np.abs(np.fft.fft(
            data, axis=-1)).reshape(data.shape[0], -1)

        # Append features for the current segment
        segment_features = np.concatenate(
            [mean, std, total_power, alpha_power, beta_power, theta_power, fourier_coeffs], axis=-1)
        features.append(segment_features)

    return np.array(features)


# Extract features for healthy and schizophrenia segments
"""
healthy_features = extract_features(healthy_epochs)
schizophrenia_features = extract_features(schizophrenia_epochs)
"""

# Assuming healthy_epochs and schizophrenia_epochs are MNE Epochs objects       
# Extract the data from epochs
healthy_data = healthy_epochs.get_data()
schizophrenia_data = schizophrenia_epochs.get_data()


# Create an array to store the labels (0 for healthy, 1 for schizophrenia)
num_channels = healthy_data.shape[1]
healthy_labels = np.zeros((len(healthy_epochs), 1, 1))  # Adjust the shape to (num_epochs, 1, 1)
schizophrenia_labels = np.ones((len(schizophrenia_epochs), 1, 1))  # Adjust the shape to (num_epochs, 1, 1)

# Ensure the shape of labels matches the number of epochs
healthy_labels_broadcasted = np.tile(healthy_labels, (1, num_channels, healthy_data.shape[2]))
schizophrenia_labels_broadcasted = np.tile(schizophrenia_labels, (1, num_channels, schizophrenia_data.shape[2]))

# Concatenate the data and labels along the last dimension
healthy_data_with_labels = np.concatenate([healthy_data, healthy_labels_broadcasted], axis=-1)
schizophrenia_data_with_labels = np.concatenate([schizophrenia_data, schizophrenia_labels_broadcasted], axis=-1)

# Concatenate the data from both groups along the first axis
all_data = np.concatenate([healthy_data_with_labels, schizophrenia_data_with_labels], axis=0)

# Create an array to store labels
labels = np.concatenate([np.zeros(len(healthy_epochs)), np.ones(len(schizophrenia_epochs))])

# Create an array to store t-test results
t_test_results = np.zeros((num_channels, 7, all_data.shape[2] - 1))  # 7 features in total, excluding the label column

# Iterate over channels
for channel in range(num_channels):
    # Iterate over features
    for feature in range(all_data.shape[2] - 1):  # Exclude the label column
        # Extract data for the current channel and feature
        healthy_channel_feature_data = all_data[:len(healthy_epochs), channel, feature]
        schizophrenia_channel_feature_data = all_data[len(healthy_epochs):, channel, feature]

        # Perform t-tests for each time point
        _, t_test_results[channel, 0, feature] = ttest_ind(healthy_channel_feature_data, schizophrenia_channel_feature_data, equal_var=False)  # Mean
        _, t_test_results[channel, 1, feature] = ttest_ind(healthy_channel_feature_data.std(), schizophrenia_channel_feature_data.std(), equal_var=False)  # Standard deviation

        # Power spectral density using Welch method
        _, psd_healthy = welch(healthy_channel_feature_data, fs=sfreq)
        _, psd_schizophrenia = welch(schizophrenia_channel_feature_data, fs=sfreq)

        _, t_test_results[channel, 2, feature] = ttest_ind(np.sum(psd_healthy), np.sum(psd_schizophrenia), equal_var=False)  # Total signal power
        _, t_test_results[channel, 3, feature] = ttest_ind(np.sum(psd_healthy[8:13]), np.sum(psd_schizophrenia[8:13]), equal_var=False)  # Alpha band power
        _, t_test_results[channel, 4, feature] = ttest_ind(np.sum(psd_healthy[13:30]), np.sum(psd_schizophrenia[13:30]), equal_var=False)  # Beta band power
        _, t_test_results[channel, 5, feature] = ttest_ind(np.sum(psd_healthy[4:8]), np.sum(psd_schizophrenia[4:8]), equal_var=False)  # Theta band power

        # Fourier series coefficients (assuming you have them in your data)
        _, t_test_results[channel, 6, feature] = ttest_ind(np.abs(np.fft.fft(healthy_channel_feature_data)[1]), np.abs(np.fft.fft(schizophrenia_channel_feature_data)[1]), equal_var=False)

# Print or use t_test_results for further analysis
print(t_test_results)


# # Create an array to store the labels (0 for healthy, 1 for schizophrenia)
# num_channels = healthy_data.shape[1]
# healthy_labels = np.zeros((len(healthy_epochs), 1, 1))  # Adjust the shape to (num_epochs, 1, 1)
# schizophrenia_labels = np.ones((len(schizophrenia_epochs), 1, 1))  # Adjust the shape to (num_epochs, 1, 1)

# # Ensure the shape of labels matches the number of epochs
# healthy_labels_broadcasted = np.tile(healthy_labels, (1, num_channels, healthy_data.shape[2]))
# schizophrenia_labels_broadcasted = np.tile(schizophrenia_labels, (1, num_channels, schizophrenia_data.shape[2]))

# # Concatenate the data and labels along the last dimension
# healthy_data_with_labels = np.concatenate([healthy_data, healthy_labels_broadcasted], axis=-1)
# schizophrenia_data_with_labels = np.concatenate([schizophrenia_data, schizophrenia_labels_broadcasted], axis=-1)

# # Concatenate the data from both groups along the first axis
# all_data = np.concatenate([healthy_data_with_labels, schizophrenia_data_with_labels], axis=0)

# # Create an array to store labels
# labels = np.concatenate([np.zeros(len(healthy_epochs)), np.ones(len(schizophrenia_epochs))])

# # Create an array to store t-test results
# t_test_results = np.zeros((num_channels, all_data.shape[2] - 1))  # -1 to exclude the label column

# # Iterate over channels and features
# for channel in range(num_channels):
#     for feature in range(all_data.shape[2] - 1):  # Exclude the label column
#         # Extract data for the current channel and feature
#         healthy_channel_feature_data = all_data[:len(healthy_epochs), channel, feature]
#         schizophrenia_channel_feature_data = all_data[len(healthy_epochs):, channel, feature]

#         # Perform independent samples t-test
#         t_stat, p_value = ttest_ind(healthy_channel_feature_data, schizophrenia_channel_feature_data)

#         # alpha = 0.05

#         # if p_value < alpha:
#         #     print("Reject the null hypothesis; there is a significant difference between the sample mean and the hypothesized population mean.")
#         # else:
#         #     print("Fail to reject the null hypothesis; there is no significant difference between the sample mean and the hypothesized population mean.")

#         # Store the p-value in the t_test_results array
#         t_test_results[channel, feature] = p_value

# # Concatenate the t-test results with the original data
# features_matrix = all_data[:, :, :-1].reshape(all_data.shape[0], -1)

# # Create column names for channels and features

# column_names = [f'Channel_{ch}_Epoch_{ep}_Feature_{f}' for ep in range(all_data.shape[0]) for ch in range(num_channels) for f in range(all_data.shape[2] - 1)]

# # Convert the features_matrix to a Pandas DataFrame
# df = pd.DataFrame(features_matrix, columns=column_names)

# Add labels to the feature matrix
# df['Label'] = labels

# Display the DataFrame
# print(df)

"""
# Define the feature columns and label column
feature_columns = df.columns[:-1]
label_column = 'Label'

# Extract features and labels
X = df[feature_columns].values
y = df[label_column].values

# Split the data into training and testing sets (hold-out)
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize the classifier (you can replace this with your chosen model)
classifier = RandomForestClassifier(random_state=42)

# Train the model on the training set
classifier.fit(X_train, y_train)

# Make predictions on the hold-out set
y_holdout_pred = classifier.predict(X_holdout)

# Evaluate the performance on the hold-out set
accuracy_holdout = accuracy_score(y_holdout, y_holdout_pred)
print(f'Accuracy on Hold-Out Set: {accuracy_holdout:.4f}')

# Perform cross-validation on the remaining data
cv_scores = cross_val_score(classifier, X_train, y_train, cv=5)

# Print cross-validation scores
print('Cross-Validation Scores:')
for i, score in enumerate(cv_scores, start=1):
    print(f'Fold {i}: {score:.4f}')

# Print average cross-validation score
print(f'Average Cross-Validation Score: {np.mean(cv_scores):.4f}')

"""