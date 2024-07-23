from scipy.io import loadmat
from pylab import * 
from math import sqrt
import numpy as np
import pandas as pd

data = loadmat('D:/computational_neuroscience/EEG_P2090_processed.mat')
print(data.keys())

Number_of_Channels = data['EEG_P2090_processed'].shape[0]
print(f'The number of channels is {Number_of_Channels}')

Number_of_Samples = data['EEG_P2090_processed'].shape[1]
print(f'The number of samples is {Number_of_Samples}')

Sampling_Frequency = 500 #Hz
Sample_Time = 1 / Sampling_Frequency
print(f'The sample time is {Sample_Time}')

Duration_of_Recording = Number_of_Samples * Sample_Time
print(f'The duration of the EEG recording is {Duration_of_Recording} seconds')
print(f'The duration of the EEG recording is {Duration_of_Recording/60} minutes')

channel_input = int(input('please enter a channel: '))

EEG = data['EEG_P2090_processed']

# Determine the number of channels and time points
num_channels, num_time_points = EEG.shape

# Create a time array based on the sampling rate and the number of time points
time = np.arange(num_time_points) / Sample_Time

# Plot EEG channel
plt.figure(figsize=(12, 6))
plt.plot(time, EEG[channel_input, :], label=f'Channel {channel_input}')

plt.title(f'EEG Data for {channel_input}')
plt.xlabel('Time (s)')
plt.ylabel('EEG Amplitude')
plt.legend()
plt.grid(True)
plt.show()

start_time , end_time = [int(x) for x in input("Please enter the start & end times (seconds): ").split()]
time_input = np.arange( start_time , end_time ) 
EEG_segment = EEG[start_time:end_time]
mean = np.mean(EEG_segment) 
std = np.std(EEG_segment)
print(f"The mean of the EEG values for the time segment is {mean} microvolts.") 
print(f"The standard deviation of the EEG values for the time segment is {std} microvolts.")

EEG_mean = EEG[channel_input, :][start_time:end_time].mean()
print(f"The mean of EEG is: {EEG_mean}")

EEG_standard_deviation = EEG[channel_input, :][start_time:end_time].std()
print(f"The standard deviation of EEG is: {EEG_standard_deviation}")

EEG_median = np.median(EEG[channel_input, :][start_time:end_time])
print(f"The median of EEG is: {EEG_median}")


EEG_range = EEG[channel_input, :][start_time:end_time]
EEG_range.sort()
print(f"The range of EEG is: {EEG_range}")

data_frame_list = [EEG_mean, EEG_standard_deviation, EEG_median, EEG_range] 
df = pd.DataFrame({
    "Mean": [EEG_mean],
    "STD": [EEG_standard_deviation],
    "MEDIAN": [EEG_median],
    "EEG_RANGE": [EEG_range]
}) 
print(df)

df.to_csv('D:\\computational_neuroscience\\EEG.csv', sep = ",")