from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

# task 2
def _mean(arr):
    return sum(arr) / len(arr)

def _std(arr):
    x=0
    for i in arr:
        s = (i - _mean(arr))**2
        x += s
    x = x / len(arr)   
    return sqrt(x)

def _median(arr):
    arr.sort()
    if len(arr) % 2 == 1:
        return arr[(len(arr)+1)/2]
    else: 
        return (arr[len(arr)/2+1] + arr[len(arr)/2])/2
    
def _var(arr):
    x=0
    for i in arr:
        s = (i - _mean(arr))**2
        x += s
    x = x / (len(arr)-1)   
    return x

# task 1
data = loadmat('D:/computational_neuroscience/EEG_P2090.mat')
print(data.keys())

EEG = data['EEG_P2090_processed']

Sampling_Frequency = 500 #Hz
Sample_Time = 1 / Sampling_Frequency

num_channels, num_time_points = EEG.shape

time = np.arange(num_time_points) / Sample_Time

time_range_1 = time[30 * Sampling_Frequency : 40 * Sampling_Frequency]  # 30-40 seconds
time_range_2 = time[150 * Sampling_Frequency : 160 * Sampling_Frequency]  # 150-160 seconds
last_5_seconds = time[-5 * Sampling_Frequency:]

plt.subplot(1, 3, 1)
plt.plot(time_range_1, EEG[0, 30 * Sampling_Frequency:40 * Sampling_Frequency])
plt.title('EEG for channel 1, 30-40(s)')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (uV)')

plt.subplot(1, 3, 2)
plt.plot(time_range_2, EEG[0, 150 * Sampling_Frequency: 160 * Sampling_Frequency])
plt.title('EEG for channel 1, 150-160(s)')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (uV)')

plt.subplot(1, 3, 3)
plt.plot(last_5_seconds, EEG[0, -5 * Sampling_Frequency:])
plt.title('EEG for channel 1, last 5(s)')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (uV)')

plt.show()