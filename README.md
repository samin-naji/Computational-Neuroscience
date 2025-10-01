# Computational Neuroscience

This repository contains my exercises and projects for **Computational Neuroscience**, including signal processing, neural data analysis, and machine learning models applied to brain signals.

---

## Dependencies

The project uses the following Python libraries, organized by functionality:

- **File I/O and system**  
  - `scipy.io.loadmat` — Load MATLAB `.mat` files  
  - `os` — File and path utilities  

- **Data handling**  
  - `pandas` — Tabular data and annotations  

- **Numerical and transforms**  
  - `numpy.fft` — Fourier transforms  

- **Signal processing**  
  - `scipy.signal` (`spectrogram`, `stft`, `welch`, `hann`, `butter`, `filtfilt`) — Spectral estimation, filtering, and windowing  

- **Statistics**  
  - `scipy.stats` — Statistical tests and distributions  

- **Visualization**  
  - `matplotlib` — Plotting and figure generation  

- **Neural/EEG data tools**  
  - `mne` — EEG/MEG data handling and preprocessing  

- **Machine Learning**  
  - `sklearn` — Classical ML models, preprocessing, evaluation  

- **Deep Learning**  
  - `torch.nn` — Building neural networks  
  - `torch.optim` — Optimization algorithms  
  - `torch.utils.data.DataLoader` — Data pipeline utilities  
  - `torchvision.datasets`, `torchvision.transforms` — Datasets and preprocessing  
  - `torchvision.io.read_image` — Image loading  

