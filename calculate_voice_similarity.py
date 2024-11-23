import librosa
from fastdtw import fastdtw
import numpy as np

# Load the two audio signals
audio1, sr1 = librosa.load('E:/1_BTech/Major_Project/FINAL/cloned_voice.wav', sr=None)  # Use None to preserve the native sampling rate
audio2, sr2 = librosa.load('E:/1_BTech/Major_Project/FINAL/uploads/recorded_audio_converted.wav', sr=None)

# Extract feature vectors (e.g., MFCC)
mfcc1 = librosa.feature.mfcc(y=audio1, sr=sr1)
mfcc2 = librosa.feature.mfcc(y=audio2, sr=sr2)

# Compute Dynamic Time Warping (DTW) distance
distance, _ = fastdtw(mfcc1.T, mfcc2.T)  # Use transpose to compare the time axis
print(f"DTW Distance: {distance}")
