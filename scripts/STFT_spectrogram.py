import librosa as librosa
import numpy as np
import matplotlib.pyplot as plt

import librosa.display
import os


root = 'C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\sound_files\\sound_files\\'
filename = root + '2022-06-05--17-41-01_2__segment0.wav'

# Size of the FFT, which will also be used as the window length
n_fft = 2048

# Step or stride between windows. If the step is smaller than the window lenght, the windows will overlap
hop_length = 512

# Load sample audio file
y, sr = librosa.load(filename)

# Calculate the spectrogram as the square of the complex magnitude of the STFT
spectrogram_librosa = (
    np.abs(
        librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window="hann",
            pad_mode="reflect",
        )
    )
    ** 2
)

def show_spectrogram(spec, title, sr, hop_length, y_axis="log", x_axis="time"):
    librosa.display.specshow(spec, sr=sr, y_axis=y_axis, x_axis=x_axis, hop_length=hop_length)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.show()
spectrogram_librosa_db = librosa.power_to_db(spectrogram_librosa, ref=np.max)


show_spectrogram(spectrogram_librosa_db, "Bee Sample Audio Spectrogram", sr, hop_length)