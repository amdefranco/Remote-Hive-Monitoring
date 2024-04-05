import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Root directory containing the audio files
root = 'C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\sound_files\\sound_files\\'

# Folder to save spectrogram images
save_folder = 'imgs/'

# Size of the FFT, which will also be used as the window length
n_fft = 2048

# Step or stride between windows. If the step is smaller than the window length, the windows will overlap
hop_length = 512

# Function to calculate and save spectrogram
def save_spectrogram(filename, save_folder, n_fft=2048, hop_length=512):
    # Load sample audio file
    y, sr = librosa.load(filename)

    # Calculate the spectrogram as the square of the complex magnitude of the STFT
    spectrogram = (
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

    # Convert to dB scale
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    # Plot and save spectrogram
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(spectrogram_db, sr=sr, y_axis='log', x_axis='time', hop_length=hop_length)
    plt.title('Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    
    # Create save directory if it does not exist
    os.makedirs(save_folder, exist_ok=True)

    # Save spectrogram as image file
    # Plot and save spectrogram
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(spectrogram_db, sr=sr, y_axis='log', x_axis='time', hop_length=hop_length)
    plt.axis('off')  # Turn off axis
    plt.savefig(os.path.join(save_folder, os.path.splitext(os.path.basename(filename))[0] + '.png'), bbox_inches='tight', pad_inches=0)  # Save with tight bounding box
    plt.close()

# Loop through root directory
for dirpath, _, filenames in os.walk(root):
    for filename in filenames:
        if filename.endswith('.wav'):
            # Construct full path to the audio file
            full_path = os.path.join(dirpath, filename)
            
            # Process and save spectrogram
            save_spectrogram(full_path, save_folder, n_fft, hop_length)
