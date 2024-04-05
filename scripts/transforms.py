
import numpy as np 
import matplotlib.pyplot as plt
import os 
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import librosa
import IPython.display as ipd

root = 'C:\\Users\\adefr\\OneDrive\\Desktop\\bees\\sound_files\\sound_files\\'
# filename = '2022-06-05--17-41-01_2__segment0.wav'
# filename = '2022-06-05--18-40-08_2__segment0.wav'
filename = '2022-06-07--13-07-01_2__segment0.wav'



file = os.path.join(root,filename)
# ipd.Audio(os.path.join(root,filename))

bee, sr = librosa.load(file)

bee_ft = np.fft.fft(bee)
mag = np.abs(bee_ft)

def plot_mag_spectrum(signal,title,sr, f_ratio=1):
    ft = np.fft.fft(signal)
    magnitude_spectrum = np.abs(ft)

    plt.figure(figsize=(18,5))

    frequency = np.linspace(0,sr,len(magnitude_spectrum))
    # calculate number of bins
    num_freq_bins = int(len(frequency)*f_ratio)

    plt.plot(frequency[:num_freq_bins],magnitude_spectrum[:num_freq_bins])
    plt.xlabel("Frequency (Hz)")
    plt.title(title)

    plt.show()

def spectral_centroid():
    FRAME_SIZE = 1024
    HOP_LENGTH = 512
    sc = librosa.feature.spectral_centroid(y=bee, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    frames = range(len(sc))
    t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)
    plt.figure(figsize=(25,10))
    plt.plot(t, sc, color='r')
    plt.title("Spectral Centroid of Bee Audio Sample")
    plt.xlabel("Time (s)")
    plt.ylabel("Spectral Centroid (Hz)")
    plt.show()

    sb = librosa.feature.spectral_bandwidth(y=bee, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    plt.title("Spectral Bandwidth of Bee Audio Sample")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    # plt.figure(figsize=(25,10))
    plt.plot(t, sb, color='b')
    plt.show()

def mel_filter():
    filter_banks = librosa.filters.mel(n_fft=2048, sr=sr, n_mels=10)
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(filter_banks, 
                            sr=sr, 
                            x_axis="linear")
    plt.colorbar(format="%+2.f")
    plt.show()

def mel_spectro():
    # import ipdb; ipdb.set_trace()
    # mel_spectrogram = librosa.feature.melspectrogram(y=bee, sr=sr, n_fft=2048, hop_length=512, n_mels=5)
    
    # log_mel_spectrogram = librosa.power_to_db(mel_spectrogram,ref=np.max)
    # plt.figure(figsize=(25, 10))
    # librosa.display.specshow(log_mel_spectrogram, 
    #                         x_axis="time",
    #                         y_axis="mel", 
    #                         sr=sr)
    # plt.colorbar(format="%+2.f")
    # plt.show()

    mel_spect = librosa.feature.melspectrogram(y=bee, sr=sr, n_fft=2048, hop_length=1024)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

def spectral_flux():
    onset_env = librosa.onset.onset_strength(y=bee, sr=sr)
    plt.figure(figsize=(20, 4))
    plt.plot(onset_env)
    plt.grid()
    plt.title('Spectral flux')
    plt.xlabel('time')
    plt.ylabel('Onset')
    plt.show()
def spectral_rolloff():
    S, phase = librosa.magphase(librosa.stft(bee))
    plt.figure(figsize=(20, 4))
    plt.plot(librosa.feature.spectral_rolloff(S=S, sr=sr).squeeze())
    plt.grid()
    plt.title('Roll off frequency')
    plt.xlabel('time')
    plt.ylabel('Hz')
    plt.show()

# Frequency bins as X, Magnitude on Y
# 0.1 shows different frequencies involved in shaping bee sound, harmonics
plot_mag_spectrum(bee,"Magnitude Spectrum for Bee Audio",sr,.0025)
spectral_centroid()
mel_filter()
mel_spectro()
spectral_flux()
spectral_rolloff()
