# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import numpy as np
import pandas as pd 
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import librosa
import librosa.display
import re

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
         os.path.join(dirname, filename)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# FEATURE EXTRACTION FUNCTIONS
    
def raw_feature_fromSample( path_audio_sample, feature2extract ):
        
    audio_sample, sr = librosa.core.load(path_audio_sample)
    
    m = re.match(r"\w+s(\d+)", feature2extract)
    n_freqs=int(m.groups()[0])
    
    Melspec = librosa.feature.melspectrogram(y=audio_sample, n_mels = n_freqs) # computes mel spectrograms from audio sample, 
    
    if 'LOG' in feature2extract: #'LOG_MELfrequencies48'
        Melspec=librosa.feature.melspectrogram(audio_sample, sr=sr, n_mels=n_freqs)
        x=librosa.power_to_db(Melspec+1)
        
    elif 'MFCCs' in feature2extract:
        n_freqs = int(feature2extract[5:len(feature2extract)])
        Melspec = librosa.feature.melspectrogram(y=audio_sample, sr=sr)
        x = librosa.feature.mfcc(S=librosa.power_to_db(Melspec),sr=sr, n_mfcc = n_freqs)
        
    else:
        x = Melspec

    return x

path_sound='/kaggle/input/beehive-sounds/sound_files/sound_files'+os.sep
full_list = os.listdir(path_sound)
os.makedirs('/kaggle/working/MFCCs')

for i, sample in enumerate(full_list):
    prefix = sample[:-4]
    raw_feature = 'MFCCs20'
    high_level_features = 0
    result = [j for j in full_list if j.startswith(prefix)]
    for filename in result:
        # raw feature extraction:
        x = raw_feature_fromSample(path_sound+filename, raw_feature ) # x.shape: (4, 20, 2584)
        x_norm = x
        
        if high_level_features:
            # high level feature extraction:
            if 'MFCCs' in raw_feature:
                X = compute_statistics_overMFCCs(x_norm, 'yes') # X.shape: (4 , 120)
            else: 
                X = compute_statistics_overSpectogram(x_norm)

            feature_map=X
        else:
            feature_map=x_norm
            
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        p = librosa.display.specshow(feature_map, ax=ax, y_axis='log', x_axis='time')

        fig.savefig('/kaggle/working/MFCCs/' + filename[:-4] + '.png')

