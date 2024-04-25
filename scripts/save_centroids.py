import pandas as pd
import librosa
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pickle

def convert_label(label:str):
    conv_map = {"queen present or original queen":0,
                "queen not present":1,
                "queen present and rejected":2, 
                "queen present and newly accepted":3}
    num_label = conv_map[label]
    # total_labels[num_label]+=1
    return num_label
def spectral_centroid(bee,sr):
    FRAME_SIZE = 1024
    HOP_LENGTH = 512
    sc = librosa.feature.spectral_centroid(y=bee, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    onset_env = librosa.onset.onset_strength(y=bee, sr=sr)
    return sc, onset_env

# Read train.csv
train_df = pd.read_csv('C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\train.csv')

# Initialize lists to store results
results = {"centroids":[], "labels": [],"flux":[]}

# Loop over sound file names in train.csv
for index, row in tqdm(train_df.iterrows(), total=len(train_df), desc='Processing audio files'):
    filename = row['modified_file_name']  # Assuming 'filename' is the column name containing sound file names
    label = row['queen status']  # Assuming 'label' is the column name containing labels
    root = "C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\sound_files\\sound_files\\"

    file = os.path.join(root,filename)
    # ipd.Audio(os.path.join(root,filename))
    if os.path.exists(file):
        # print(file)
        bee, sr = librosa.load(file)
        # Call your function on the sound file
        sc, sf = spectral_centroid(bee,sr)

        # Append results to lists
        results["flux"].append(sf)
        results["centroids"].append(sc)
        results["labels"].append(label)
# import ipdb; ipdb.set_trace()

with open('centroids.pkl', 'wb') as f:
    pickle.dump(results, f)