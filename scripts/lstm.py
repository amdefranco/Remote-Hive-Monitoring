import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import random_split
from PIL import Image
import pandas as pd
import wandb
from sklearn.metrics import confusion_matrix
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
import random 
import pickle

class CustomDataset(Dataset):
    def __init__(self, file_path, sequence_length):
        # Load data from pickle file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        centroids = np.array(data['centroids'])
        flux = np.array(data['flux'])
        self.labels = data['labels']
        self.sequence_length = sequence_length
        self.centroids = (centroids - centroids.mean()) / centroids.std()
        self.flux = (flux - flux.mean()) / flux.std()
        
        # 2584
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Get centroids, flux, and label for the given index
        centroids = self.centroids[idx]
        flux = self.flux[idx]
        label = self.labels[idx]
        
        # Divide sequences into segments of length sequence_length
        centroids_segments = self._divide_into_segments(centroids, self.sequence_length)
        flux_segments = self._divide_into_segments(flux, self.sequence_length)
        x = np.array([centroids_segments, flux_segments])
        return torch.from_numpy(x).float(), label
    
    def _divide_into_segments(self, sequence, length):
        num_segments = len(sequence) // length
        segments = [sequence[i*length:(i+1)*length] for i in range(num_segments)]
        
        # Pad the last segment if necessary
        
        if len(sequence) % length != 0:
            padding = [0] * (length - len(sequence) % length)
            segments.append(sequence[-length:] + padding)
        
        return segments
        return np.array([np.array(self.centroids[idx]), np.array(self.flux[idx])]), self.labels[idx]



class AudioLSTM(nn.Module):
    def __init__(self, input_size_centroids, input_size_flux, hidden_size, num_classes):
        super(AudioLSTM, self).__init__()
        
        # LSTM for spectral centroids
        self.lstm_centroids = nn.LSTM(input_size=input_size_centroids, hidden_size=hidden_size, batch_first=True)
        
        # LSTM for spectral flux
        self.lstm_flux = nn.LSTM(input_size=input_size_flux, hidden_size=hidden_size, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(17408, 64)  # Concatenating outputs of both LSTMs
        
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        # LSTM for spectral centroids
        # bs, _, n  = x.shape
        x_centroids = x[:,0,:]
        x_flux = x[:,1,:]
        # x_flux = x_flux.reshape((bs,n,1))
        # x_centroids = x_flux.reshape((bs,n,1))
        out_cent, (h_n_centroids, _) = self.lstm_centroids(x_centroids)
        
        # LSTM for spectral flux
        out_flux, (h_n_flux, _) = self.lstm_flux(x_flux)
        
        # Concatenate the last hidden states of both LSTMs
        out = torch.cat((out_cent.reshape((x.shape[0],-1)), out_flux.reshape((x.shape[0],-1))), dim=1)
        
        # Fully connected layers
        out = torch.sigmoid(self.fc1(out))
        out = self.fc2(out)
        
        
        return out


# Define input sizes and other hyperparameters
input_size_centroids = input_size_flux = 152
hidden_size = 64
num_classes = 4


def transform(sample):
    centroids, flux, label = sample
    # Normalize centroids and flux
   
    return centroids, flux, label

hidden_sizes = [512]
batch_sizes = [1028]
epochs_list = [1]
epochs = 1000
lrs= [0.000005]
layers=[1]
num_layers=5
for hidden_size in hidden_sizes:  
    for batch_size in batch_sizes:
        for learning_rate in lrs:
            # start a new wandb run to track this script
            wandb.init(
                # set the wandb project where this run will be logged
                project="Honey_Bees",

                # track hyperparameters and run metadata
                config = {'batch_size': batch_size, 
                        'learning_rate': learning_rate, 
                        'epochs': epochs},
                name= "LSTM_"   + "_" + str(batch_size) + "_" + str(learning_rate)[2:] + "_" + str(hidden_size) + "_" + str(num_layers)
                    
            )
            # # Create custom dataset and DataLoader
            # dataset = CustomDataset(csv_file='C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\train.csv', transform=transform)
            # Example usage in training loop
            # model = CRNN( input_size=(3,224,224), num_classes=4,hidden_size=hidden_size
            model = AudioLSTM(input_size_centroids, input_size_flux, hidden_size, num_classes=4)
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)

            print("Created model")
            criterion = torch.nn.CrossEntropyLoss()
            criterion = criterion.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            path = 'C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\scripts\\centroids.pkl'
            dataset = CustomDataset(path,input_size_centroids)

            # Define the sizes of train and test sets
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size

            # Split the dataset into train and test sets
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

            # Define data loaders for train and test sets
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            # optimizer = optimizer.to(device)
            losses=[]
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                for inputs, labels in train_loader:
                    
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    losses.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                avg_loss = running_loss / len(train_loader)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")
                wandb.log({'epoch': epoch, 'loss': avg_loss})
                if (epoch % 10 == 0):
                    # Validation loop
                    model.eval()
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for inputs, labels in val_loader:
                            inputs = inputs.to(device)
                            labels = labels.to(device)
                            outputs = model(inputs)
                            # import ipdb; ipdb.set_trace()
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                        avg_loss = running_loss / len(val_loader)
                        accuracy = 100 * correct / total
                        wandb.log({'val_loss': avg_loss, 'val_accuracy': accuracy})
                    accuracy = correct / total
                    print(f"Validation Accuracy: {accuracy}")
            # Validation loop
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                avg_loss = running_loss / len(val_loader)
                accuracy = 100 * correct / total
                wandb.log({'final_val_loss': avg_loss, 'final_val_accuracy': accuracy})
            accuracy = correct / total
            print(f"Validation Accuracy: {accuracy}")
           

            torch.save(model.state_dict(), "models2\\LSTM_"   + "_" + str(batch_size) + "_" + str(learning_rate)[2:] + "_" + str(hidden_size) + ".pth")
            wandb.finish()

        def create_confusion_matrix(model, test_loader):
            model.eval()
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    labels = targets
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    predicted = predicted.cpu()
                    all_preds.extend(predicted.numpy())
                    all_targets.extend(targets.numpy())
            cm = confusion_matrix(all_targets, all_preds)
            return cm

        # Get predictions and targets from the test set
        conf_matrix = create_confusion_matrix(model, val_loader)

        # Display the confusion matrix
        print("Confusion Matrix:")
        print(conf_matrix)


        # Create confusion matrix plot
        def plot_confusion_matrix(conf_matrix, classes):
            conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix_norm, annot=True, fmt=".2%", cmap="Blues", xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted Labels')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix (Normalized)')
            plt.show()


        plot_confusion_matrix(conf_matrix, classes=["Present/Original", "Not Present", "Rejected", "Accepted"])
