
import torch.nn as nn
import torch
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
# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224
    transforms.ToTensor(),           # Convert PIL image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize image
])
 
def convert_label(label:str):
    conv_map = {"queen present or original queen":0,
                "queen not present":1,
                "queen present and rejected":2, 
                "queen present and newly accepted":3}
    num_label = conv_map[label]
    return num_label
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.valid_indices = []  # List to store indices of valid samples
        
        # Check for valid samples
        for idx, row in self.data.iterrows():
            img_name = row['file name']
            img_name = "C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\scripts\\spectrograms\\" + img_name
            img_name = img_name[:-4] + ".png"
            if os.path.exists(img_name):
                self.valid_indices.append(idx)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        idx = self.valid_indices[idx]  # Get the index of valid sample
        img_name = self.data.iloc[idx, 0]  # Assuming "file name" is the first column
        label = self.data.iloc[idx, 1]  # Assuming "queen states" is the second column
        label = convert_label(label)
        img_name = "C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\scripts\\spectrograms\\" + img_name
            
        img_name = img_name[:-4] + ".png"
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
dataset = CustomDataset(csv_file='C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\train.csv', transform=transform)
class CRNN(nn.Module):
    def __init__(self, num_classes=4, hidden_size=256, num_layers=2):
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.rnn = nn.LSTM(input_size=32 * 56 * 56, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(batch_size, -1, 32 * 56 * 56)  # reshape for the RNN
        # Forward pass through RNN layer
        h_0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        c_0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        x, _ = self.rnn(x, (h_0, c_0))
        # Reshape output from RNN to fit into fully connected layer
        x = x[:, -1, :]  # get the last timestep output
        x = self.fc(x)
        return x
# Define the size of train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Split dataset into train and validation sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoader for train and validation sets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# Example usage in training loop
# model = CRNN( input_size=(3,224,224), num_classes=4,hidden_size=hidden_size)
model = CRNN(num_classes=4,hidden_size=256,num_layers=256)

# Specify the path to your saved model
PATH = 'C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\scripts\\models\\CRNN__64_0005_256.pth'
# Initialize your model (make sure it has the same architecture)
# Load the state dictionary from the saved model
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

# Now your model is ready for inference


def create_confusion_matrix(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
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
