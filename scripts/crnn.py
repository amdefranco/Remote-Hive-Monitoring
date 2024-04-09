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

 
def convert_label(label:str):
    conv_map = {"queen present or original queen":0,
                "queen not present":1,
                "queen present and rejected":2, 
                "queen present and newly accepted":3}
    num_label = conv_map[label]
    return num_label

# Define a custom dataset class
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


# class CRNN(nn.Module):
#     def __init__(self, , num_classes):
#         super(CRNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
#         self.lstm = nn.LSTM(input_size=128 * (input_shape[0] // 8) * (input_shape[1] // 8), hidden_size=64, num_layers=1, batch_first=True)
#         self.fc = nn.Linear(64, num_classes)

#     def forward(self, x):
#         import ipdb; ipdb.set_trace()
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = x.permute(0, 2, 3, 1)  # Change the dimensions for LSTM
#         batch_size, seq_length, channels, height = x.size()
#         x = x.reshape(batch_size, seq_length, -1)
#         lstm_out, _ = self.lstm(x)
#         x = lstm_out[:, -1, :]  # Take the last output of LSTM
#         x = self.fc(x)
#         return x
# class CRNN(nn.Module):
#     def __init__(self, num_classes, input_size=(3, 32, 100), hidden_size=256, num_layers=2):
#         super(CRNN, self).__init__()
#         self.conv1 = nn.Conv2d(input_size[0], 64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
#         self.rnn = nn.LSTM(input_size[1]//8 * 256, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)
        
#     def forward(self, x):
#         # Convolutional layers
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = F.relu(self.conv3(x))
#         x = self.pool(x)
        
#         # Reshape for RNN
#         batch_size, _, height, width = x.size()
#         x = x.permute(0, 2, 3, 1)  # [batch_size, height, width, channels]
#         x = x.reshape(batch_size, height, -1)  # [batch_size, height, width*channels]
        
#         # RNN layer
#         x, _ = self.rnn(x)
        
#         # Fully connected layer
#         x = self.fc(x[:, -1, :])  # Take the last time step output
        
#         return x

class CRNN(nn.Module):
    def __init__(self, num_classes=4, hidden_size=256, num_layers=2):
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.rnn = nn.LSTM(input_size=32 * 56 * 56, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 4)

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
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224
    transforms.ToTensor(),           # Convert PIL image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize image
])
# Hyperparameters
# batch_sizes = [64,256,128,24]
hidden_sizes = [256]
batch_size = 64
epochs_list = [110]
lrs= [0.0005]
num_layers=2
for hidden_size in hidden_sizes:  
    for epochs in epochs_list:
        for learning_rate in lrs:
            # start a new wandb run to track this script
            wandb.init(
                # set the wandb project where this run will be logged
                project="Honey_Bees",

                # track hyperparameters and run metadata
                config = {'batch_size': batch_size, 
                        'learning_rate': learning_rate, 
                        'epochs': epochs},
                name= "CRNN_"   + "_" + str(batch_size) + "_" + str(learning_rate)[2:] + "_" + str(hidden_size) + "_" + str(num_layers)
                    
            )
            # Create custom dataset and DataLoader
            dataset = CustomDataset(csv_file='C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\train.csv', transform=transform)

            # Define the size of train and validation sets
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size

            # Split dataset into train and validation sets
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            # Create DataLoader for train and validation sets
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


            # Example usage in training loop
            # model = CRNN( input_size=(3,224,224), num_classes=4,hidden_size=hidden_size)
            model = CRNN(num_classes=4,hidden_size=hidden_size,num_layers=num_layers)
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print(device)
            model = model.to(device)


            print("Created model")
            criterion = torch.nn.CrossEntropyLoss()
            criterion = criterion.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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

                print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss}")
                avg_loss = running_loss / len(train_loader)
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
           

            torch.save(model.state_dict(), "models2\\aSDCRNN_"   + "_" + str(batch_size) + "_" + str(learning_rate)[2:] + "_" + str(hidden_size) + ".pth")
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
