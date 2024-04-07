
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

def convert_label(label:str):
    conv_map = {"queen present or original queen":0,
                "queen not present":1,
                "queen present and rejected":2, 
                "queen present and newly accepted":3}
    return conv_map[label]

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.valid_indices = []  # List to store indices of valid samples

        # Check for valid samples
        for idx, row in self.data.iterrows():
            img_name = row['file name']
            img_name = "C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\scripts\\imgs\\" + img_name
            img_name = img_name[:-4] + ".png"
            if os.path.exists(img_name):
                self.valid_indices.append(idx)
                print(row["queen status"])

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        idx = self.valid_indices[idx]  # Get the index of valid sample
        img_name = self.data.iloc[idx, 0]  # Assuming "file name" is the first column
        label = self.data.iloc[idx, 1]  # Assuming "queen states" is the second column
        label = convert_label(label)
        img_name = "C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\scripts\\imgs\\" + img_name
            
        img_name = img_name[:-4] + ".png"
        image = test_image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
class CNNModel(nn.Module):
    def __init__(self, num_classes=4):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 56 * 56)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224
    transforms.ToTensor(),           # Convert PIL image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize image
])

# Hyperparameters
batch_size = 1
epochs = 500
learning_rate = 0.001

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Honey_Bees",

    # track hyperparameters and run metadata
    config = {'batch_size': batch_size, 
              'learning_rate': learning_rate, 
              'epochs': epochs},
    name= "CNN_"   + "_" + str(batch_size) + "_" + str(learning_rate)[2:]
           
)
# Create custom dataset and DataLoader
dataset = CustomDataset(csv_file='C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\train.csv', transform=transform)

# Define the size of train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
exit()
# Split dataset into train and validation sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoader for train and validation sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Example usage in training loop
model = CNNModel(num_classes=4)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
losses=[]
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        # import ipdb; ipdb.set_trace()
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
                outputs = model(inputs)
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
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    wandb.log({'final_val_loss': avg_loss, 'final_val_accuracy': accuracy})
accuracy = correct / total
print(f"Validation Accuracy: {accuracy}")



wandb.finish()

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
