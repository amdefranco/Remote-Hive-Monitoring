import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
import wandb
import seaborn as sns
import matplotlib.pyplot as plt

   

# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.data = self.data.dropna()
        self.means = []
        self.stds = []
        self.cols = []
        self.columns_to_exclude = ["weatherID","long","lat","device", "hive number",self.data.columns[2], self.data.columns[10], self.data.columns[-1], self.data.columns[-2], self.data.columns[-3], self.data.columns[-4], self.data.columns[-5], self.data.columns[-6], self.data.columns[-7]]
        for col in self.data.columns:

            if col not in self.columns_to_exclude:
                print(col)
                self.means.append(np.mean(self.data[col]))
                std = np.std(self.data[col])
                if std == 0:
                    std = 1
                self.stds.append(std)
                self.cols.append(col)
        self.means = np.array(self.means)
        self.stds = np.array(self.stds)


        #  [self.data.columns[2], self.data.columns[10], self.data.columns[-1], self.data.columns[-2], self.data.columns[-3], self.data.columns[-4], self.data.columns[-5], self.data.columns[-6], self.data.columns[-7]]
       


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        # features = self.data.iloc[idx,3:-7,self.data.columns != self.data.columns[4]]  # Exclude filename and target columns
        target = self.data.iloc[idx, -1]
        features = self.data.iloc[idx, ~self.data.columns.isin(self.columns_to_exclude)]
        features=np.array(list(features))
        features = (features-self.means)/self.stds

        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.long)

# Hyperparameters
input_size = 9  # Assuming 10 features excluding filename and target
hidden_size = 64
output_size = 4  # Number of classes for target prediction
batch_size = 128
epochs = 500
learning_rate = 0.001


# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Honey_Bees",

    # track hyperparameters and run metadata
    config = {'input_size': input_size, 
              'hidden_size': hidden_size, 
              'output_size': output_size, 
              'batch_size': batch_size, 
              'learning_rate': learning_rate, 
              'epochs': epochs},
    name= "MLP_"  + str(hidden_size) + "_" + str(batch_size) + "_" + str(learning_rate)[2:]
           
)


# Load CSV data
csv_file = 'C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\all_data_updated.csv'
# csv_file = 'all_data_updated.csv'
dataset = CustomDataset(csv_file)

# Extract features and targets separately
X = np.array([data[0] for data in dataset])
y = np.array([data[1] for data in dataset])


# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create datasets and data loaders
train_set = [(torch.tensor(X_train[i]), torch.tensor(y_train[i])) for i in range(len(X_train))]
val_set = [(torch.tensor(X_val[i]), torch.tensor(y_val[i])) for i in range(len(X_val))]

# Create data loaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)
# Initialize model, loss function, and optimizer
model = MLP(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
losses= []
# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
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
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            avg_loss = running_loss / len(val_loader)
            accuracy = 100 * correct / total
            wandb.log({'val_loss': avg_loss, 'val_accuracy': accuracy})
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
