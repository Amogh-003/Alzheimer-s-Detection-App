import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image

# Hyperparameters
batch_size = 32
num_epochs = 10
img_size = 128

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Data transformations
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Update path to your dataset folder here
data_dir = "dataset/Data"
labels_map = {
    "Mild Dementia": 0,
    "Moderate Dementia": 1,
    "Non Demented": 2,
    "Very mild Dementia": 3
}

# Custom dataset class
class AlzheimerDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, labels_map, transform=None):
        self.data_dir = data_dir
        self.labels_map = labels_map
        self.transform = transform
        self.data = []

        for label_name, label in labels_map.items():
            folder = os.path.join(data_dir, label_name)
            for img_name in os.listdir(folder):
                img_path = os.path.join(folder, img_name)
                self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Dataset and DataLoader
train_dataset = AlzheimerDataset(data_dir, labels_map, transform)
from torch.utils.data import random_split

# Lengths for each subset
total_size = len(train_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size  # ensures all data is used

# Split the dataset
train_data, val_data, test_data = random_split(train_dataset, [train_size, val_size, test_size])

# Create DataLoaders for each subset
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# Model Definition
class AlzheimerCNN(nn.Module):
    def __init__(self):
        super(AlzheimerCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the training process in a function
def train():
    # Instantiate model and move to device
    model = AlzheimerCNN().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 0:
                print(f"Batch {i}/{len(train_loader)} | Loss: {running_loss / (i + 1):.4f}")

        print(f"Epoch {epoch+1} | Average Loss: {running_loss / len(train_loader):.4f}")

    print("Training Complete!")
    torch.save(model.state_dict(), "alzheimer_model.pth")
    print("Model saved successfully!")

# Only run training if this script is executed directly
if __name__ == "__main__":
    train()