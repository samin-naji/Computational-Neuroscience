import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.io import read_image
import matplotlib.pyplot as plt

# Set the path to the directory containing the saved spectrogram images
spectrogram_dir = 'D:\\computational_neuroscience\\assignment8\\spectrogrum_data_of_h0'

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Adjust size as needed
    transforms.ToTensor(),
])

# Create a custom dataset
class SpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)

    def __len__(self):
        return sum(len(files) for _, _, files in os.walk(self.root_dir))

    def __getitem__(self, idx):
        class_idx = idx // len(self.classes)        
        class_dir = os.path.join(self.root_dir, self.classes[class_idx])
        
        
        # file_idx = idx % len(self.classes)
        # filename = os.listdir(class_dir)[file_idx]
        # img_path = os.path.join(class_dir, filename)
        image = read_image(class_dir)
       
        if self.transform:
            image = self.transform(image)

        # Assuming class names are the same as directory names
        label = class_idx
        return image, label

# Create datasets and dataloaders
dataset = SpectrogramDataset(spectrogram_dir, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the CNN model
class SpectrogramCNN(nn.Module):
    def __init__(self):
        super(SpectrogramCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  # Adjust input size based on your image size
        self.fc2 = nn.Linear(128, 2)  # Adjust output size based on the number of classes

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)  # Adjust input size based on your image size
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model and define loss function and optimizer
model = SpectrogramCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# Evaluate on the test set
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")