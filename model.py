import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import os
from transformers import ViTForImageClassification, ViTFeatureExtractor

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Define paths
data_dir = "./dataset"  # Adjust the path accordingly
train_csv = os.path.join(data_dir, "train_labels.csv")
val_csv = os.path.join(data_dir, "val_labels.csv")
test_csv = os.path.join(data_dir, "test_labels.csv")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ViT input
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize pixel values to [-1, 1]
])

# Custom dataset class
class PrescriptionDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        
        return image, label

# Load datasets
train_dataset = PrescriptionDataset(train_csv, data_dir, transform)
val_dataset = PrescriptionDataset(val_csv, data_dir, transform)
test_dataset = PrescriptionDataset(test_csv, data_dir, transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pre-trained ViT model
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=10)
model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total
        val_acc = evaluate_model(model, val_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

# Evaluation function
def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5)

# Test the model
test_acc = evaluate_model(model, test_loader)
print(f"Test Accuracy: {test_acc:.4f}")

# Save the model
torch.save(model.state_dict(), "vit_prescription_digits.pth")
