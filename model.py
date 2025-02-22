import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import ViTFeatureExtractor, ViTForImageClassification

class PrescriptionDataset(Dataset):
    def __init__(self, csv_path, img_dir, feature_extractor):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert("RGB")
        encoding = self.feature_extractor(images=image, return_tensors="pt")
        item = {k: v.squeeze() for k, v in encoding.items()}
        label = torch.tensor(row['label'], dtype=torch.long)
        item["labels"] = label
        return item

def train_loop(model, data_loader, optimizer, device):
    model.train()
    for batch in data_loader:
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def eval_loop(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(pixel_values=pixel_values)
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=10 
    ).to(device)

    train_dataset = PrescriptionDataset(
        csv_path="dataset/Training/training_labels.csv",
        img_dir="dataset/Training/training_images",
        feature_extractor=feature_extractor
    )
    val_dataset = PrescriptionDataset(
        csv_path="dataset/Validation/validation_labels.csv",
        img_dir="dataset/Validation/validation_images",
        feature_extractor=feature_extractor
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(5):
        train_loop(model, train_loader, optimizer, device)
        accuracy = eval_loop(model, val_loader, device)
        print(f"Epoch {epoch+1} - Validation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
