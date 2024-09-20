import os
import pandas as pd
import pydicom
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# Definieren des Pfads zu den Bildern und Labels
image_dir = 'stage_2_train_images'
label_file = 'stage_2_train_labels.csv'

# Labels einlesen
labels_df = pd.read_csv(label_file)

# Dataset-Klasse definieren
class PneumoniaDataset(Dataset):
    def __init__(self, image_dir, labels_df, transform=None):
        self.image_dir = image_dir
        self.labels_df = labels_df
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.labels_df.iloc[idx, 0] + '.dcm')
        ds = pydicom.dcmread(img_name)
        image = ds.pixel_array
        image = Image.fromarray(image).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels_df.iloc[idx, -1]
        return image, label

# Transformations definieren
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Daten vorbereiten
train_df, val_df = train_test_split(labels_df, test_size=0.08, random_state=42)
train_dataset = PneumoniaDataset(image_dir, train_df, transform=transform)
val_dataset = PneumoniaDataset(image_dir, val_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ResNet-34 Modell laden und anpassen
model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Verlustfunktion und Optimierer definieren
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trainingsschleife
num_epochs = 10
best_acc = 0.0
eval_interval = 100  # Intervall, nach dem das Modell evaluiert und gespeichert wird

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)

    # Training
    model.train()
    running_loss = 0.0
    running_corrects = 0
    train_labels = []
    train_preds = []
    batch_count = 0

    for inputs, labels in tqdm(train_loader, desc='Training', leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        train_labels.extend(labels.cpu().numpy())
        train_preds.extend(preds.cpu().numpy())

        batch_count += 1

        if batch_count % eval_interval == 0:
            epoch_loss = running_loss / (batch_count * train_loader.batch_size)
            epoch_acc = accuracy_score(train_labels, train_preds)

            print(f'Batch {batch_count} Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Validation
            model.eval()
            val_loss = 0.0
            val_corrects = 0
            val_labels = []
            val_preds = []

            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc='Validation', leave=False):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    val_corrects += torch.sum(preds == labels.data)
                    val_labels.extend(labels.cpu().numpy())
                    val_preds.extend(preds.cpu().numpy())

            val_loss = val_loss / len(val_dataset)
            val_acc = accuracy_score(val_labels, val_preds)

            print(f'Batch {batch_count} Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

            # Konfusionsmatrix anzeigen
            conf_matrix = confusion_matrix(val_labels, val_preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - Epoch {epoch+1} Batch {batch_count}')
            plt.show()

            # Bestes Modell speichern
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f'resnet34_best.pth')

            # Modell nach jedem Evaluierungsintervall speichern
            torch.save(model.state_dict(), f'resnet34_epoch_{epoch+1}_batch_{batch_count}.pth')

    
    #Nach jeder Epoch evaluieren und speichern
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = accuracy_score(train_labels, train_preds)

    print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # Validation
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    val_labels = []
    val_preds = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation', leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())

    val_loss = val_loss / len(val_dataset)
    val_acc = accuracy_score(val_labels, val_preds)

    print(f'Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

    # Konfusionsmatrix anzeigen
    conf_matrix = confusion_matrix(val_labels, val_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Epoch {epoch+1}')
    plt.show()

    # Bestes Modell speichern
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), f'resnet34_best.pth')

    # Modell nach jeder Epoche speichern
    torch.save(model.state_dict(), f'resnet34_epoch_{epoch+1}.pth')