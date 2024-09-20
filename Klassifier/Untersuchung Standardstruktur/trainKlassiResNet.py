import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import time
import os
import copy
from PIL import Image
import pandas as pd
from tqdm import tqdm
import pydicom
import numpy as np

# Pfade
image_dir = 'rsna-pneumonia-dataset/stage_2_train_images'
label_csv = 'rsna-pneumonia-dataset/stage_2_train_labels.csv'
checkpoint_dir = './checkpoints/'
history_file = 'history.txt'
split_info_path = './split_info.pth'

# Parameter
num_epochs = 150
batch_size = 32
learning_rate = 0.001
test_split_ratio = 0.2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Erstelle Ordner zum Speichern der Checkpoints
os.makedirs(checkpoint_dir, exist_ok=True)


# Initialisiere das Modell (ResNet101), andere Netze sind auskommentiert
def initialize_model():
    # resnet18 = models.resnet18(pretrained=True)
    # resnet34 = models.resnet34(pretrained=True)
    model = models.resnet101(pretrained=True)
    # densenet121 = models.densenet121(pretrained=True)
    # inception_v3 = models.inception_v3(pretrained=True)
    # efficientnet_b0 = models.efficientnet_b0(pretrained=True)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 2 Klassen (Pneumonia / Normal)
    return model.to(device)




# Custom Dataset zum Laden der Bilder und Labels
class PneumoniaDataset(Dataset):
    def __init__(self, image_dir, label_csv, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # CSV-Datei einlesen
        self.labels = pd.read_csv(label_csv)
        
        # Entferne Duplikate (es gibt mehrere Zeilen pro Bild, wenn eine Bbox vorhanden ist)
        self.labels = self.labels.drop_duplicates(subset=['patientId'])
        
        # Mapping der Bild-IDs zu den Labels (0 für gesund, 1 für krank)
        self.labels = self.labels[['patientId', 'Target']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Hole die patientId (Bildname) und das Label
        patient_id = self.labels.iloc[idx, 0]
        label = self.labels.iloc[idx, 1]
        
        # Lade das DICOM-Bild
        image_path = os.path.join(self.image_dir, f"{patient_id}.dcm")
        
        try:
            dicom_image = pydicom.dcmread(image_path)
            image = dicom_image.pixel_array.astype(np.float32)
        except Exception as e:
            print(f"Error reading {image_path}: {e}")
            # Setze das Bild auf eine leere Array (schwarz) und setze das Label auf 0 (kein Bild vorhanden)
            image = np.zeros((224, 224), dtype=np.float32)  # Ersetze 224x224 durch die gewünschte Bildgröße
            label = 0
        
        # Normalisierung und Konvertierung
        image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalisiere Bild
        image = np.expand_dims(image, axis=-1)  # Füge Kanalachse hinzu
        image = np.repeat(image, 3, axis=-1)  # Wiederhole Kanäle für RGB

        image = Image.fromarray((image * 255).astype(np.uint8))  # Konvertiere zu PIL Image
        
        # Wende Transformierungen an, falls angegeben
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Erstelle Dataloader mit angepasstem Dataset
def create_dataloaders(image_dir, label_csv, batch_size, test_split_ratio):
    # Transformierungen definieren (inklusive Normalisierung)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # Data Augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard ImageNet Normalisierung
    ])

    # PneumoniaDataset instanziieren
    dataset = PneumoniaDataset(image_dir, label_csv, transform=transform)

    # Test-Train Split
    num_test = int(test_split_ratio * len(dataset))
    num_train = len(dataset) - num_test
    train_data, test_data = random_split(dataset, [num_train, num_test])

    #Speichern der Teilung
    torch.save({'train_indices': train_data.indices, 'test_indices': test_data.indices}, split_info_path)
    
    # DataLoader erstellen
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Trainings- und Validierungsprozess
def train_model(model, train_loader, test_loader, num_epochs, criterion, optimizer):
    best_accuracy = 0.0
    history = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training
        model.train()
        running_loss = 0.0
        running_corrects = 0
        epoch_start_time = time.time()

        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc='Training Batches', leave=False)):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward Pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            # Backpropagation und Optimierung
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Speichern des Modells nach jedem Viertel der Epoche
            if (i + 1) % max(1, (len(train_loader) // 4)) == 0:
                current_loss = running_loss / ((i + 1) * train_loader.batch_size)
                save_checkpoint(model, optimizer, epoch, i, current_loss, best_accuracy, 'quarter')

        # Ende der Epoche: Berechne Verlust und Genauigkeit
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = running_corrects.double() / len(train_loader.dataset)
        epoch_time = time.time() - epoch_start_time

        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f} Time: {epoch_time:.2f}s')

        # Validierung auf dem Testdatensatz
        test_loss, test_accuracy = validate_model(model, test_loader, criterion)
        print(f'Test Loss: {test_loss:.4f} Test Acc: {test_accuracy:.4f}')

        # Speicherung, wenn das Modell die beste Testgenauigkeit erreicht
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            save_checkpoint(model, optimizer, epoch, i, epoch_loss, best_accuracy, 'best')

        # Speichere den Verlauf
        history.append({
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'train_accuracy': epoch_accuracy.item(),
            'test_loss': test_loss,
            'test_accuracy': test_accuracy.item()
        })
        save_history(history)

        # Speichere die aktuellen Parameter nach jeder Epoche
        save_checkpoint(model, optimizer, epoch, i, epoch_loss, best_accuracy, 'epoch')

    print('Training abgeschlossen.')

def validate_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Validation Batches', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    test_loss = running_loss / len(test_loader.dataset)
    test_accuracy = running_corrects.double() / len(test_loader.dataset)
    return test_loss, test_accuracy

# Speichert die Checkpoints (Modell & Optimizer)
def save_checkpoint(model, optimizer, epoch, batch_idx, loss, best_acc, checkpoint_type):
    checkpoint = {
        'epoch': epoch + 1,
        'batch_idx': batch_idx,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
        'best_accuracy': best_acc
    }
    checkpoint_path = os.path.join(checkpoint_dir, f'{checkpoint_type}_checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint gespeichert: {checkpoint_path}')

# Speichere den Trainingsverlauf in eine Datei
def save_history(history):
    with open(history_file, 'w') as f:
        for entry in history:
            f.write(f"Epoch {entry['epoch']}: Train Loss: {entry['train_loss']:.4f}, Train Acc: {entry['train_accuracy']:.4f}, "
                    f"Test Loss: {entry['test_loss']:.4f}, Test Acc: {entry['test_accuracy']:.4f}\n")
    print(f'Trainingsverlauf gespeichert: {history_file}')

if __name__ == "__main__":
    model = initialize_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, test_loader = create_dataloaders(image_dir, label_csv, batch_size, test_split_ratio)
    train_model(model, train_loader, test_loader, num_epochs, criterion, optimizer)