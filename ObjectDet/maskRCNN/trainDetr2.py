import os
import torch
import pandas as pd
import pydicom
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import DetrForObjectDetection, AutoImageProcessor
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import torch.optim as optim

# Pfade zu den Dateien
image_dir = 'rsna-pneumonia-dataset/stage_2_train_images/'
label_file = 'rsna-pneumonia-dataset/stage_2_train_labels.csv'
checkpoint_dir = './checkpoints'
split_file = 'dataset_split.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Rsnadataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.df = pd.read_csv(label_file)
        self.df.fillna(0, inplace=True)  # Füllt alle NaN-Werte mit 0 auf

        # Ensure only unique patient IDs are used
        #self.unique_patients = self.df['patientId'].unique()
        #self.dfu = self.df[self.df['patientId'].isin(self.unique_patients)]

    def __len__(self):
        return self.df['patientId'].unique()

    def __getitem__(self, idx):
        # Extrahiere das Bild anhand des Index
        #image_id = self.unique_patients[idx]
        image_id = self.df['patientId'][idx]
        img_path = os.path.join(self.image_dir, image_id + ".dcm")
        img = pydicom.dcmread(img_path).pixel_array  # Read the DICOM image
        
        # Wandle das Graustufenbild in ein RGB-Bild um
        img = np.stack([img] * 3, axis=0)  # [H, W, 3]

        # Filtere die Zeilen, die sich auf dieses Bild beziehen (für mehrere Boxen)
        records = self.df[self.df['patientId'] == image_id]

        # Initialisiere Listen für Boxen und Labels
        boxes = []
        labels = []

        for _, row in records.iterrows():
            label = int(row['Target'])  # Label (1 = positiver Fall, 0 = negativer Fall)
            if label == 1:
                # Konvertiere (x, y, width, height) zu (xmin, ymin, xmax, ymax)
                xmin = row['x']
                ymin = row['y']
                xmax = xmin + row['width']
                ymax = ymin + row['height']
                
                box = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)
                boxes.append(box)
                labels.append(label)
                break

        if len(boxes) == 0:
            # Falls keine Boxen vorhanden sind, füge eine leere Box hinzu
            boxes.append(torch.tensor([0, 0, 0, 0], dtype=torch.float32))
            labels.append(0)

        # Staple alle Boxen zu einem Tensor
        boxes = torch.stack(boxes)

        # Erstelle den Sample-Dictionary
        sample = {
            "image": img,
            "boxes": boxes,  # Boxen als Tensor [N, 4]
            "labels": torch.tensor(labels, dtype=torch.int64)  # Labels als Tensor [N]
        }

        # Wende Transformationen auf das Bild an
        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

# Funktion zum Speichern der Trainings- und Validierungsaufteilung
def save_split(train_indices, val_indices):
    torch.save({'train_indices': train_indices, 'val_indices': val_indices}, split_file)

# Funktion zum Laden der Trainings- und Validierungsaufteilung
def load_split():
    if os.path.exists(split_file):
        return torch.load(split_file)
    else:
        return None

# Funktion zum Laden des Datensatzes und Splitten in Training und Validierung
def load_data(image_dir, label_file, test_size=0.2, batch_size=4, resume_split=False):
    dataset = Rsnadataset(image_dir, label_file)

    # Prüfen, ob eine gespeicherte Aufteilung vorhanden ist
    split = load_split() if resume_split else None
    if split:
        train_indices = split['train_indices']
        val_indices = split['val_indices']
        print("Datensplit geladen.")
    else:
        # Patienten IDs aufteilen
        patient_ids = dataset.df['patientId'].unique()
        train_ids, val_ids = train_test_split(patient_ids, test_size=test_size)
        train_indices = dataset.df[dataset.df['patientId'].isin(train_ids)].index.tolist()
        val_indices = dataset.df[dataset.df['patientId'].isin(val_ids)].index.tolist()
        
        # Speichern des Splits
        save_split(train_indices, val_indices)
        print("Datensplit gespeichert.")

    # Subset basierend auf den Indizes erstellen
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# Funktion zum Speichern von Checkpoints
def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(state, os.path.join(checkpoint_dir, filename))
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, 'best_model.pth.tar'))

# Funktion zum Laden von Checkpoints
def load_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    if os.path.isfile(os.path.join(checkpoint_dir, filename)):
        checkpoint = torch.load(os.path.join(checkpoint_dir, filename))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        print(f"=> loaded checkpoint '{filename}' (epoch {epoch})")
        return epoch
    else:
        print(f"=> no checkpoint found at '{filename}'")
        return 0

# Training und Validierung
def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-5):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    best_acc = 0.0
    start_epoch = 0

    # Lade Checkpoint, falls vorhanden
    if os.path.exists(os.path.join(checkpoint_dir, "checkpoint.pth.tar")):
        start_epoch = load_checkpoint(model, optimizer, "checkpoint.pth.tar")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for batch in train_loader:
                images = batch['image'].float().to(device)
                labels = [{'class_labels': batch['labels'].to(device), 'boxes': batch['boxes'].to(device)}]
                outputs = model(images, labels=labels)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
                pbar.update(1)

        # Validation
        val_acc, val_loss = validate_model(model, val_loader)
        print(f'Validation loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}')

        # Speichern des Checkpoints
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'best_acc': best_acc}, is_best)
        
        # Fortschritt in Datei speichern
        with open('historyDetr.txt', 'a') as f:
            f.write(f'Epoch {epoch+1}/{num_epochs}, Training loss: {running_loss:.4f}, Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.2f}, Learning rate: {lr}\n')

# Validierungsfunktion
def validate_model(model, val_loader):
    model.eval()
    total = 0
    correct = 0
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].float().cuda()
            labels = [{'class_labels': batch['labels'].cuda(), 'boxes': batch['boxes'].cuda()}]
            outputs = model(images, labels=labels)

            val_loss += outputs.loss.item()

            pred = outputs.logits.argmax(-1)
            correct += (pred == batch['labels']).sum().item()
            total += len(batch['labels'])

    val_acc = correct / total
    return val_acc, val_loss / len(val_loader)

# Hauptfunktion
if __name__ == '__main__':
    resume_training = False  # Setze auf True, wenn das Training fortgesetzt werden soll
    train_loader, val_loader = load_data(image_dir, label_file, resume_split=resume_training)
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
    train_model(model, train_loader, val_loader, num_epochs=10)
