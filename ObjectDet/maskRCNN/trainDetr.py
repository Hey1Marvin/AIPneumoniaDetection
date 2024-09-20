#zu installierende bibliotheken:
# pip install torchvision
# pip install timm
# pip install effdet

import pydicom
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision
import math
import random
import imgaug
from tqdm import tqdm
from sklearn.model_selection import train_test_split

#Für EfficentDet
import effdet
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet

# Für DETR
import torchvision.models.detection as detection
from transformers import DetrForObjectDetection, DetrImageProcessor
#from torchvision.models.detection.transformer import Detr

#Für Swin
import timm
from timm.models.swin_transformer import swin_base_patch4_window7_224





# Pfade
image_dir = 'rsna-pneumonia-dataset/stage_2_train_images'
label_file = 'rsna-pneumonia-dataset/stage_2_train_labels.csv'
checkpoint_dir = './checkpointsMask/'
history_file = 'historyMask.txt'
#split_info_path = './split_info_Mask.pth'
train_label = 'train.csv'
val_label = 'val.csv'


# Parameter
num_epochs = 50
batch_size = 32
learning_rate = 0.0001
test_split_ratio = 0.2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PneumoniaDataset(Dataset):
    def __init__(self, image_dir, df, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.labels = df
        self.labels.fillna(-1, inplace=True)
        self.labels['Target'] = self.labels['Target'].astype(int)
        self.image_ids = self.labels['patientId'].unique()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id + '.dcm')
        dicom = pydicom.read_file(image_path)
        image = dicom.pixel_array
        
        # Convert grayscale image to RGB
        if len(image.shape) == 2:  # Grayscale image
            image = np.stack([image] * 3, axis=2)
        
        image = F.to_tensor(image).float()

        # Initialize lists for boxes, class labels, and masks
        boxes = []
        class_labels = []
        masks = []

        # Get image data from the dataset
        patient_data = self.labels[self.labels['patientId'] == image_id]

        # Create empty mask for this image (all zeros initially)
        pixel_mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.uint8)

        # Add boxes, class labels, and masks for each pneumonia region
        for _, row in patient_data.iterrows():
            if row['Target'] == 1:
                # Add box coordinates in the format (x_min, y_min, x_max, y_max)
                boxes.append([row['x'], row['y'], row['x'] + row['width'], row['y'] + row['height']])
                class_labels.append(1)  # Pneumonia label
                
                # Create a mask for the region where pneumonia is present
                mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.uint8)
                mask[int(row['y']):int(row['y'] + row['height']), int(row['x']):int(row['x'] + row['width'])] = 1
                masks.append(mask)
            else:
                class_labels.append(0)  # No pneumonia

        # Handle case where no pneumonia region is present (Label 0)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            class_labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.uint8)  # Empty mask for no pneumonia
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            class_labels = torch.as_tensor(class_labels, dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)

        # Format target dictionary for DETR model
        target = {
            'boxes': boxes,        # Bounding Boxes
            'labels': class_labels,  # Class Labels (1 for pneumonia, 0 otherwise)
            'masks': masks,        # Pixel Masks (for segmentation)
        }

        # Apply transformations if available
        if self.transforms:
            image, target = self.transforms(image, target)

        # Return image, pixel mask (used in DETR for padding), and the target
        return image, pixel_mask, target




#Initialize the Object Detection Models
def get_model_instance_segmentation(num_classes): #Mask RCNN
    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, num_classes)
    return model

def get_efficientdet_model(num_classes):
    config = get_efficientdet_config('tf_efficientdet_d0')  # 'd0' ist eine kleinere Version, passe an je nach Bedarf (d0, d1, d2...)
    config.num_classes = num_classes
    config.image_size = (512, 512)  # Standardgröße für EfficientDet
    
    model = EfficientDet(config, pretrained_backbone=True)
    model.class_net = HeadNet(config, num_outputs=num_classes)
    
    return DetBenchTrain(model, config)

def get_detr_model(num_classes):
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    # Möglicherweise musst du hier noch Anpassungen vornehmen, um die Anzahl der Klassen zu ändern
    return model

def get_swin_model(num_classes):
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_classes)
    return model




def train_model(model, optimizer, dataloader, val_loader, device, start_epoch=0, num_epochs=50, best_acc=0):
    model.to(device)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = len(dataloader)
        
        with tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            running_loss = 0.0
            for i, (images, pixel_masks, targets) in enumerate(dataloader):
                # Move data to the correct device
                images = images.to(device)  # Move images to device
                pixel_masks = pixel_masks.to(device)  # Move pixel masks to device

                # Convert targets into the required format and move to device
                labels = [{'class_labels': t['labels'].to(device), 'boxes': t['boxes'].to(device)} for t in targets]
                
                # Forward pass
                outputs = model(pixel_values=images, pixel_mask=pixel_masks, labels=labels)
                
                # Compute losses
                loss_dict = outputs.losses
                losses = sum(loss for loss in loss_dict.values())
                
                # Backward pass
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                # Logging
                running_loss += losses.item()
            

                # Anzeige des aktuellen Verlusts im Fortschrittsbalken
                pbar.set_postfix({'Loss': f'{losses.item():.4f}'})
                pbar.update(1)

                # Speichern nach jedem Viertel
                if (i + 1) % (num_batches // 4) == 0:
                    quarter_checkpoint = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}_quarter_{i // (num_batches // 4) + 1}.pth')
                    torch.save(model.state_dict(), quarter_checkpoint)
                    if (i + 1) > num_batches // 4:
                        old_quarter = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}_quarter_{(i // (num_batches // 4))}.pth')
                        if os.path.exists(old_quarter):
                            os.remove(old_quarter)

        # Lernratenplaner aktualisieren
        lr_scheduler.step()
        
        # Validierung und Testgenauigkeit nach jeder Epoche
        val_acc, avg_loss = evaluate_model(model, val_loader, device)
        print(f"Validation Accuracy after epoch {epoch+1}: {val_acc:.4f}")

        # Speichern des letzten Modells nach jeder Epoche
        epoch_checkpoint = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc
        }, epoch_checkpoint)

        # Bestes Modell speichern, falls bessere Genauigkeit
        if val_acc > best_acc:
            best_acc = val_acc
            best_checkpoint = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc
            }, best_checkpoint)
        
        # Historie aktualisieren
        with open(history_file, 'a') as f:
            f.write(f'Epoch {epoch+1}: Loss {epoch_loss/num_batches:.4f}, Validation Accuracy {val_acc:.4f}, Validation Loss {avg_loss:.4f}\n')
        

def evaluate_model(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            
            for output, target in zip(outputs, targets):
                pred_labels = output['labels'].cpu().numpy()
                true_labels = target['labels'].cpu().numpy()
                total += len(true_labels)
                correct += (pred_labels == true_labels).sum()
                # Summieren der Verluste für die Validierung
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    return accuracy, avg_loss

def main_old():
    # Daten in Trainings- und Validierungssets aufteilen (im Speicher, kein CSV-Speichern)
    #df = pd.read_csv(label_file)
    #train_df, val_df = train_test_split(df, test_size=0.2)
    train_df, val_df = pd.read_csv(train_label), pd.read_csv(val_label)

    # Datensätze erstellen
    dataset = PneumoniaDataset(image_dir, train_df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    val_dataset = PneumoniaDataset(image_dir, val_df)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Modell und Optimizer initialisieren
    model = get_model_instance_segmentation(num_classes=2)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=learning_rate, momentum=0.9, weight_decay=0.0005)

    # Falls ein Checkpoint existiert, laden
    start_epoch = 0
    best_acc = 0
    checkpoint_path = os.path.join('modelParamMask', 'model_epoch_6_0.3753.pth')
    if os.path.exists(checkpoint_path):
        try: 
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint.get('best_acc', 0)
        except: 
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"Model parameters loaded from {checkpoint_path}")
            start_epoch = 6
            
            

    # Training
    train_model(model, optimizer, dataloader, val_loader, device, start_epoch=start_epoch, num_epochs=num_epochs, best_acc=best_acc)

def collate_fn(batch):
    images, pixel_masks, targets = zip(*batch)
    
    # Staple Bilder zu einem Tensor
    images = torch.stack(images, dim=0)
    
    # Konvertiere pixel_masks von numpy zu torch.Tensor und staple sie
    pixel_masks = [torch.tensor(mask, dtype=torch.long) if isinstance(mask, np.ndarray) else mask for mask in pixel_masks]
    pixel_masks = torch.stack(pixel_masks, dim=0)
    
    return images, pixel_masks, targets


def main(model_type='efficientdet'):
    # Daten vorbereiten
    train_df, val_df = pd.read_csv(train_label), pd.read_csv(val_label)
    
    dataset = PneumoniaDataset(image_dir, train_df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    val_dataset = PneumoniaDataset(image_dir, val_df)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Modell und Optimizer auswählen
    if model_type == 'maskrcnn':
        model = get_model_instance_segmentation(num_classes=2)
    elif model_type == 'efficientdet':
        model = get_efficientdet_model(num_classes=2)
    elif model_type == 'detr':
        model = get_detr_model(num_classes=2)
    elif model_type == 'swin':
        model = get_swin_model(num_classes=2)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    
    start_epoch = 0
    best_acc = 0
    checkpoint_path = os.path.join('modelParam', 'model_epoch_6.pth')
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint.get('best_acc', 0)
        except:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"Model parameters loaded from {checkpoint_path}")
            start_epoch = 6

    # Training starten
    train_model(model, optimizer, dataloader, val_loader, device, start_epoch=start_epoch, num_epochs=num_epochs, best_acc=best_acc)

if __name__ == "__main__":
    # Modellauswahl basierend auf Input
    model_type = 'detr'  # Alternativen: 'detr', 'swin'
    main(model_type)
