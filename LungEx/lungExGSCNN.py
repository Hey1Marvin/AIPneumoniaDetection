import os
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class LungDataset(Dataset):
    def __init__(self, image_dir, mask_dir, indices, transform=None, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.indices = indices
        self.transform = transform
        self.augment = augment
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
        
    def __len__(self):
        return len(self.indices) * (3 if self.augment else 1)  # Augment: 3x größere Dataset
        
    def __getitem__(self, idx):
        # Berechne den Index für das Bild ohne Augmentation
        original_idx = self.indices[idx // 3] if self.augment else self.indices[idx]
        img_name = self.images[original_idx]
        
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        # Augmentationen anwenden, wenn aktiviert
        if self.augment:
            aug_transforms = self.get_augmentation_transforms()
            image = aug_transforms(image)
            mask = aug_transforms(mask)
        
        return image, mask
    
    def get_augmentation_transforms(self):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])

def create_dataloaders(image_dir, mask_dir, batch_size=4, train_val_split=0.9, augment=True):
    # Lade alle Bildnamen
    all_images = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
    
    # Generiere die Indizes für Training und Validierung
    num_images = len(all_images)
    indices = list(range(num_images))
    
    train_indices, val_indices = train_test_split(indices, test_size=1-train_val_split, shuffle=False)
    
    # Transformationen definieren (Resize + ToTensor)
    base_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Erstelle Trainings- und Validierungs-Datensets
    train_dataset = LungDataset(image_dir, mask_dir, train_indices, transform=base_transform, augment=augment)
    val_dataset = LungDataset(image_dir, mask_dir, val_indices, transform=base_transform, augment=augment)
    
    # Erstelle DataLoader für Trainings- und Validierungs-Datensets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader

# Beispiel zur Verwendung
image_dir = 'data/Images'
mask_dir = 'data/Masks'
train_loader, val_loader = create_dataloaders(image_dir, mask_dir, batch_size=4, train_val_split=0.9)


def visualize_prediction(image, mask):
    
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title('Original Image')
    
    plt.subplot(1, 4, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Ground Truth Mask')
    
    plt.subplot(1, 4, 3)
    plt.imshow(prediction, cmap='gray')
    plt.title('Predicted Mask')
    
    plt.subplot(1, 4, 4)
    plt.imshow(image)
    plt.imshow(prediction, cmap='jet', alpha=0.5)  # Überlagere die Maske
    plt.title('Image with Predicted Mask')
    
    plt.show()



for images, masks in train_loader:
    print(images.shape, masks.shape)  # Überprüfe die Größe der geladenen Bilder und Masken
