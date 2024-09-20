import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm




class UNet(nn.Module):
    
    
    def __init__(self, n=1, batch_size=10, chan_out=1, batchnorm = False):
        super(UNet, self).__init__()
        self.conv_1 = self.conv_layer(n, 0, batchnorm=batchnorm)
        self.conv_2 = self.conv_layer(n, 1, batchnorm=batchnorm)
        self.conv_3 = self.conv_layer(n, 2, batchnorm=batchnorm)
        self.conv_4 = self.conv_layer(n, 3, batchnorm=batchnorm)
        self.conv_5 = self.conv_layer(n, 4, batchnorm=batchnorm)
        self.conv_6 = self.conv_layer(n, 5, batchnorm=batchnorm)
        self.conv_7 = self.conv_layer(n, 6, batchnorm=batchnorm)
        self.conv_8 = self.conv_layer(n, 7, batchnorm=batchnorm)
        self.conv_9 = self.conv_layer(n, 8, batchnorm=batchnorm)
        
        # Upsampling layers
        self.up_6 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=16*n, out_channels=8*n, kernel_size=2, padding='same'),
            nn.ReLU()
        )

        self.up_7 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=8*n, out_channels=4*n, kernel_size=2, padding='same'),
            nn.ReLU()
        )

        self.up_8 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=4*n, out_channels=2*n, kernel_size=2, padding='same'),
            nn.ReLU()
        )

        self.up_9 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=2*n, out_channels=n, kernel_size=2, padding='same'),
            nn.ReLU()
        )
        self.conv_A = nn.Sequential( # output layer with Sigmoid function
            nn.Conv2d(in_channels=n, out_channels=chan_out, kernel_size=1, padding='same'),
            nn.Sigmoid()
        )

    def conv_layer(self, n, l_n, batchnorm=False):
        if l_n == 0:
            in_  = 1
            out_ = n
        elif l_n < 5:
            in_  = int(n * (2 ** (l_n - 1)))
            out_ = int(n * (2 ** l_n))
        else:
            in_  = int(n * (2 ** (9 - l_n)))
            out_ = int(n * (2 ** (9 - l_n - 1)))

        if batchnorm:
            conv = nn.Sequential(
                nn.Conv2d(in_channels=in_, out_channels=out_, kernel_size=3, padding='same'),
                nn.ReLU(),
                nn.BatchNorm2d(out_),
                nn.Conv2d(in_channels=out_, out_channels=out_, kernel_size=3, padding='same'),
                nn.ReLU(),
                nn.BatchNorm2d(out_),
            )
        else:
            conv = nn.Sequential(
                nn.Conv2d(in_channels=in_, out_channels=out_, kernel_size=3, padding='same'),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_, out_channels=out_, kernel_size=3, padding='same'),
                nn.ReLU(),
            )
        return conv

    def forward(self, input_):
        conv_1 = self.conv_1(input_)
        pool_1 = nn.MaxPool2d(kernel_size=(2,2))(conv_1)

        conv_2 = self.conv_2(pool_1)
        pool_2 = nn.MaxPool2d(kernel_size=(2,2))(conv_2)

        conv_3 = self.conv_3(pool_2)
        pool_3 = nn.MaxPool2d(kernel_size=(2,2))(conv_3)

        conv_4 = self.conv_4(pool_3)
        pool_4 = nn.MaxPool2d(kernel_size=(2,2))(conv_4)

        conv_5 = self.conv_5(pool_4)

        up_6    = self.up_6(conv_5)
        merge_6 = torch.cat([conv_4, up_6], dim=1)
        conv_6  = self.conv_6(merge_6)

        up_7    = self.up_7(conv_6)
        merge_7 = torch.cat([conv_3, up_7], dim=1)
        conv_7  = self.conv_7(merge_7)

        up_8    = self.up_8(conv_7)
        merge_8 = torch.cat([conv_2, up_8], dim=1)
        conv_8  = self.conv_8(merge_8)

        up_9    = self.up_9(conv_8)
        merge_9 = torch.cat([conv_1, up_9], dim=1)
        conv_9  = self.conv_9(merge_9)

        return self.conv_A(conv_9)

class LungDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, augmentations=3, target_size=(224, 224), size = 1, test = False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.augmentations = augmentations
        self.target_size = target_size
        self.images = os.listdir(image_dir)
        
        if (size < 1):#splitting in test and train
            numImg = int(size * len(self.images))
            if test: self.images = self.images[len(self.images)-numImg:]
            else: self.images = self.images[:numImg]
            
                
        self.resize = transforms.Resize(self.target_size)

    def __len__(self):
        return len(self.images) * (1 + self.augmentations)

    def __getitem__(self, idx):
        original_idx = idx // (1 + self.augmentations)
        img_path = os.path.join(self.image_dir, self.images[original_idx])
        mask_path = os.path.join(self.mask_dir, self.images[original_idx])

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # Resize image and mask to the target size
        image = self.resize(image)
        mask = self.resize(mask)

        if idx % (1 + self.augmentations) == 0:
            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)
        else:
            # Apply data augmentation
            augmentation_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor()
            ])
            image = augmentation_transform(image)
            mask = augmentation_transform(mask)

        return image, mask

# Data Augmentation und Resize
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = LungDataset('./Images', './Masks', transform=transform, augmentations=3, target_size=(224, 224), size = 0.9, test = False)
test_dataset = LungDataset('./Images', './Masks', transform=transform, augmentations=3, target_size=(224, 224), size = 0.1, test = True)
'''
# Split in Training und Test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
'''
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

# Model, Loss, Optimizer
model = UNet()
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def train_model(model, train_loader, criterion, optimizer, num_epochs=150):
    best_loss = float('inf')

    with open("history.txt", "w") as f:
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0

            # Fortschrittsanzeige mittels tqdm
            for images, masks in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            f.write(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss}, LR: {optimizer.param_groups[0]["lr"]}\n')

            # Save model and optimizer state if loss improves
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                }, f'checkpoint_epoch_{epoch + 1}.pt')

# Training ausführen
train_model(model, train_loader, criterion, optimizer)
