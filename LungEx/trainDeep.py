import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# Setup paths
dataset_dir = 'dataset'
images_dir = os.path.join(dataset_dir, 'CXR_png')
masks_dir = os.path.join(dataset_dir, 'masks')

# Model / learning Parameters
num_epochs = 50
learning_rate = 0.001
batch_size = 8

class ChestXrayDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.images[idx].replace(".", "_mask."))

        # Load the X-ray image (Grayscale)
        image = Image.open(img_path).convert('L')

        # Load the mask (Grayscale)
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            # Convert grayscale to 3 channels if required
            image = image.repeat(3, 1, 1)
            mask = self.transform(mask)

        # Convert mask to binary mask (values 0 or 1)
        mask = torch.where(mask > 0.5, 1.0, 0.0)

        return image, mask, self.images[idx]

# Define the DeepLabv3+ model
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1):
        super(DeepLabV3Plus, self).__init__()
        # Load DeepLabV3+ pre-trained on COCO dataset
        self.deeplab = models.segmentation.deeplabv3_resnet50(pretrained=True)
        # Replace the classifier with a binary classification layer
        self.deeplab.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))

    def forward(self, x):
        # DeepLabV3+ expects input in the form of [batch, channels, height, width]
        return self.deeplab(x)['out']  # Output logits (without sigmoid)

# Function to save the model
def save_model(epoch, model, optimizer, scheduler, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }, path)

# Function to calculate accuracy
def calculate_accuracy(pred, mask):
    pred = (pred > 0.5).float()
    correct = (pred == mask).sum().item()
    return correct / mask.numel()

def validation(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0.0
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for images, masks, _ in val_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            outputs = torch.sigmoid(outputs)  # Apply sigmoid for binary classification
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            # Calculate accuracy
            correct_pixels += calculate_accuracy(outputs, masks) * masks.numel()
            total_pixels += masks.numel()

    val_loss /= len(val_loader)
    accuracy = correct_pixels / total_pixels

    return val_loss, accuracy

def train(model, device, train_loader, val_loader, num_epochs=50):
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')

        # Training phase
        for images, masks, _ in train_bar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            outputs = torch.sigmoid(outputs)  # Apply sigmoid for binary classification
            outputs = F.interpolate(outputs, size=masks.shape[2:], mode="bilinear", align_corners=False)
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        val_loss, accuracy = validation(model, val_loader, device, criterion)

        # Logging
        print(f"Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

        with open('history.txt', 'a') as f:
            f.write(f"Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader):.4f}, "
                    f"Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}\n")

        # Save the model twice within an epoch
        save_model(epoch, model, optimizer, scheduler, val_loss, f'model_deeplab_epoch{epoch+1}_step2.pth')
        if epoch != 0: os.remove(f'model_deeplab_epoch{epoch}.pth')
        
        # Save the best model
        if val_loss < best_loss:
            best_loss = val_loss
            save_model(epoch, model, optimizer, scheduler, val_loss, 'best_model_deeplab.pth')

        # Update the learning rate
        scheduler.step()

if __name__ == '__main__':
    # Transformations for the dataset
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resizing to match DeepLabV3+ input size
        transforms.ToTensor()
    ])

    # Load dataset
    dataset = ChestXrayDataset(images_dir, masks_dir, transform=transform)

    # Split the dataset into training and validation sets and shuffle
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, optimizer, and learning rate scheduler
    model = DeepLabV3Plus()
    criterion = nn.BCELoss()  # Binary Cross Entropy loss for segmentation
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Train the model
    train(model, device, train_loader, val_loader, num_epochs)
