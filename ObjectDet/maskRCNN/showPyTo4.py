import os
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class PneumoniaDataset(Dataset):
    def __init__(self, image_dir, label_file, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.labels = pd.read_csv(label_file)
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
        if len(image.shape) == 2:  # Grayscale image
            image = np.stack([image] * 3, axis=2)
        image = F.to_tensor(image).float()
        
        boxes = []
        masks = []
        labels = []
        
        patient_data = self.labels[self.labels['patientId'] == image_id]
        for _, row in patient_data.iterrows():
            if row['Target'] == 1:
                boxes.append([row['x'], row['y'], row['x'] + row['width'], row['y'] + row['height']])
                mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.uint8)
                mask[int(row['y']):int(row['y'] + row['height']), int(row['x']):int(row['x'] + row['width'])] = 1
                masks.append(mask)
                labels.append(1)
            else:
                labels.append(0)
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
        }
        
        if self.transforms:
            image, target = self.transforms(image, target)
        
        return image, target

def get_model_instance_segmentation(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, num_classes)
    return model

def save_images_with_masks(model, dataloader, device, num_images, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    with torch.no_grad():
        with tqdm(total=num_images, desc="Processing images", unit="image") as pbar:
            for images, targets in dataloader:
                if count >= num_images:
                    break
                
                images = list(image.to(device) for image in images)
                outputs = model(images)
                
                for idx, output in enumerate(outputs):
                    if count >= num_images:
                        break
                    
                    labels = output['labels'].cpu().numpy()
                    if 1 not in labels:
                        continue  # Skip images without label 1
                    
                    image = images[idx].cpu().numpy().transpose((1, 2, 0))
                    image = (image * 255).astype(np.uint8)
                    
                    masks = output['masks'].cpu().numpy()
                    scores = output['scores'].cpu().numpy()
                    
                    fig, ax = plt.subplots(1, figsize=(12, 12))
                    ax.imshow(image, cmap='gray')
                    
                    for mask, label, score in zip(masks, labels, scores):
                        #if score < 0.5:  # Skip detections with low confidence
                        #    continue
                        mask = mask[0]
                        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                        colored_mask[mask > 0.5] = [255, 0, 0]  # Red color
                        blended_image = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
                        
                        # Draw mask boundary
                        contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:  # Check if contours are found
                            cv2.drawContours(blended_image, contours, -1, (255, 0, 0), 2)  # Red color for boundary
                            
                            # Add score text
                            x, y, w, h = cv2.boundingRect(contours[0])
                            ax.text(x, y, f'Score: {score:.2f}', color='white', backgroundcolor='red')
                        
                        ax.imshow(blended_image, alpha=0.5)  # Overlay mask with transparency
                    
                    ax.axis('off')
                    output_path = os.path.join(output_dir, f'image_{count}_class_1.png')
                    plt.savefig(output_path, bbox_inches='tight')
                    plt.close(fig)
                    
                    count += 1
                    pbar.update(1)

def main():
    image_dir = 'stage_2_train_images'
    label_file = 'stage_2_train_labels.csv'
    val_file = 'val.csv'
    model_path = 'modelParam/model_epoch_2_num_600_loss240.3482.pth'  # Pfad zur vortrainierten Modell-Datei
    
    #Datensatz in train und val zerteilen:
    # CSV-Datei einlesen
    df = pd.read_csv(label_file)

    # Zeilen mischen
    df = df.sample(frac=1)

    # Daten in Trainings- und Validierungssets aufteilen
    _, val_df = train_test_split(df, test_size=0.2)

    # Validierungsdaten in CSV-Datei speichern
    val_df.to_csv(val_file, index=False)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model_instance_segmentation(num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    # Load validation dataset
    dataset_val = PneumoniaDataset(image_dir, val_file)
    dataloader_val = DataLoader(dataset_val, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    
    # Process and save images with masks
    print("Processing and saving images with masks...")
    output_dir = 'output_images'
    num_images_to_process = 10  # Specify the number of images to process
    save_images_with_masks(model, dataloader_val, device, num_images_to_process, output_dir)

if __name__ == "__main__":
    main()
