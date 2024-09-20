import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision
import cv2
import sys
import time
import PIL
import matplotlib.pyplot as plt
import urllib.request
from torchvision import transforms  # 


imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
imagenet_std = torch.tensor([0.229, 0.224, 0.225])

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(imagenet_mean, imagenet_std),
])


def generate_heatmap(model, img, class_index, size=(224, 224)):
    ################################ remove last 2 layers
    newmodel = torch.nn.Sequential(*(list(model.children())[:-2]))
    ################################ Load and preprocess image
    if isinstance(img, str):
        image = Image.open(img)
    else: 
        image = Image.fromarray(img)
    image.thumbnail(size, Image.LANCZOS)
    print("Image shape:", image.size)
    batch_t = transform(image)[None]
    ################################ extract weights from the last FC layer
    m_weights = model.fc.weight.data[class_index]
    print(m_weights.shape) #torch.Size([512])
    ################################ Forward
    fms = newmodel(batch_t)# [1, 512, 7, 7]
    fms = fms.squeeze(0)# [512,7,7]
    out_t = fms * m_weights[:, None, None]
    y_0 = torch.mean(out_t, dim=0) # AVG on channel dim. [512,7,7] --> [7,7]

    ################################ Viz
    if class_index == 0 : ti = "Gesund"
    if class_index == 1: ti = "Virale Pneumonie"
    else: ti = "Bakterielle Pneumonie"
    plt.figure(figsize=(30, 12))
    ax = plt.subplot(1, 3, 1)
    ax.axis("off")
    plt.title(ti)
    plt.imshow(img)

    ax = plt.subplot(1, 3, 2)
    ax.axis("off")
    plt.title(ti)
    plt.imshow(y_0.detach().numpy())

    ax = plt.subplot(1, 3, 3)
    plt.title(ti)
    ax.axis("off")
    ax.imshow(img)
    ax.imshow(y_0.detach().numpy(), alpha=0.6, interpolation="bilinear", cmap="magma",
            extent=(0,img.shape[0],img.shape[1],0))
    
#loading model:
# Laden des vortrainierten ResNet-34-Modells
resnet = models.resnet34(pretrained=True)

# Ersetzen der letzten Schicht durch einen FC-Layer mit 3 Ausgabeneuronen
num_classes = 3
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
model_zustand  = torch.load("resnet_model.pth")
resnet.load_state_dict(model_zustand)

#loading sample images
# Einlesen der Lerndaten aus einer .npz-Datei
data = np.load('learnset.npz')
images = data['bilder'][2000:2010]  # Annahme: 'images' enthält die Bilder
y = data['labels'] [2000:2010] # Annahme: 'labels' enthält die entsprechenden Labels

# Mischen der Daten
indices = np.arange(len(images))
np.random.shuffle(indices)
#images
images = np.expand_dims(images[indices], axis=-1).repeat(3, axis=-1)
#labels
y = y[indices]
labels = np.zeros((y.shape[0], num_classes))
np.put_along_axis(labels, y[:, None], 1, axis=1)
print("Daten vorbereitet")

#
resnet.eval()
number = 0
pic = images[number]
l = labels[number]
if l[0] == 1: l = 0
elif l[1] ==1: l = 1
else: l = 2
generate_heatmap(resnet, pic, l)
plt.show()

