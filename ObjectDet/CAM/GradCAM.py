import torch
import torch.nn as nn
from torch.utils import data
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import urllib.request
from PIL import Image
import json
from functools import partial 
import cv2
from sklearn.model_selection import train_test_split
import time
import sys


classes = ["Gesund", "Virale Pneumonie", "Bakterielle Pneumonie"]
num_classes = len(classes)
#loading the images and labels
# Einlesen der Lerndaten aus einer .npz-Datei
data = np.load('learnset.npz')
images = data['bilder']  # Annahme: 'images' enthält die Bilder
y = data['labels']  # Annahme: 'labels' enthält die entsprechenden Labels

# Mischen der Daten
indices = np.arange(len(images))
np.random.shuffle(indices)
images = np.expand_dims(images[indices], axis=-1).repeat(3, axis=-1)


y = y[indices]
labels = np.zeros((y.shape[0], num_classes))
np.put_along_axis(labels, y[:, None], 1, axis=1)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Aufteilen in Trainings- und Validierungsdatensätze
_, val_images, _, val_labels = train_test_split(
    images, labels, test_size=0.01, random_state=42
)
'''
# Konvertieren Sie die Bilder und Labels in Tensoren
tensor_images = torch.stack([transform(image) for image in train_images])
tensor_labels = torch.tensor(train_labels)

# Erstellen Sie ein Dataset und DataLoader für die Trainingsdaten
train_dataset = TensorDataset(tensor_images, tensor_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# Konvertieren Sie die Validierungsbilder und -labels in Tensoren
tensor_val_images = torch.stack([transform(image) for image in val_images])
tensor_val_labels = torch.tensor(val_labels)

# Erstellen Sie ein Dataset und DataLoader für die Validierungsdaten
val_dataset = TensorDataset(tensor_val_images, tensor_val_labels)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
'''


############## modell ###########
# Laden des vortrainierten VGG19-Modells
vgg19 = models.vgg19(pretrained=True)

# Anzahl der Klassen für die neue Ausgabeschicht
num_classes = 3

# Ersetzen der letzten Schicht (classifier) des VGG19-Modells
# Die letzte Schicht ist ein Linear Layer mit 4096 Eingängen
vgg19.classifier[6] = nn.Linear(4096, num_classes)

# Laden des Modellzustands, falls vorhanden
modell_zustand = torch.load('vgg192_model.pth')
vgg19.load_state_dict(modell_zustand)


#function to preprocess image
def load_image(img, size=None):
    if isinstance(img, str):
        img = Image.open(img)
    else: img = Image.fromarray(img)
    img = img.convert(mode='RGB')
    
    if size is not None:
        img = img.resize(size)
    return img

def preprocess_image(pil_img):
    tensor = transforms.ToTensor()(pil_img)
    tensor = torch.unsqueeze(tensor, dim=0)  # (B, C, H, W)
    
    return tensor

def batch_load(img, size=[224,224], viz=True):
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
    imagenet_std = torch.tensor([0.229, 0.224, 0.225])
    # input_size = [224, 224]

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    # img = Image.open("dog1.jpg")
    # img = Image.open('./data/Elephant/1.png')
    img = load_image(img, size=size)
    #if viz: display(img)
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0) # shape : [1,3,224,224]
    img
    return batch_t

test_image = batch_load(val_images[0])
print("image-label: ", val_labels[0])



class VGG(nn.Module):
    def __init__(self, model):
        super(VGG, self).__init__()
        # get the pretrained VGG19 network
        self.vgg = model

        # disect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:36]
        
        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        # get the classifier of the vgg19
        self.classifier = self.vgg.classifier
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        self.activations = x
        
        # register the hook at the feature map that we are interested in.
        h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x
    
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)
    
# initialize the VGG model & set the evaluation mode
vgg = VGG(vgg19)
vgg.eval()

# define Infer function w/ top 3 classes
def infer(model, input): #vgg  pred
    # get the most likely prediction of the model
    pred = model(input) # .argmax(dim=1)
    # print(pred.shape)
    #print the top 3 classes predicted by the model
    topk=3
    _, indices = torch.sort(pred, descending=True)
    percentage = torch.nn.functional.softmax(pred, dim=1)[0] * 100
    print([(classes[idx], idx.data, percentage[idx].item()) for idx in indices[0][:topk]])
    return pred


# forward
pred = infer(vgg, test_image)
print("Prediction:", pred)

def get_gradcam_plus_plus2(model, input, classPred=0):
    # forward pass
    pred = model(input)
    
    # 1. get the gradient of the output with respect to the parameters of the model
    pred[:, classPred].backward(retain_graph=True)

    # 2. get the activations of the last convolutional layer
    activations = model.get_activations(input).detach()

    # 3. pull the gradients out of the model
    gradients = model.get_activations_gradient()

    # 4. calculate the weights using the new method for Grad-CAM++
    weights = model.get_gradcam_plus_weights(gradients, activations, classPred)

    # 5. weight the channels by corresponding gradients
    for i in range(512):
        activations[:, i, :, :] *= weights[i]
    
    # 6. average the channels of the activations to get the heatmap
    heatmap = torch.mean(activations, dim=1).squeeze()

    # 7. apply ReLU to the heatmap
    heatmap = F.relu(heatmap)

    # 8. normalize the heatmap
    heatmap /= torch.max(heatmap)

    # draw the heatmap
    plt.matshow(heatmap.squeeze())
    plt.show()

    return heatmap

def get_gradcam_plus_plus(model, input, classPred=0):
    # Forward pass
    pred = model(input)

    # Get the gradient of the output with respect to the parameters of the model
    pred[:, classPred].backward(retain_graph=True)

    # Get the activations of the last convolutional layer
    activations = model.get_activations(input).detach()

    # Pull the gradients out of the model
    gradients = model.get_activations_gradient()

    # Calculate the weights using Grad-CAM++
    alpha_numer = gradients.pow(2)
    alpha_denom = gradients.pow(2).mul(2) + activations.mul(gradients.pow(3)).sum(dim=[2, 3], keepdim=True)
    alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
    alpha = alpha_numer.div(alpha_denom + 1e-7)
    positive_gradients = F.relu(pred[:, classPred].exp()).detach()
    weights = (alpha * positive_gradients).sum(dim=[2, 3])

    # Expand weights to match the channel dimension
    weights_expanded = weights.view(1, -1, 1, 1)

    # Weight the channels by corresponding gradients
    activations *= weights_expanded

    # Average the channels of the activations to get the heatmap
    heatmap = torch.mean(activations, dim=1).squeeze()

    # Apply ReLU to the heatmap
    heatmap = F.relu(heatmap)

    # Normalize the heatmap
    heatmap /= torch.max(heatmap)

    # Display the heatmap (you can customize this part)
    plt.matshow(heatmap.squeeze())
    plt.show()

    return heatmap






def get_cam(pred, input, classPred = 0):
    # 1. get the gradient of the output with respect to the parameters of the model
    pred[:, classPred].backward()

    # 2. get the activations of the last convolutional layer
    activations = vgg.get_activations(input).detach()

    # 3. pull the gradients out of the model
    gradients = vgg.get_activations_gradient()

    # 4. pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])


    # 5. weight the channels by corresponding gradients
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]
    #   average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()

    # 6. relu on top of the heatmap
    #   expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)

    # 7. normalize the heatmap
    heatmap /= torch.max(heatmap)

    # draw the heatmap
    plt.matshow(heatmap.squeeze())
    return heatmap

classPred = val_labels[0].tolist().index(1)
heatmap = get_cam(pred, test_image, classPred)
#pred = infer(vgg, test_image)
#print("Prediction:", pred)
heatmappp = get_gradcam_plus_plus(vgg, test_image, classPred)

def overlayCAM(heatmap, origin_img_path, title, outPath):
    if isinstance(origin_img_path, str):
        img = cv2.imread(origin_img_path)
    else: img = origin_img_path
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite(outPath, superimposed_img)
    plt.imshow(superimposed_img)
    
    
overlayCAM(heatmap, val_images[0], classes[classPred], './map.jpg')
overlayCAM(heatmappp, val_images[0], classes[classPred], './map2.jpg')