import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils import data
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

#class for the progress
class progress:
    def __init__(self, maxVal, step = 0, accuracy = 0, loss = 0, bar_length = 60):
        """
        Zeigt oder aktualisiert einen Ladebalken in der Konsole.
        
        :param progress: Ein Float-Wert zwischen 0 und 1, der den Fortschritt angibt.
        :param accuracy: Ein Float-Wert, der die Genauigkeit angibt.
        :param loss: Ein Float-Wert, der den Verlust angibt.
        """
        self.step = step
        self.bar_length = bar_length
        self.loss = loss
        self.accuracy = 0
        self.avracc = 0
        self.maxVal = maxVal
        self.time = time.time()
        self.initTime = time.time()
        progress =  step/maxVal
        block = int(round(bar_length * progress))
        text = "\rProgress: [{0}] Step: {1:.0f}/{2:.0f} = {3:.2f}% acuracy: ---- loss: ----".format(
            "=" * block + ">"+"-" * (bar_length - block-1), step, maxVal, progress * 100)
        sys.stdout.write(text)
        sys.stdout.flush()
    
    def update(self, acc, loss):
        self.step += 1
        progress =  self.step/self.maxVal
        block = int(round(self.bar_length * progress))
        stepTime = time.time() - self.time
        self.time = time.time()
        self.avracc = (self.avracc * (self.step-1) + acc) / self.step
        text = "\rProgress: [{0}] Step: {1:.0f}/{2:.0f} = {3:.2f}% acc: {4:.4f}% (avr: {9:.2f}%) loss: {5:.4f} -- {6:.4f}s/step ({7:.2f}min/{8:.2f}min)".format(
            "=" * block + ">"+"-" * (self.bar_length - block-1), self.step, self.maxVal, progress * 100, acc*100, loss, stepTime, (time.time()-self.initTime)/60, ((time.time()-self.initTime) + stepTime*(self.maxVal-self.step))/60, self.avracc)
        sys.stdout.write(text)
        sys.stdout.flush()

#num classes
num_classes = 3

#Daten laden:
# Einlesen der Lerndaten aus einer .npz-Datei
data = np.load('learnset.npz')
images = data['bilder']  # Annahme: 'images' enthält die Bilder
y = data['labels']  # Annahme: 'labels' enthält die entsprechenden Labels

# Mischen der Daten
indices = np.load("vgg19_indices.npy")
print("Indizes geladen")
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
#_, val_images, _, val_labels = train_test_split(images, labels, test_size=0.07, shuffle=False)

# Konvertieren Sie die Validierungsbilder und -labels in Tensoren
tensor_val_images = torch.stack([transform(image) for image in images])
tensor_val_labels = torch.tensor(labels)

# Erstellen Sie ein Dataset und DataLoader für die Validierungsdaten
val_dataset = TensorDataset(tensor_val_images, tensor_val_labels)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)



#create model
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
    


# Trainieren des Netzwerks mit den Batches
optimizer = torch.optim.Adam(vgg19.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

#testen und evaluieren des models:
vgg19.eval()  # Evaluationsmodus
correct = 0
total = 0
loss = 0
'''
confmat: Konfidenz Matrix
->label; v predicted
        Gesund  VirPneu  BakPneu
  Gesund    0       0       0
 VirPneu    0       0       0
 BakPneu    0       0       0
'''
confmat = np.array([[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]])

with torch.no_grad():  # Keine Gradientenberechnung
    #progress erstellen
    prog = progress(len(val_loader))
    for images, labels in val_loader:
        outputs = vgg19(images)
        loss += float(criterion(outputs, labels))
        _, predicted = torch.max(outputs.data, 1)
        _, labpred = torch.max(labels, 1)
        total += labels.size(0)
        correct += (predicted == labpred).sum().item()
        for pred, lab in zip(predicted.tolist(), labpred.tolist()):
            confmat[pred, lab] += 1
        prog.update(correct/total, loss)

print("confmat: ", confmat)