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


#create model
# Laden des vortrainierten VGG19-Modells
vgg19 = models.vgg19(pretrained=True)

# Anzahl der Klassen für die neue Ausgabeschicht
num_classes = 3

# Ersetzen der letzten Schicht (classifier) des VGG19-Modells
# Die letzte Schicht ist ein Linear Layer mit 4096 Eingängen
vgg19.classifier[6] = nn.Linear(4096, num_classes)

# Laden des Modellzustands, falls vorhanden
modell_zustand = torch.load('vgg19_modelLast.pth')
vgg19.load_state_dict(modell_zustand)



# Einlesen der Lerndaten aus einer .npz-Datei
data = np.load('learnset.npz')
images = data['bilder']  # Annahme: 'images' enthält die Bilder
y = data['labels']  # Annahme: 'labels' enthält die entsprechenden Labels

# Mischen der Daten
#indices = np.arange(len(images))
#np.random.shuffle(indices)
#np.save("vgg19_indices", indices)
indices = np.load("vgg19_indices.npy")
print("Indizes gespeichert")
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
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.1, shuffle=False
)
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


# Trainieren des Netzwerks mit den Batches
optimizer = torch.optim.Adam(vgg19.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

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
    
    def overwrite(self, text):
        sys.stdout.write(text)
        sys.stdout.flush()
    
    def end(self):
        sys.stdout.write('\n')
        sys.stdout.flush()
'''        
#eval the model
vgg19.eval()  # Evaluationsmodus
correct = 0
total = 0
loss = 0
with torch.no_grad():  # Keine Gradientenberechnung
    for images, labels in val_loader:
        outputs = vgg19(images)
        loss += float(criterion(outputs, labels))
        _, predicted = torch.max(outputs.data, 1)
        _, labpred = torch.max(labels, 1)
        total += labels.size(0)
        correct += (predicted == labpred).sum().item()

accuracy = 100 * correct / total
loss = loss / total
print(f'Epoch {0} abgeschlossen. Genauigkeit: {accuracy:.4f}% Loss: {loss:.4f}')
'''
prevacc = 84.1296928327645
#prevacc = False
# Trainieren und Evaluieren des Netzwerks
for epoch in range(4,50):  # Anzahl der Epochen anpassen
    vgg19.train()  # Trainingsmodus
    print("Epoche: ", epoch)
    progBar = progress(len(train_loader))
    history = []
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = vgg19(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        #display accuracy
        _, predicted = torch.max(outputs.data, 1)
        _, labpred = torch.max(labels, 1)
        total = labels.size(0)
        correct = (predicted == labpred).sum().item()
        acc = correct/total
        progBar.update(acc, float(loss))
        history.append((acc, float(loss)))
    progBar.end()
    vgg19.eval()  # Evaluationsmodus
    correct = 0
    total = 0
    loss = 0
    progEval = progress(len(val_loader))
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
        for images, labels in val_loader:
            outputs = vgg19(images)
            loss += float(criterion(outputs, labels))
            _, predicted = torch.max(outputs.data, 1)
            _, labpred = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == labpred).sum().item()
            progEval.update(correct/total, loss)
            for pred, lab in zip(predicted.tolist(), labpred.tolist()):
                confmat[pred, lab] += 1
    progEval.end()
    accuracy = 100 * correct / total
    loss = loss / total
    #Abspeicher der genauigkeit
    with open("historyVGG19.txt", "a") as file:
        file.write("Epoche : "+str(epoch)+" acc.: "+str(accuracy)+" loss: "+str(loss)+" hist.: "+str(history)+" confmat: "+str(confmat)+"\n")

    progEval.overwrite('Epoch {0:.0f} abgeschlossen. Genauigkeit: {1:.4f}% Loss: {2:.4f} '.format(epoch, accuracy, loss)+str(confmat)+'\n')
    
    if prevacc is False or prevacc < accuracy:
        torch.save(vgg19.state_dict(), 'vgg19_model.pth')
        print("Modell erfolgreich gespeichert! (accuracy:", accuracy,")")
        prevacc = accuracy
    torch.save(vgg19.state_dict(), 'vgg19_modelLast.pth')
    print("Modell erfolgreich gespeichert(last)! (accuracy:", accuracy,")")

# Speichern des trainierten Modells
#torch.save(vgg19.state_dict(), 'vgg192_model.pth')
print("Modell erfolgreich gespeichert!")
