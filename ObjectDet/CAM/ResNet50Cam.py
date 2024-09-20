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


# Laden des vortrainierten ResNet-34-Modells
resnet = models.resnet50(pretrained=True)

# Ersetzen der letzten Schicht durch einen FC-Layer mit 3 Ausgabeneuronen
num_classes = 3
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
#model_zustand  = torch.load("resnet_model2.pth")
#resnet.load_state_dict(model_zustand)
print("Netz geladen")

# Einlesen der Lerndaten aus einer .npz-Datei
data = np.load('learnset.npz')
images = data['bilder']  # Annahme: 'images' enthält die Bilder
y = data['labels']  # Annahme: 'labels' enthält die entsprechenden Labels

# Mischen der Daten
indices = np.arange(len(images))
np.random.shuffle(indices)
np.save("indicesResNet50", indices)
print("indices gespeichert")
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
    images, labels, test_size=0.1, random_state=42
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

print("Data loaded and processed")

# Trainieren des Netzwerks mit den Batches
optimizer = torch.optim.Adam(resnet.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

class progress:
    def __init__(self, maxVal, step = 0, accuracy = 0, loss = 0, bar_length = 80):
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
resnet.eval()  # Evaluationsmodus
correct = 0
total = 0
loss = 0
with torch.no_grad():  # Keine Gradientenberechnung
    for images, labels in val_loader:
        outputs = resnet(images)
        loss += float(criterion(outputs, labels))
        _, predicted = torch.max(outputs.data, 1)
        _, labpred = torch.max(labels, 1)
        total += labels.size(0)
        correct += (predicted == labpred).sum().item()

accuracy = 100 * correct / total
loss = loss / total
print(f'Epoch {0} abgeschlossen. Genauigkeit: {accuracy:.4f}% Loss: {loss:.4f}')

# Trainieren und Evaluieren des Netzwerks
for epoch in range(10):  # Anzahl der Epochen anpassen
    resnet.train()  # Trainingsmodus
    print("Epoche: ", epoch)
    progBar = progress(len(train_loader))
    history = []
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = resnet(images)
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
    
    resnet.eval()  # Evaluationsmodus
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():  # Keine Gradientenberechnung
        for images, labels in val_loader:
            outputs = resnet(images)
            loss += float(criterion(outputs, labels))
            _, predicted = torch.max(outputs.data, 1)
            _, labpred = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == labpred).sum().item()
    
    accuracy = 100 * correct / total
    loss = loss / total
    with open("history.txt", "a") as file:
        file.write("Epoche : "+str(epoch)+" acc.: "+str(accuracy)+" loss: "+str(loss)+" hist.: "+str(history)+"\n")

    print(f'Epoch {epoch+1} abgeschlossen. Genauigkeit: {accuracy:.4f}% Loss: {loss:.4f}')
    proceed = input("Do you wanna proceed: y/n")
    if proceed == "n": break

# Speichern des trainierten Modells
torch.save(resnet.state_dict(), 'resnet2_model.pth')
print("Modell erfolgreich gespeichert!")
'''
prevacc = False
# Trainieren und Evaluieren des Netzwerks
for epoch in range(0,50):  # Anzahl der Epochen anpassen
    resnet.train()  # Trainingsmodus
    print("Epoche: ", epoch)
    progBar = progress(len(train_loader))
    history = []
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = resnet(images)
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
    resnet.eval()  # Evaluationsmodus
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
            outputs = resnet(images)
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
    with open("historyResNet50.txt", "a") as file:
        file.write("Epoche : "+str(epoch)+" acc.: "+str(accuracy)+" loss: "+str(loss)+" hist.: "+str(history)+" confmat: "+str(confmat)+"\n")

    progEval.overwrite('Epoch {0:.0f} abgeschlossen. Genauigkeit: {1:.4f}% Loss: {2:.4f} '.format(epoch+1, accuracy, loss)+str(confmat)+'\n')
    
    if prevacc is False or prevacc < accuracy:
        torch.save(resnet.state_dict(), 'resnet50_model.pth')
        print("Modell erfolgreich gespeichert! (accuracy:", accuracy,")")
    prevacc = accuracy

# Speichern des trainierten Modells
#torch.save(vgg19.state_dict(), 'vgg192_model.pth')
print("Modell erfolgreich gespeichert!")


'''
# Ändern der Bildgröße auf 224x224
imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
imagenet_std = torch.tensor([0.229, 0.224, 0.225])
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(imagenet_mean, imagenet_std),
])

# Trainieren des Netzwerks mit den Bildern und Labels
optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):  # Anzahl der Epochen anpassen
    for i in range(len(images)):
        resized_image = cv2.resize(images[i], (224, 224), interpolation=cv2.INTER_AREA)
        rgb_image = np.zeros((224, 224, 3), dtype=np.uint8)
        rgb_image[:, :, 0] = resized_image  # Roter Kanal
        rgb_image[:, :, 1] = resized_image  # Grüner Kanal
        rgb_image[:, :, 2] = resized_image  # Blauer Kanal
        image = Image.fromarray(rgb_image)  # Annahme: images[i] ist ein NumPy-Array
        image = transform(image)
        label = [[0, 0, 0]]
        label[0][labels[i] ] = 1        
        label = torch.tensor(label, dtype=torch.float)

        optimizer.zero_grad()
        output = resnet(image.unsqueeze(0))
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}')

print("Training abgeschlossen!")

#Speichern des model
torch.save(resnet.state_dict(), 'resnet_model.pth')
print("Modell erfolgreich gespeichert!")'''