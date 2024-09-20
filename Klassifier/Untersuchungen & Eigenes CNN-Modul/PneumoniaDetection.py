from Neural import Network
from Neural import Layer
from Neural import Utility
from Neural import BatchNorm
#from musterloesung import *
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence 
import cv2



# Definieren Sie die Funktion load_images, um die Bilder und die Labels aus einer .npz-Datei zu laden
def load_images (filename):
  # Laden Sie die Arrays aus der Datei mit np.load
  data = np.load(filename)
  # Holen Sie sich die Arrays mit den entsprechenden Schlüsseln
  bilder = data["bilder"]
  labels = data["labels"]
  # Geben Sie eine Nachricht aus, um den Erfolg zu bestätigen
  print ("Die Bilder und die Labels wurden aus", filename, "geladen")
  # Geben Sie die Arrays zurück
  return bilder, labels

#Hier muss der Absolute Pfad der Trainingsdaten als 'npz' Datei Angegeben werden (Numpy Dateityp)
X, labels = load_images("learnset.npz")

test = 100

X_train = X[test:]
X_test = X[:test]
y_train = labels[test:]
y_test = labels[:test]



X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
#X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])

X_train = X_train/255
X_test = X_test/255

utility = Utility()

# train validation split
X_train_new, X_val, y_train_new, y_val = utility.train_test_split(X_train, y_train, test_ratio=0.2, seed=42)

Y_1hot_train, _ = utility.onehot(y_train_new)
print("Datensatz wurde gesplittet in Validierungs, Trainings und Testdaten")
print("y: ",Y_1hot_train)

input_shape = X_train_new.shape[1:]
output_dim = Y_1hot_train.shape[1]

input_shape, output_dim


print("Input: ", input_shape)
#Eingabegröße

#input_shape = (1, 254, 254) #256x256 Pixel Bild mit 1 Farbchannel (Grayscale)


net = Network()
#net = CNN()

#Erstelle das Netz

#Eingabe Schicht:
net.add(net.Input(input_shape=input_shape))
'''
#CNN/Pooling Schichten
net.add(Layer.Conv2D(1, 3, activation_type='relu', padding = 'same'))

net.add(Layer.Pooling2D((2,2), (2,2), pool_type='mean', padding="same"))

net.add(Layer.Conv2D(2, 3, activation_type='relu', padding='same'))

net.add(Layer.Pooling2D((2,2), (1,1), pool_type='max'))

#Dens Layer
net.add(Layer.Flatten())

net.add(Layer.Dense(64, activation_type = 'relu'))

net.add(Layer.Dense(2, activation_type = 'softmax'))



'''
net.add(Layer.Conv2D(1, (3,3) , stride = (1, 1) , padding = 'same' , activation_type = 'relu' ))
net.add(BatchNorm())
net.add(Layer.Pooling2D((2,2) , stride = (2, 2) , padding = 'same', pool_type = "max"))
net.add(Layer.Conv2D(2 , (3,3) , stride = (1, 1) , padding = 'same' , activation_type = 'relu'))
net.add(Layer.Dropout(0.1))
net.add(BatchNorm())
net.add(Layer.Pooling2D((2,2) , stride = (2, 2) , padding = 'same', pool_type = "max"))
net.add(Layer.Conv2D(1 , (3,3) , stride = (1, 1) , padding = 'same' , activation_type = 'relu'))
net.add(BatchNorm())
net.add(Layer.Pooling2D((2,2) , stride = (2, 2) , padding = 'same'))
net.add(Layer.Conv2D(2 , (3,3) , stride = (1, 1) , padding = 'same' , activation_type = 'relu'))
net.add(Layer.Dropout(0.2))
net.add(BatchNorm())
net.add(Layer.Pooling2D((2,2) , stride = (2, 2) , padding = 'same', pool_type = "max"))
net.add(Layer.Conv2D(2 , (3,3) , stride = (1, 1) , padding = 'same' , activation_type = 'relu'))
net.add(Layer.Dropout(0.2))
net.add(BatchNorm())
net.add(Layer.Pooling2D((2,2) , stride = (2, 2) , padding = 'same', pool_type = "max"))
net.add(Layer.Flatten())
net.add(Layer.Dense(neurons = 128 , activation_type = 'relu'))
net.add(Layer.Dropout(0.2))
net.add(Layer.Dense(neurons = 3 , activation_type = 'sigmoid'))

#test
'''
#CNN/Pooling Schichten
net = CNN()
net.Input(input_shape = input_shape)
net.add(Conv2D(32, kernel_size=(3, 3), activation_type='relu'))

net.add(Pooling2D((2,2), (2,2), pool_type='max'))

net.add(Conv2D(32, 3, p='same', activation_type='relu'))

net.add(Pooling2D((2,2), (1,1), pool_type='max'))

#Dens Layer
net.add(Flatten())

net.add(Dense(64, activation_type = 'relu'))

net.add(Dense(3, activation_type = 'softmax'))


net.summary()
net.compile(cost_type="cross-entropy", optimizer_type="adam")

batch_size = 2
epochs = 1
lr = 0.05

net.fit(X_train_new, Y_1hot_train, epochs, batch_size, lr, X_val, y_val)



model = CNN()

model.add(model.Input(input_shape=input_shape))

model.add(Conv2D(32, kernel_size=(5, 5), p='same', activation_type="relu"))

model.add(Pooling2D(pool_size=(2, 2), p = 'valid', pool_type = 'max'))

model.add(Flatten())

model.add(Dropout(0.4))

model.add(Dense(3, activation_type="softmax"))

model.summary()

'''
#compiliere das Netz
net.summary("Pneumonia - CNN")


batch_size = 32
epochs = 8
lr = 0.05

net.compile(kosten_typ="cross-entropy", optimierer_typ="rmsprop")


#lerne das Netz an
net.fit(X_train_new, Y_1hot_train, epochs, batch_size, lr, X_val, y_val)

net.save("pneumoniaParameter.pkl")

net.loss_plot
net.accuracy_plot




'''
batch_size = 1 
epochs = 10
lr = 0.05

model.compile(cost_type="cross-entropy", optimizer_type="adam")


# In[ ]:


LR_decay = LearningRateDecay()

model.fit(X_train_new, Y_1hot_train, epochs=epochs, batch_size=batch_size, lr=lr, X_val=X_val,
        y_val=y_val, verbose=1, lr_decay=LR_decay.constant, lr_0=lr)
'''