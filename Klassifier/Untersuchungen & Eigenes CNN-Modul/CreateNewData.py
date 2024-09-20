import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir=""
#Data Directory to where the data is saved
categories=[""]
#categories the data are classified to
img_size= 
#size of images
training_data=[]
def create_training_data():#create training-data function
    for category in categories:#looping through the categories based on their index
        path=os.path.join(data_dir, category)
        class_num= categories.index(category)
        for img in os.listdir(path):#looping through the images and resizing them to img_size
            try:#for errors to not completely stop the program
                img_array=cv2.imread(os.path.join(path, img))
                array_new=cv2.resize(img_array, (img_size, img_size))
                training_data.append([array_new, class_num])
#reading in the data from where its stored, resize them to img_size and append or connect them to the function
                datagen=ImageDataGenerator(
                    rotation_range=20,#rotation
                    width_shift_range=0.2,
                    height_shift_range=0.2,#verschiebung
                    shear_range=0.2,#scherung
                    zoom_range=0.2,#zoomen
                    horizontal_flip=True#flippen
                    fill_mode="nearest",#füllen leere pixel mit nähestem
                )
                img_array=img_array.reshape((1,)+img_array.shape+(1,))#damit neues array kompatibel mit IDG ist 
                i=0
                for batch in datagen.flow(img_array, batch_size=1):
                    augmented_img=batch[0].astype("uint8").reshape(img_size, img_size)
                    training_data.append([augmented_img, class_num])
                    i+=1
                    #für den for loop
                    if i>=4:
                        break
                #dreimaliges vervielfältigen der Daten
                
            except Exception as e:
                print("Bild fehlerhaft") #ersetzen eines Error mit dieser Fehlermeldung

create_training_data()

random.shuffle(training_data)
#shuffling the  Data to stop Overfitting
X=[] #features
y=[] # labels to the features
for feature, label in training_data:
    X.append(feature) #giving X and y their values
    y.append(label)
X=np.array(X).reshape(-1, img_size, img_size, 1)#reshaping it to img size, 1 for grey

pickle_out=open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()
#storing x
pickle_out=open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
#storing y
pickle_in=open("X.pickle","rb")
X=pickle.load(pickle_in)
#storing X and y in the pickle file while still training the CNN, don't bave to restart training
