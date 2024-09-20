import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


# set the recourse/dataset location
train_dir = 'Learnset\chest_xray\\train'
val_dir = 'Learnset\chest_xray\\val'
test_dir = 'Learnset\chest_xray\\test'

# store the shape of the x-ray pictures
IMGHeight = 128
IMGWidth = 128
Batch_Size = 32

# load the dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    color_mode='grayscale',
    batch_size=32,
    image_size=(256, 256)
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    color_mode='grayscale',
    batch_size=32,
    image_size=(256, 256)
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    color_mode='grayscale',
    batch_size=32,
    image_size=(256, 256)
)
'''

def load_images (filename):
    # Laden Sie die Arrays aus der Datei mit np.load
    data = np.load (filename)
    # Holen Sie sich die Arrays mit den entsprechenden Schlüsseln
    bilder = data ["bilder"]
    labels = data ["labels"]
    # Geben Sie eine Nachricht aus, um den Erfolg zu bestätigen
    print("Die Bilder und die Labels wurden aus", filename, "geladen")
    # Geben Sie die Arrays zurück
    return bilder, labels

X, labels = load_images('learnset.npz')

# Erstellen Sie einen TensorFlow-Datensatz aus Ihrer Liste mit tf.data.Dataset.from_tensor_slices
dataset = tf.data.Dataset.from_tensor_slices( (X, labels))

# Teilen Sie den Datensatz in Train, Validation und Test Datensätze mit tf.data.Dataset.take und tf.data.Dataset.skip
train_ds = dataset.take(5000) # Nehmen Sie die ersten 5000 Elemente für den Train Datensatz
val_ds = dataset.skip(5000).take(500) # Überspringen Sie die ersten 5000 Elemente und nehmen Sie die nächsten 500 Elemente für den Validation Datensatz
test_ds = dataset.skip(5500) # Überspringen Sie die ersten 5500 Elemente und nehmen Sie den Rest für den Test Datensatz

# Geben Sie die Größe jedes Teildatensatzes aus
print ("Train size:", len (train_ds))
print ("Validation size:", len (val_ds))
print ("Test size:", len (test_ds))

'''





#prepare the data
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size = AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)

'''
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
'''

schichten = [1, 2, 3, 4]
genauigkeiten = []

for anzahl in schichten:
    model = models.Sequential()
    model.add(layers.experimental.preprocessing.Rescaling(1. / 255))
    model.add(layers.Conv2D(32, 3, activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    for i in range(1, anzahl):
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    # Kompilieren und Trainieren #rmsprop
    model.compile(optimizer='sgd',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(train_ds, epochs=10, validation_data=test_ds)

    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    genauigkeiten.append(test_acc)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()
    plt.close()

plt.plot(schichten, genauigkeiten, marker="o")
plt.title("Genauigkeit des CNN in Abhängigkeit von der Anzahl der Schichten")
plt.xlabel("Anzahl der Schichten")
plt.ylabel("Genauigkeit")

# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)
plt.show()
