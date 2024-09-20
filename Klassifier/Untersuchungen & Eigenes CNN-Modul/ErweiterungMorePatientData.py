import tensorflow as tf
import numpy
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Datensatz Speicherort
train_dir = 'Learnset\chest_xray\\train'
val_dir = 'Learnset\chest_xray\\val'
test_dir = 'Learnset\chest_xray\\test'

# Größe der Bilder abspeichern
IMGHeight = 128
IMGWidth = 128
Batch_Size = 32

# Datensatz laden
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

# Daten vorbereiten
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Dataset splitten
train_label = []
train_images = []
test_label = []
test_images = []

for train_batch_images, train_batch_labels in train_ds.as_numpy_iterator():
    train_label.extend(train_batch_labels)
    train_images.extend(train_batch_images)
for test_batch_images, test_batch_labels in test_ds.as_numpy_iterator():
    test_label.extend(test_batch_labels)
    test_images.extend(test_batch_images)

# Numpy Array daraus erstellen
train_label = numpy.array(train_label)
train_images = numpy.array(train_images)
test_label = numpy.array(test_label)
test_images = numpy.array(test_images)


# Für Testdaten
# Alters Liste für Pneumonia Patienten erstellen
liste_pneumonia_test = numpy.random.normal(loc=85, scale=23, size=700)
liste_pneumonia_test = [age for age in liste_pneumonia_test if age < 90]
while len(liste_pneumonia_test) < 390 or len(liste_pneumonia_test) > 390:
    liste_pneumonia_test = numpy.random.normal(loc=85, scale=23, size=700)
    liste_pneumonia_test = [age for age in liste_pneumonia_test if age < 90]
liste_pneumonia_test = [age.round() for age in liste_pneumonia_test if age < 90]

# Alters Liste für Normale Patienten erstellen
liste_normal_test = numpy.random.normal(loc=0, scale=23, size=500)
liste_normal_test = [age for age in liste_normal_test if age < 90]
liste_normal_test = [age for age in liste_normal_test if age > 0]
while len(liste_normal_test) < 234 or len(liste_normal_test) > 234:
    liste_normal_test = numpy.random.normal(loc=0, scale=23, size=500)
    liste_normal_test = [age for age in liste_normal_test if age < 90]
    liste_normal_test = [age for age in liste_normal_test if age > 0]
liste_normal_test = [age.round() for age in liste_normal_test if age < 90]

# Für Trainingsdaten
# Alters Liste für Pneumonia Patienten erstellen
liste_pneumonia = numpy.random.normal(loc=85, scale=23, size=6580)
liste_pneumonia = [age for age in liste_pneumonia if age < 90]
while len(liste_pneumonia) < 3875 or len(liste_pneumonia) > 3875:
    liste_pneumonia = numpy.random.normal(loc=85, scale=23, size=6580)
    liste_pneumonia = [age for age in liste_pneumonia if age < 90]
liste_pneumonia = [age.round() for age in liste_pneumonia if age < 90]

# Alters Liste für Normale Patienten erstellen
liste_normal = numpy.random.normal(loc=0, scale=23, size=2700)
liste_normal = [age for age in liste_normal if age < 90]
liste_normal = [age for age in liste_normal if age > 0]
while len(liste_normal) < 1341 or len(liste_normal) > 1341:
    liste_normal = numpy.random.normal(loc=0, scale=23, size=2700)
    liste_normal = [age for age in liste_normal if age < 90]
    liste_normal = [age for age in liste_normal if age > 0]
liste_normal = [age.round() for age in liste_normal if age < 90]

train_liste = [liste_normal, liste_pneumonia]
test_liste = [liste_normal_test, liste_pneumonia_test]

# Trainingsaltersliste
train_personParameter = []
n, p = 0, 0
for label in train_label:
    # Hinzufügen der Normalen parameter
    if label == 0:
        train_personParameter.append(liste_normal[n])
        n += 1
    # Hinzufügen der Pneumonie parameter
    else:
        train_personParameter.append(liste_pneumonia[p])
        p += 1
train_personParameter = numpy.array(train_personParameter)

# Test Alters Liste
test_personParameter = []
n, p = 0, 0
for l in test_label:
    # Hinzufügen der Normalen parameter
    if l == 0:
        test_personParameter.append(liste_normal_test[n])
        n += 1
    # Hinzufügen der Pneumonie parameter
    else:
        test_personParameter.append(liste_pneumonia_test[p])
        p += 1
test_personParameter = numpy.array(test_personParameter)

# Model für die Röntgenbilder
model1 = models.Sequential([
    layers.Input(shape=(256, 256, 1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(11)
])

# Model für das Alter
model2 = models.Sequential([
    layers.Input(shape=(1,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="relu")
])

# Kombinierter Layer
combined_layer = layers.Concatenate()([model1.output, model2.output])

# Dense layer für die beiden verbundenen layer
dense_layer = layers.Dense(32, activation="relu")(combined_layer)
output_layer = layers.Dense(1, activation="sigmoid")(dense_layer)

model = tf.keras.Model(inputs=[model1.input, model2.input], outputs=output_layer)

# Kompilieren und Trainieren
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Trainingsdaten Reshapen
cnn_data = numpy.stack(train_images, axis=0)
cnn_data = cnn_data.reshape(-1, 256, 256, 1)
dense_data = numpy.array(train_personParameter).reshape(-1, 1) # Umwandlung in ein 2D-Array

# Testdaten Reshapen
cnn_data_test = numpy.stack(test_images, axis=0)
cnn_data_test = cnn_data_test.reshape(-1, 256, 256, 1)
dense_data_test = numpy.array(test_personParameter).reshape(-1, 1) # Umwandlung in ein 2D-Array

# Trainieren des Netzes
history = model.fit([cnn_data, dense_data], train_label, epochs=8, batch_size=512, validation_data=([cnn_data_test, dense_data_test], test_label))

# Auswerten des Netzes
test_loss, test_acc = model.evaluate([cnn_data_test, dense_data_test], test_label, verbose=2)
