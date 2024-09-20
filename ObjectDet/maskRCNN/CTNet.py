import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

# Laden des ResNet50-Modells ohne den obersten (top) Layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Hinzufügen neuer Schichten am Ende des Netzwerks
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Zusammenstellen des Modells
model = Model(inputs=base_model.input, outputs=predictions)

# Datengenerator für die Datenvorbereitung und -vervielfältigung
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Trainingsdaten einlesen
train_generator = train_datagen.flow_from_directory(
    'path_to_train_data',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

# Modell für das Training konfigurieren
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Training des Modells
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=num_epochs)

# Modell speichern
model.save('resnet50_model.h5')