#Import TensorFlow dan library lainnya
import os
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers, models

#Memanggil base directory project
BASE_DIR = BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#Memanggil directory model dan dataset
data_model = BASE_DIR + "/model"
dataset_dir = BASE_DIR + "/dataset"

#Mengambil semua dataset
data_dir = pathlib.Path(dataset_dir)

#Menentukan batch size dan size gambar
batch_size = 32
size = 180

#Memecah data untuk digunakan sebagai data latih 80%
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(size, size),
    batch_size=batch_size
)

#Memecah data untuk digunakan sebagai validasi data 20%
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(size, size),
    batch_size=batch_size
)

#Menentukan nama setiap klasifikasi dari nama klasifikasi yang diambil dari data latih
class_names = train_dataset.class_names

#Normalisasi data latih
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

normalized_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_dataset))
first_image = image_batch[0]

#Pengambilan jumlah klasifikasi dari banyaknya nama klasifikasi
num_classes = len(class_names)

#Membuat object sequential
model = models.Sequential()

#Operasi konvolusi & Pooling
model.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=(size, size, 3)))
model.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
model.add(layers.MaxPooling2D())

#Flattening
model.add(layers.Flatten())

#Dense
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes))

#Kompilasi model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

#Menampilkan model yang sudah dikompilasi 
model.summary()

#Menentukan jumlah epoch, satu epoch adalah ketika seluruh dataset terlewati baik forward ataupun backward melalui jaringan syaraf 
epochs = 10

#Proses training
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)

#Proses menampilkan grafik
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

#Grafik training and validation accuracy
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

#Grafik training and validation loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#Augmentasi
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal",
                                                    input_shape = (size, 
                                                                  size, 
                                                                  3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
    ]
)

#Membuat object sequential
model = models.Sequential()

#Membuat data augmentation model
model.add(data_augmentation)

#Operasi konvolusi & Pooling
model.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=(size, size, 3)))
model.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
model.add(layers.MaxPooling2D())

#Menambah dropout
model.add(layers.Dropout(0.25))

#Flattening
model.add(layers.Flatten())

#Dense
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes))

#Kompilasi model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

#Menampilkan model yang sudah dikompilasi
model.summary()

#Menentukan jumlah epoch
epochs = 15
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)

#Proses menampilkan grafik
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

#Grafik training and validation accuracy
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

#Grafik training and validation loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#Proses penyimpanan model
if not os.path.exists(data_model):
  os.makedirs(data_model)

model.save(data_model, overwrite=True)
