import os
import pathlib
import numpy as np
import tensorflow as tf

from tensorflow import keras

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

#Memecah data untuk digunakan sebagai data latih sebanyak 80%
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(size, size),
    batch_size=batch_size
)

#Memecah data untuk digunakan sebagai validasi data sebanyak 20%
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

#Pengambilan jumlah klasifikasi dari banyaknya nama klasifikasi
num_classes = len(class_names)

#Proses load data model
model = keras.models.load_model(data_model)

#Mengambil data gambar yang akan ditest klasifikasinya
img_pred_dir = BASE_DIR + '/prediction/poksai_kuda2.jpg'

img_pred = keras.preprocessing.image.load_img(
    img_pred_dir, target_size=(size, size)
)

#Merubah data gambar ke array
img_pred_array = keras.preprocessing.image.img_to_array(img_pred)
img_pred_array = tf.expand_dims(img_pred_array, 0)

#Proses klasifikasi
prediction = model.predict(img_pred_array)
score = tf.nn.softmax(prediction[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
