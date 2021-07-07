import os
import pathlib

from tensorflow import keras

BASE_DIR = BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_model = BASE_DIR + "/model"

model = keras.models.load_model(data_model)

model.summary()
print(model.trainable_variables) 
