#!/home/avian-flu/miniconda3/bin/python3
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras import backend as K
import keras
import numpy as np
import sys
import re
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.callbacks import Callback

# GPU memory setting
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9600)])
    except RuntimeError as e:
        print(e)

# Handling System Arguments
if len(sys.argv) > 1:
    nlay = sys.argv[1]
else:
    nlay = 'default_layer_value'  # Default setting

if len(sys.argv) > 2:
    trial = sys.argv[2]
else:
    trial = 'default_trial_value'  # Default setting

# Set the model storage path
MODEL_SAVE_FOLDER_PATH = './model1/'
if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
    os.mkdir(MODEL_SAVE_FOLDER_PATH)

# image data setting
img_width, img_height = 150, 150
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16

# Keras back-end setting
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Create a Model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Define Custom Callback Classes
class Metrics(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_data, val_labels = self.validation_data
        val_predict = np.asarray(self.model.predict(val_data))
        val_predict = np.round(val_predict)
        val_targ = val_labels
        _val_precision = precision_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_f1 = f1_score(val_targ, val_predict)
        logs['val_precision'] = _val_precision
        logs['val_recall'] = _val_recall
        logs['val_f1'] = _val_f1
        print(f" — val_precision: {_val_precision} — val_recall: {_val_recall} — val_f1: {_val_f1}")
        return

# Function to import validation data
def get_validation_data(generator, steps):
    batchX, batchY = [], []
    for i in range(steps):
        x, y = generator.next()
        batchX.extend(x)
        batchY.extend(y)
    return np.array(batchX), np.array(batchY)

# Preparing validation data
val_steps = nb_validation_samples // batch_size
val_data, val_labels = get_validation_data(validation_generator, val_steps)
validation_data = (val_data, val_labels)

# Create a Callback Instance
metrics = Metrics(validation_data)

# Set up a callback and learn the model
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.01, verbose=0, patience=10, restore_best_weights=True)

from keras.callbacks import ModelCheckpoint
model_path = MODEL_SAVE_FOLDER_PATH + '_best_for_model' + nlay + '.hdf5'
cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', mode='min', verbose=0, save_best_only=True, save_weights_only=False)

tb_hist = keras.callbacks.TensorBoard(log_dir='./log/layer{}_trial{}'.format(nlay, trial), histogram_freq=0, write_graph=True, write_images=True)

hist = model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[tb_hist, early_stopping, cb_checkpoint, metrics])  # Added metrics callback

model.save('final_model_layer{}_trial{}.h5'.format(nlay, trial))

# Visualize learning outcomes
fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['accuracy'], 'b', label='train accuracy')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val accuracy')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.savefig('./fig/learning_curve_layer{}_trial{}'.format(nlay, trial), dpi=300)
