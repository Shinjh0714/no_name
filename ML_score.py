import os
from keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import tensorflow as tf

# Image Data Settings
img_width, img_height = 150, 150
validation_data_dir = 'data/validation'
batch_size = 16

# Function to import validation data
def get_validation_data(generator, steps, img_width, img_height):
    batchX, batchY = [], []
    for _ in range(steps):
        x, y = generator.next()
        resized_x = np.array([image.smart_resize(img, (img_width, img_height)) for img in x])
        batchX.extend(resized_x)
        batchY.extend(y)
    return np.array(batchX), np.array(batchY)

# data preprocessing
test_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)

# Preparing validation data
nb_validation_samples = 800
val_steps = nb_validation_samples // batch_size
val_data, val_labels = get_validation_data(validation_generator, val_steps, img_width, img_height)

# Model File List
model_files = [
    'final_model_layer0_trial1.h5',
    'final_model_layer1_trial1.h5',
    'final_model_layer2_trial1.h5',
    'final_model_layer3_trial1.h5',
    'final_model_layer4_trial1.h5',
    'final_model_layer5_trial1.h5',
    'final_model_layer6_trial1.h5'
]

# Check and evaluate if the model exists in the current directory
current_dir = os.getcwd()
for model_file in model_files:
    model_path = os.path.join(current_dir, model_file)
    if not os.path.exists(model_path):
        print(f"Model file {model_file} does not exist at path {model_path}. Skipping this model.")
        continue
    
    model = load_model(model_path)
    
    # Verifying and Setting the Model Input Size
    if hasattr(model, 'input_shape'):
        if model.input_shape[1:3] != (img_width, img_height):
            print(f"Model {model_file} was constructed with input shape {model.input_shape[1:3]}. Resizing inputs.")
            val_data_resized = np.array([image.smart_resize(img, (model.input_shape[1], model.input_shape[2])) for img in val_data])
        else:
            val_data_resized = val_data
    else:
        val_data_resized = val_data
    
    # Performing predictions
    val_predict = np.asarray(model.predict(val_data_resized))
    val_predict = np.round(val_predict)
    
    # calculation of scores
    _val_precision = precision_score(val_labels, val_predict)
    _val_recall = recall_score(val_labels, val_predict)
    _val_f1 = f1_score(val_labels, val_predict)
    
    print(f"Model: {model_file} — val_precision: {_val_precision:.4f} — val_recall: {_val_recall:.4f} — val_f1: {_val_f1:.4f}")
