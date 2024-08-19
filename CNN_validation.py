import os
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing import image

def load_image(img_path, target_size=(400, 400)):
    img = image.load_img(img_path, target_size=target_size)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    return img_tensor

# model file list 
model_files = [
    "model/layer0_trial1_best_for_model0.hdf5",
    "model/layer1_trial1_best_for_model1.hdf5",
    "model/layer2_trial1_best_for_model2.hdf5",
    "model/layer3_trial1_best_for_model3.hdf5",
    "model/layer4_trial1_best_for_model4.hdf5",
    "model/layer5_trial1_best_for_model5.hdf5",
    "model/layer6_trial1_best_for_model6.hdf5"
]

# model load
models = [load_model(model_file) for model_file in model_files]

# image directory and list load
image_dir = "in/put/your image"  
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]

# Data frames to store results
results = pd.DataFrame(columns=["image"] + [f"model_{i}" for i in range(len(models))])

# Load and predict images
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    new_image = load_image(image_path)

    # Perform predictions for each model
    predictions = [model.predict_classes(new_image)[0][0] for model in models]
    results = results.append({"image": image_file, **{f"model_{i}": pred for i, pred in enumerate(predictions)}}, ignore_index=True)

# Save results as CSV files
results.to_csv("model_predictions.csv", index=False)
