# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 12:27:12 2026

@author: manot
"""

# -*- coding: utf-8 -*-

"""
Run Score-CAM
Medical-XAI-Lab
"""


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


from config import (
    MODEL_PATH,
    IMAGE_DIR,
    OUTPUT_DIR_SCORECAM,
    IMG_SIZE,
    CLASS_NAMES,
    LAST_CONV_LAYER
)


from scorecam import generate_scorecam


from utils.file_manager import create_output_folder


from utils.save_results import (
    save_original,
    save_heatmap,
    save_overlay,
    save_prediction
)



# ===============================
# Load Model
# ===============================

print("Loading model...")

model = load_model(
    MODEL_PATH
)

print("Model loaded successfully")



# ===============================
# Select Image
# ===============================

image_path = os.path.join(
    IMAGE_DIR,
#    "COVID",
#    "COVID-1000.png"
    
#    "Lung_Opacity",
 #   "Lung_Opacity-1000.png"
    
    "Normal",
    "Normal-1000.png"
)


print("Image:")
print(image_path)



# ===============================
# Preprocess Image
# ===============================

img = load_img(
    image_path,
    target_size=IMG_SIZE
)


img_array = img_to_array(
    img
)


img_array = np.expand_dims(
    img_array,
    axis=0
)



# ===============================
# Prediction
# ===============================

prediction = model.predict(
    img_array
)


predicted_class = np.argmax(
    prediction
)


confidence = np.max(
    prediction
) * 100



print(
    "Prediction:",
    CLASS_NAMES[predicted_class]
)


print(
    "Confidence:",
    confidence
)



# ===============================
# Create Output Folder
# ===============================

image_name = os.path.basename(
    image_path
)


output_folder = create_output_folder(
    OUTPUT_DIR_SCORECAM,
    CLASS_NAMES[predicted_class],
    image_name
)


# ===============================
# Generate Score-CAM
# ===============================


heatmap = generate_scorecam(
    model,
    img_array,
    LAST_CONV_LAYER,
    predicted_class
)



# ===============================
# Create Overlay
# ===============================


original = cv2.imread(
    image_path
)


original = cv2.resize(
    original,
    IMG_SIZE
)



heatmap = cv2.resize(
    heatmap,
    IMG_SIZE
)


heatmap = np.uint8(
    255 * heatmap
)


heatmap = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)



superimposed_img = heatmap * 0.4 + original


superimposed_img = np.uint8(
    superimposed_img
)



# ===============================
# Save Results
# ===============================


save_original(
    output_folder,
    original
)


save_heatmap(
    output_folder,
    heatmap
)


save_overlay(
    output_folder,
    superimposed_img
)


save_prediction(
    output_folder,
    CLASS_NAMES[predicted_class],
    confidence
)



print("Score-CAM results saved in:")
print(output_folder)



# ===============================
# Display
# ===============================


plt.figure(figsize=(10,4))


plt.subplot(1,2,1)

plt.imshow(
    cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )
)

plt.title("Original X-ray")
plt.axis("off")



plt.subplot(1,2,2)

plt.imshow(
    cv2.cvtColor(
        superimposed_img,
        cv2.COLOR_BGR2RGB
    )
)


plt.title(
    f"Score-CAM\n{CLASS_NAMES[predicted_class]}"
)

plt.axis("off")


plt.show()