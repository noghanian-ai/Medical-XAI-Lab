import os
import cv2
import numpy as np
import tensorflow as tf


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/resnet50_3class_finetuned.keras"

IMAGE_PATH = "images/Lung_Opacity/Lung_Opacity-1000.png"

OUTPUT_DIR = "outputs/GradCAM_ResNet50_FineTuned"

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "COVID",
    "Lung_Opacity",
    "Normal"
]

TARGET_LAYER_NAME = "conv5_block3_3_conv"


# ============================================================
# Load model
# ============================================================

print("\nLoading fine-tuned ResNet50 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# Locate ResNet50 backbone
# ============================================================

backbone = None

for layer in model.layers:
    if isinstance(layer, tf.keras.Model) and "resnet50" in layer.name.lower():
        backbone = layer
        break

if backbone is None:
    raise ValueError("ResNet50 backbone could not be found.")

print(f"Backbone: {backbone.name}")


# ============================================================
# Locate target convolutional layer
# ============================================================

target_layer = backbone.get_layer(TARGET_LAYER_NAME)

print(f"Target layer: {target_layer.name}")
print(f"Target layer output shape: {target_layer.output.shape}")


# ============================================================
# Load image
# ============================================================

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise ValueError("Could not read the image.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

resized = cv2.resize(image_rgb, IMG_SIZE)

img_array = np.expand_dims(
    resized.astype(np.float32),
    axis=0
)


# ============================================================
# ResNet50 preprocessing
# ============================================================

img_preprocessed = tf.keras.applications.resnet50.preprocess_input(
    img_array
)


# ============================================================
# Build Grad-CAM model
# ============================================================

# The ResNet50 backbone is nested inside the fine-tuned model.
# Therefore we create a model that exposes:
#   1. the target convolutional feature maps
#   2. the final ResNet50 feature output

backbone_grad_model = tf.keras.models.Model(
    inputs=backbone.input,
    outputs=[
        target_layer.output,
        backbone.output
    ]
)


# ============================================================
# Forward pass + gradients
# ============================================================

with tf.GradientTape() as tape:

    conv_outputs, backbone_output = backbone_grad_model(
        img_preprocessed,
        training=False
    )

    # Recreate the classifier head of the original model.
    x = backbone_output

    for layer in model.layers:
        if layer.name == backbone.name:
            continue

        x = layer(x)

    predictions = x

    predicted_class = tf.argmax(
        predictions[0]
    )

    class_score = predictions[:, predicted_class]


# ============================================================
# Calculate gradients
# ============================================================

grads = tape.gradient(
    class_score,
    conv_outputs
)

if grads is None:
    raise RuntimeError(
        "Gradients could not be calculated."
    )


# ============================================================
# Grad-CAM calculation
# ============================================================

pooled_grads = tf.reduce_mean(
    grads,
    axis=(0, 1, 2)
)

conv_outputs = conv_outputs[0]

heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

heatmap = tf.squeeze(heatmap)

heatmap = tf.maximum(
    heatmap,
    0
)

max_value = tf.reduce_max(heatmap)

if max_value > 0:
    heatmap /= max_value

heatmap = heatmap.numpy()


# ============================================================
# Prediction information
# ============================================================

probabilities = predictions[0].numpy()

predicted_index = int(predicted_class.numpy())

predicted_label = CLASS_NAMES[predicted_index]

confidence = float(
    probabilities[predicted_index]
)

print("\nPrediction:")
print(f"Class: {predicted_label}")
print(f"Confidence: {confidence * 100:.2f}%")


# ============================================================
# Prepare Grad-CAM heatmap
# ============================================================

heatmap_uint8 = np.uint8(
    255 * heatmap
)

heatmap_resized = cv2.resize(
    heatmap_uint8,
    (image.shape[1], image.shape[0])
)

heatmap_color = cv2.applyColorMap(
    heatmap_resized,
    cv2.COLORMAP_JET
)


# ============================================================
# Create overlay
# ============================================================

overlay = cv2.addWeighted(
    image,
    0.55,
    heatmap_color,
    0.45,
    0
)


# ============================================================
# Create output directory
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# Save results
# ============================================================

original_path = os.path.join(
    OUTPUT_DIR,
    "original.png"
)

heatmap_path = os.path.join(
    OUTPUT_DIR,
    "heatmap.png"
)

overlay_path = os.path.join(
    OUTPUT_DIR,
    "overlay.png"
)

prediction_path = os.path.join(
    OUTPUT_DIR,
    "prediction.txt"
)


cv2.imwrite(
    original_path,
    image
)

cv2.imwrite(
    heatmap_path,
    heatmap_color
)

cv2.imwrite(
    overlay_path,
    overlay
)


with open(
    prediction_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        f"Model: ResNet50 Fine-Tuned\n"
        f"Image: {IMAGE_PATH}\n"
        f"Target Layer: {TARGET_LAYER_NAME}\n\n"
        f"Predicted Class: {predicted_label}\n"
        f"Confidence: {confidence * 100:.2f}%\n\n"
        f"Class Probabilities:\n"
    )

    for class_name, probability in zip(
        CLASS_NAMES,
        probabilities
    ):
        f.write(
            f"{class_name}: "
            f"{probability * 100:.2f}%\n"
        )


# ============================================================
# Finished
# ============================================================

print("\nGrad-CAM completed successfully.")

print("\nSaved files:")

print(original_path)
print(heatmap_path)
print(overlay_path)
print(prediction_path)