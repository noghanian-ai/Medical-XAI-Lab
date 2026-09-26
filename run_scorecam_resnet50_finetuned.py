import os
import cv2
import numpy as np
import tensorflow as tf


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/resnet50_3class_finetuned.keras"

IMAGE_PATH = "images/Lung_Opacity/Lung_Opacity-1000.png"

OUTPUT_DIR = "outputs/ScoreCAM_ResNet50_FineTuned"

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "COVID",
    "Lung_Opacity",
    "Normal"
]

TARGET_LAYER_NAME = "conv5_block3_3_conv"

# Small batch size to prevent CPU RAM exhaustion
SCORECAM_BATCH_SIZE = 8


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

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

resized = cv2.resize(
    image_rgb,
    IMG_SIZE
)

img_array = np.expand_dims(
    resized.astype(np.float32),
    axis=0
)


# ============================================================
# ResNet50 preprocessing
# ============================================================

img_preprocessed = (
    tf.keras.applications.resnet50.preprocess_input(
        img_array
    )
)


# ============================================================
# Feature model
# ============================================================

feature_model = tf.keras.models.Model(
    inputs=backbone.input,
    outputs=target_layer.output
)


# ============================================================
# Get feature maps
# ============================================================

feature_maps = feature_model(
    img_preprocessed,
    training=False
)

feature_maps = feature_maps[0].numpy()

print(
    f"Feature maps shape: {feature_maps.shape}"
)


# ============================================================
# Normalize activation maps
# ============================================================

activation_maps = []

for i in range(feature_maps.shape[-1]):

    activation = feature_maps[:, :, i]

    activation = np.maximum(
        activation,
        0
    )

    max_value = np.max(activation)

    if max_value > 0:
        activation = activation / max_value

    activation_maps.append(
        activation
    )

activation_maps = np.array(
    activation_maps,
    dtype=np.float32
)

print(
    f"Activation maps prepared: "
    f"{activation_maps.shape[0]}"
)


# ============================================================
# Resize activation maps
# ============================================================

upsampled_maps = []

for activation in activation_maps:

    resized_activation = cv2.resize(
        activation,
        IMG_SIZE,
        interpolation=cv2.INTER_LINEAR
    )

    upsampled_maps.append(
        resized_activation
    )

upsampled_maps = np.array(
    upsampled_maps,
    dtype=np.float32
)


# ============================================================
# Select valid activation maps
# ============================================================

valid_indices = []

for i, mask in enumerate(upsampled_maps):

    if np.max(mask) > 0:
        valid_indices.append(i)


print(
    f"Valid activation maps: "
    f"{len(valid_indices)}"
)


# ============================================================
# Original prediction
# ============================================================

print("\nCalculating original prediction...")

original_predictions = model(
    img_preprocessed,
    training=False
).numpy()[0]

predicted_index = int(
    np.argmax(original_predictions)
)

predicted_label = CLASS_NAMES[
    predicted_index
]

confidence = float(
    original_predictions[predicted_index]
)

print("\nOriginal prediction:")
print(
    f"Class: {predicted_label}"
)

print(
    f"Confidence: {confidence * 100:.2f}%"
)


# ============================================================
# Build classifier head
# ============================================================

classifier_model = tf.keras.Sequential(
    [
        model.layers[-2],
        model.layers[-1]
    ],
    name="scorecam_classifier"
)


# ============================================================
# Score-CAM batch processing
# ============================================================

print("\nRunning masked-image predictions...")
print(
    f"Batch size: {SCORECAM_BATCH_SIZE}"
)

class_scores = []

total_maps = len(valid_indices)

for start in range(
    0,
    total_maps,
    SCORECAM_BATCH_SIZE
):

    end = min(
        start + SCORECAM_BATCH_SIZE,
        total_maps
    )

    batch_indices = valid_indices[
        start:end
    ]

    batch_masks = upsampled_maps[
        batch_indices
    ]

    # Create masked versions of the same image
    batch_images = (
        img_preprocessed[0][np.newaxis, ...]
        * batch_masks[:, :, :, np.newaxis]
    )

    batch_images = batch_images.astype(
        np.float32
    )

    # Forward pass through ResNet50
    batch_backbone_output = backbone(
        batch_images,
        training=False
    )

    # Classifier head
    batch_predictions = classifier_model(
        batch_backbone_output,
        training=False
    ).numpy()

    batch_scores = batch_predictions[
        :,
        predicted_index
    ]

    class_scores.extend(
        batch_scores.tolist()
    )

    print(
        f"Processed "
        f"{end}/{total_maps} activation maps"
    )

    # Release temporary tensors
    del batch_images
    del batch_backbone_output
    del batch_predictions


class_scores = np.array(
    class_scores,
    dtype=np.float32
)


# ============================================================
# Calculate Score-CAM weights
# ============================================================

weights = np.maximum(
    class_scores,
    0
)

weight_sum = np.sum(weights)

if weight_sum > 0:

    weights = (
        weights /
        weight_sum
    )

else:

    weights = (
        np.ones_like(weights) /
        len(weights)
    )


# ============================================================
# Combine activation maps
# ============================================================

scorecam = np.zeros(
    IMG_SIZE,
    dtype=np.float32
)

for weight, map_index in zip(
    weights,
    valid_indices
):

    scorecam += (
        weight *
        upsampled_maps[map_index]
    )


# ============================================================
# Normalize Score-CAM
# ============================================================

scorecam = np.maximum(
    scorecam,
    0
)

max_scorecam = np.max(
    scorecam
)

if max_scorecam > 0:

    scorecam /= max_scorecam


# ============================================================
# Create heatmap
# ============================================================

scorecam_uint8 = np.uint8(
    255 * scorecam
)

scorecam_resized = cv2.resize(
    scorecam_uint8,
    (image.shape[1], image.shape[0]),
    interpolation=cv2.INTER_LINEAR
)

heatmap_color = cv2.applyColorMap(
    scorecam_resized,
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


# ============================================================
# Save prediction information
# ============================================================

with open(
    prediction_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        f"Model: ResNet50 Fine-Tuned\n"
        f"Image: {IMAGE_PATH}\n"
        f"Target Layer: {TARGET_LAYER_NAME}\n"
        f"Explainability Method: Score-CAM\n"
        f"Batch Size: {SCORECAM_BATCH_SIZE}\n\n"
        f"Predicted Class: {predicted_label}\n"
        f"Confidence: {confidence * 100:.2f}%\n\n"
        f"Class Probabilities:\n"
    )

    for class_name, probability in zip(
        CLASS_NAMES,
        original_predictions
    ):

        f.write(
            f"{class_name}: "
            f"{probability * 100:.2f}%\n"
        )


# ============================================================
# Finished
# ============================================================

print("\nScore-CAM completed successfully.")

print("\nSaved files:")

print(original_path)
print(heatmap_path)
print(overlay_path)
print(prediction_path)