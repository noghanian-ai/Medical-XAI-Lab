import os
import cv2
import numpy as np
import tensorflow as tf

# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "resnet50_3class_finetuned.keras"
)

IMAGE_PATH = os.path.join(
    BASE_DIR,
    "images",
    "Lung_Opacity",
    "Lung_Opacity-1000.png"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "Raw_XAI_Maps_ResNet50_FineTuned"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

CLASS_NAMES = [
    "COVID",
    "Lung_Opacity",
    "Normal"
]

TARGET_CLASS = "Lung_Opacity"
TARGET_CLASS_INDEX = CLASS_NAMES.index(
    TARGET_CLASS
)

TARGET_LAYER_NAME = "conv5_block3_3_conv"

SCORECAM_BATCH_SIZE = 8


# ============================================================
# Image loading
# ============================================================

def load_image(image_path):
    """
    Load image as RGB float32 with shape (224, 224, 3).
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image:\n{image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    return image.astype(np.float32)


# ============================================================
# Model structure
# ============================================================

def find_resnet50_backbone(model):
    """
    Find the nested ResNet50 backbone.
    """

    for layer in model.layers:
        if "resnet50" in layer.name.lower():
            return layer

    raise ValueError(
        "Could not find the ResNet50 backbone."
    )


def apply_classifier_head(model, backbone_output):
    """
    Apply the outer classifier layers after the ResNet50
    backbone.

    This follows the structure used by the existing
    ResNet50 fine-tuned Grad-CAM runner.
    """

    x = backbone_output

    backbone_found = False

    for layer in model.layers:

        if layer.name == "resnet50":
            backbone_found = True
            continue

        if not backbone_found:
            continue

        x = layer(x)

    return x


# ============================================================
# Prediction
# ============================================================

def predict_with_full_model(
        model,
        image
):
    """
    Predict using the complete saved model.

    The model was saved without embedded ResNet50
    preprocessing, so preprocessing is applied here.
    """

    batch = np.expand_dims(
        image,
        axis=0
    )

    batch = tf.keras.applications.resnet50.preprocess_input(
        batch
    )

    prediction = model.predict(
        batch,
        verbose=0
    )

    return prediction[0]


# ============================================================
# Raw Grad-CAM
# ============================================================

def generate_raw_gradcam(
        model,
        backbone,
        image,
        target_layer,
        predicted_class
):
    """
    Generate the raw normalized Grad-CAM map.

    Returns:
        raw_heatmap: float32 array in [0, 1]
    """

    input_tensor = tf.convert_to_tensor(
        np.expand_dims(
            tf.keras.applications.resnet50.preprocess_input(
                image.copy()
            ),
            axis=0
        ),
        dtype=tf.float32
    )

    grad_model = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=[
            target_layer.output,
            backbone.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, backbone_output = grad_model(
            input_tensor
        )

        predictions = apply_classifier_head(
            model,
            backbone_output
        )

        loss = predictions[:, predicted_class]

    gradients = tape.gradient(
        loss,
        conv_outputs
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = (
        conv_outputs
        @ pooled_gradients[..., tf.newaxis]
    )

    heatmap = tf.squeeze(
        heatmap
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap
    )

    return heatmap.numpy().astype(
        np.float32
    )


# ============================================================
# Raw Score-CAM
# ============================================================

def generate_raw_scorecam(
        model,
        backbone,
        image,
        target_layer,
        predicted_class
):
    """
    Generate the raw Score-CAM importance map.

    Activation maps are processed in small batches to avoid
    excessive RAM consumption on the CPU-only machine.
    """

    print()
    print("=" * 60)
    print("Generating raw Score-CAM")
    print("=" * 60)

    # --------------------------------------------------------
    # Extract feature maps
    # --------------------------------------------------------

    feature_model = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=target_layer.output
    )

    input_tensor = tf.convert_to_tensor(
        np.expand_dims(
            tf.keras.applications.resnet50.preprocess_input(
                image.copy()
            ),
            axis=0
        ),
        dtype=tf.float32
    )

    feature_maps = feature_model(
        input_tensor,
        training=False
    )

    feature_maps = feature_maps[0].numpy()

    print(
        "Feature maps shape:",
        feature_maps.shape
    )

    height = image.shape[0]
    width = image.shape[1]

    # --------------------------------------------------------
    # Normalize each activation map
    # --------------------------------------------------------

    normalized_maps = np.zeros_like(
        feature_maps,
        dtype=np.float32
    )

    valid_indices = []

    for channel in range(
        feature_maps.shape[-1]
    ):

        activation = feature_maps[:, :, channel]

        min_value = np.min(
            activation
        )

        max_value = np.max(
            activation
        )

        if max_value > min_value:

            normalized = (
                (activation - min_value)
                / (max_value - min_value)
            )

            normalized_maps[:, :, channel] = (
                normalized.astype(np.float32)
            )

            valid_indices.append(
                channel
            )

    print(
        "Valid activation maps:",
        len(valid_indices)
    )

    # --------------------------------------------------------
    # Process activation maps in batches
    # --------------------------------------------------------

    score_weights = np.zeros(
        len(valid_indices),
        dtype=np.float32
    )

    for start in range(
        0,
        len(valid_indices),
        SCORECAM_BATCH_SIZE
    ):

        end = min(
            start + SCORECAM_BATCH_SIZE,
            len(valid_indices)
        )

        batch_indices = valid_indices[
            start:end
        ]

        masked_images = []

        for channel_index in batch_indices:

            mask = normalized_maps[
                :, :,
                channel_index
            ]

            mask = cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_LINEAR
            )

            masked_image = (
                image * mask[:, :, np.newaxis]
            )

            masked_images.append(
                masked_image
            )

        masked_images = np.stack(
            masked_images,
            axis=0
        ).astype(np.float32)

        preprocessed = (
            tf.keras.applications.resnet50.preprocess_input(
                masked_images
            )
        )

        predictions = model.predict(
            preprocessed,
            verbose=0
        )

        scores = predictions[
            :,
            predicted_class
        ]

        score_weights[
            start:end
        ] = scores.astype(
            np.float32
        )

        print(
            f"Processed {end}/{len(valid_indices)} "
            "activation maps"
        )

    # --------------------------------------------------------
    # Weighted combination
    # --------------------------------------------------------

    raw_scorecam = np.zeros(
        feature_maps.shape[:2],
        dtype=np.float32
    )

    for i, channel_index in enumerate(
        valid_indices
    ):

        raw_scorecam += (
            score_weights[i]
            * normalized_maps[
                :, :,
                channel_index
            ]
        )

    raw_scorecam = np.maximum(
        raw_scorecam,
        0
    )

    max_value = np.max(
        raw_scorecam
    )

    if max_value > 0:
        raw_scorecam /= max_value

    return raw_scorecam.astype(
        np.float32
    )


# ============================================================
# Save visualization
# ============================================================

def save_visualization(
        heatmap,
        filename
):
    """
    Save a visualization of a raw XAI map.
    """

    heatmap_resized = cv2.resize(
        heatmap,
        (224, 224),
        interpolation=cv2.INTER_LINEAR
    )

    heatmap_uint8 = np.uint8(
        np.clip(
            heatmap_resized,
            0,
            1
        ) * 255
    )

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    cv2.imwrite(
        output_path,
        heatmap_color
    )

    return output_path


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Raw XAI Map Generation")
    print("ResNet50 Fine-Tuned")
    print("=" * 60)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("Loading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

    # --------------------------------------------------------
    # Find backbone
    # --------------------------------------------------------

    backbone = find_resnet50_backbone(
        model
    )

    print(
        "Backbone:",
        backbone.name
    )

    # --------------------------------------------------------
    # Find target layer
    # --------------------------------------------------------

    target_layer = backbone.get_layer(
        TARGET_LAYER_NAME
    )

    print(
        "Target layer:",
        target_layer.name
    )

    print(
        "Target layer output shape:",
        target_layer.output.shape
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    print()
    print("Loading image...")

    image = load_image(
        IMAGE_PATH
    )

    print(
        "Image shape:",
        image.shape
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print()
    print("Running prediction...")

    prediction = predict_with_full_model(
        model,
        image
    )

    predicted_class = int(
        np.argmax(prediction)
    )

    print(
        "Predicted class:",
        CLASS_NAMES[predicted_class]
    )

    print(
        "Confidence:",
        f"{prediction[predicted_class] * 100:.2f}%"
    )

    print(
        "Target class:",
        TARGET_CLASS
    )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    print()
    print("Generating raw Grad-CAM...")

    gradcam = generate_raw_gradcam(
        model,
        backbone,
        image,
        target_layer,
        TARGET_CLASS_INDEX
    )

    gradcam_npy_path = os.path.join(
        OUTPUT_DIR,
        "gradcam_raw.npy"
    )

    np.save(
        gradcam_npy_path,
        gradcam
    )

    gradcam_png_path = save_visualization(
        gradcam,
        "gradcam_raw_visualization.png"
    )

    print(
        "Raw Grad-CAM shape:",
        gradcam.shape
    )

    print(
        "Saved:",
        gradcam_npy_path
    )

    print(
        "Saved:",
        gradcam_png_path
    )

    # --------------------------------------------------------
    # Score-CAM
    # --------------------------------------------------------

    scorecam = generate_raw_scorecam(
        model,
        backbone,
        image,
        target_layer,
        TARGET_CLASS_INDEX
    )

    scorecam_npy_path = os.path.join(
        OUTPUT_DIR,
        "scorecam_raw.npy"
    )

    np.save(
        scorecam_npy_path,
        scorecam
    )

    scorecam_png_path = save_visualization(
        scorecam,
        "scorecam_raw_visualization.png"
    )

    print()
    print(
        "Raw Score-CAM shape:",
        scorecam.shape
    )

    print(
        "Saved:",
        scorecam_npy_path
    )

    print(
        "Saved:",
        scorecam_png_path
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Raw XAI map generation completed successfully.")
    print("=" * 60)

    print()
    print("Output directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()