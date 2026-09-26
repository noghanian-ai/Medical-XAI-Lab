import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = (
    "models/resnet50_3class_finetuned.keras"
)

SELECTION_FILE = (
    "outputs/XAI_Evaluation_Dataset/"
    "selected_images.json"
)

DATASET_DIR = (
    r"C:\Users\manot\OneDrive\Desktop\Medical_Image_Classification"
    r"\Datasets\3-class-3000-covid-normal-Lung_Opacity-Separated"
)

OUTPUT_DIR = (
    "outputs/XAI_Evaluation_Dataset/"
    "Raw_XAI_Maps"
)

IMG_SIZE = (224, 224)

TARGET_LAYER_NAME = (
    "conv5_block3_3_conv"
)

SCORECAM_BATCH_SIZE = 8


def load_selected_images():

    with open(
        SELECTION_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    return data["images"]


def load_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        IMG_SIZE
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    return image_array


def preprocess_image(image_array):

    image_batch = np.expand_dims(
        image_array,
        axis=0
    )

    image_batch = (
        tf.keras.applications.resnet50
        .preprocess_input(image_batch)
    )

    return image_batch


def find_backbone(model):

    for layer in model.layers:

        if (
            "resnet50" in layer.name.lower()
            and hasattr(layer, "layers")
        ):
            return layer

    raise ValueError(
        "ResNet50 backbone not found."
    )


def get_classifier_layers(model, backbone):

    backbone_index = (
        model.layers.index(backbone)
    )

    classifier_layers = (
        model.layers[backbone_index + 1:]
    )

    if not classifier_layers:

        raise ValueError(
            "Classifier head not found."
        )

    return classifier_layers


def apply_classifier(
    features,
    classifier_layers
):

    x = features

    for layer in classifier_layers:
        x = layer(x)

    return x


def generate_gradcam(
    backbone,
    classifier_layers,
    image_batch,
    target_layer
):

    grad_model = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=[
            target_layer.output,
            backbone.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, features = (
            grad_model(image_batch)
        )

        predictions = apply_classifier(
            features,
            classifier_layers
        )

        predicted_class = tf.argmax(
            predictions[0]
        )

        loss = predictions[
            0,
            predicted_class
        ]

    grads = tape.gradient(
        loss,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = (
        conv_outputs
        @ pooled_grads[..., tf.newaxis]
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

    return (
        heatmap.numpy(),
        int(predicted_class.numpy()),
        predictions.numpy()[0]
    )


def generate_scorecam(
    backbone,
    classifier_layers,
    image,
    image_batch,
    target_layer,
    predicted_class
):

    feature_model = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=target_layer.output
    )

    feature_maps = (
        feature_model(
            image_batch,
            training=False
        )[0]
        .numpy()
    )

    height = image_batch.shape[1]
    width = image_batch.shape[2]

    feature_map_count = (
        feature_maps.shape[-1]
    )

    upsampled_maps = []

    for index in range(
        feature_map_count
    ):

        activation = feature_maps[
            :, :, index
        ]

        activation = np.maximum(
            activation,
            0
        )

        max_value = np.max(
            activation
        )

        if max_value > 0:

            activation = (
                activation / max_value
            )

        activation_tensor = tf.convert_to_tensor(
            activation[..., np.newaxis],
            dtype=tf.float32
        )

        activation_tensor = tf.image.resize(
            activation_tensor,
            (height, width)
        )

        upsampled_maps.append(
            activation_tensor.numpy()[
                :, :, 0
            ]
        )

    upsampled_maps = np.stack(
        upsampled_maps,
        axis=0
    )

    valid_maps = []

    for index in range(
        feature_map_count
    ):

        activation = upsampled_maps[
            index
        ]

        if np.max(activation) > 0:
            valid_maps.append(index)

    scores = []

    for start in range(
        0,
        len(valid_maps),
        SCORECAM_BATCH_SIZE
    ):

        batch_indices = valid_maps[
            start:
            start + SCORECAM_BATCH_SIZE
        ]

        masked_images = []

        for index in batch_indices:

            mask = upsampled_maps[
                index
            ]

            masked = (
                image
                * mask[..., np.newaxis]
            )

            masked_images.append(
                masked
            )

        masked_images = np.stack(
            masked_images,
            axis=0
        )

        masked_images = (
            tf.keras.applications.resnet50
            .preprocess_input(
                masked_images.copy()
            )
        )

        features = backbone(
            masked_images,
            training=False
        )

        predictions = apply_classifier(
            features,
            classifier_layers
        )

        batch_scores = (
            predictions.numpy()[
                :,
                predicted_class
            ]
        )

        scores.extend(
            batch_scores.tolist()
        )

        print(
            f"      Score-CAM: "
            f"{min(start + len(batch_indices), len(valid_maps))}"
            f"/{len(valid_maps)}"
        )

    scores = np.array(
        scores,
        dtype=np.float32
    )

    scorecam = np.zeros(
        (height, width),
        dtype=np.float32
    )

    for score, index in zip(
        scores,
        valid_maps
    ):

        scorecam += (
            score
            * upsampled_maps[index]
        )

    scorecam = np.maximum(
        scorecam,
        0
    )

    max_value = np.max(
        scorecam
    )

    if max_value > 0:

        scorecam /= max_value

    return scorecam


def main():

    print("=" * 70)
    print(
        "Raw XAI Maps — Evaluation Dataset"
    )
    print("=" * 70)

    print()
    print("Loading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    backbone = find_backbone(
        model
    )

    classifier_layers = (
        get_classifier_layers(
            model,
            backbone
        )
    )

    target_layer = backbone.get_layer(
        TARGET_LAYER_NAME
    )

    print(
        f"Backbone: {backbone.name}"
    )

    print(
        f"Target layer: "
        f"{target_layer.name}"
    )

    print(
        f"Target output shape: "
        f"{target_layer.output.shape}"
    )

    selected_images = (
        load_selected_images()
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    metadata = []

    total = len(
        selected_images
    )

    for number, item in enumerate(
        selected_images,
        start=1
    ):

        class_name = item[
            "class_name"
        ]

        filename = item[
            "filename"
        ]

        relative_path = item[
            "relative_path"
        ]

        image_path = os.path.join(
            DATASET_DIR,
            "validation",
            relative_path
        )

        print()
        print(
            "=" * 70
        )

        print(
            f"[{number}/{total}] "
            f"{class_name}/{filename}"
        )

        image = load_image(
            image_path
        )

        image_batch = preprocess_image(
            image
        )

        gradcam, predicted_class, predictions = (
            generate_gradcam(
                backbone,
                classifier_layers,
                image_batch,
                target_layer
            )
        )

        class_names = [
            "COVID",
            "Lung_Opacity",
            "Normal"
        ]

        predicted_name = (
            class_names[
                predicted_class
            ]
        )

        confidence = float(
            predictions[
                predicted_class
            ]
        )

        print(
            f"Predicted: "
            f"{predicted_name}"
        )

        print(
            f"Confidence: "
            f"{confidence * 100:.2f}%"
        )

        scorecam = generate_scorecam(
            backbone,
            classifier_layers,
            image,
            image_batch,
            target_layer,
            predicted_class
        )

        image_id = os.path.splitext(
            filename
        )[0]

        image_output_dir = os.path.join(
            OUTPUT_DIR,
            image_id
        )

        os.makedirs(
            image_output_dir,
            exist_ok=True
        )

        np.save(
            os.path.join(
                image_output_dir,
                "gradcam_raw.npy"
            ),
            gradcam
        )

        np.save(
            os.path.join(
                image_output_dir,
                "scorecam_raw.npy"
            ),
            scorecam
        )

        metadata.append(
            {
                "class_name": class_name,
                "filename": filename,
                "relative_path": relative_path,
                "predicted_class": predicted_name,
                "predicted_class_index": predicted_class,
                "confidence": confidence
            }
        )

        print(
            "Raw maps saved."
        )

    metadata_file = os.path.join(
        OUTPUT_DIR,
        "metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "total_images": total,
                "target_layer": TARGET_LAYER_NAME,
                "metadata": metadata
            },
            f,
            indent=4,
            ensure_ascii=False
        )

    print()
    print("=" * 70)
    print(
        "Raw XAI generation completed."
    )
    print(
        f"Images processed: {total}"
    )
    print()
    print(
        f"Saved to: {OUTPUT_DIR}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()