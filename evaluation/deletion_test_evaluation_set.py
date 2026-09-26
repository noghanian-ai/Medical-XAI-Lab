"""
Multi-image Deletion Test for Grad-CAM and Score-CAM
ResNet50 Fine-Tuned XAI Evaluation

Dataset:
- 30 validation images
- 10 COVID
- 10 Lung_Opacity
- 10 Normal

For each image:
- Use the model's original predicted class as the target class.
- Mask top 0%, 10%, ..., 100% important pixels.
- Measure target-class confidence.
- Compute deletion AUC using trapezoidal integration.

Important:
- Raw XAI maps are used directly.
- Score-CAM maps must have been generated from raw images
  with preprocessing applied only after masking.
"""

import os
import json
import numpy as np
import tensorflow as tf


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/resnet50_3class_finetuned.keras"

DATASET_DIR = (
    r"C:\Users\manot\OneDrive\Desktop"
    r"\Medical_Image_Classification\Datasets"
    r"\3-class-3000-covid-normal-Lung_Opacity-Separated"
)


SELECTION_PATH = (
    "outputs/XAI_Evaluation_Dataset/"
    "selected_images.json"
)

RAW_MAP_DIR = (
    "outputs/XAI_Evaluation_Dataset/"
    "Raw_XAI_Maps"
)

OUTPUT_DIR = (
    "outputs/XAI_Evaluation_Dataset/"
    "Deletion_Test"
)

MASK_PERCENTAGES = list(range(0, 101, 10))

CLASS_NAMES = [
    "COVID",
    "Lung_Opacity",
    "Normal"
]

IMG_SIZE = (224, 224)


# ============================================================
# Utility functions
# ============================================================

def load_image(image_path):
    """
    Load image as RGB float32 array in [0, 255].
    """

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE,
        color_mode="rgb"
    )

    image = tf.keras.utils.img_to_array(
        image
    ).astype(np.float32)

    return image


def preprocess_batch(images):
    """
    Apply ResNet50 preprocessing exactly once.
    """

    return tf.keras.applications.resnet50.preprocess_input(
        images.copy()
    )


def normalize_map(xai_map):
    """
    Normalize raw XAI map to [0, 1].
    """

    xai_map = np.asarray(
        xai_map,
        dtype=np.float32
    )

    xai_map = np.maximum(
        xai_map,
        0
    )

    max_value = np.max(
        xai_map
    )

    if max_value > 0:
        xai_map = (
            xai_map / max_value
        )

    return xai_map


def rank_pixels(xai_map):
    """
    Resize a 7x7 XAI map to 224x224 and
    return flattened pixel importance.
    """

    xai_tensor = tf.convert_to_tensor(
        xai_map[..., np.newaxis],
        dtype=tf.float32
    )

    xai_tensor = tf.image.resize(
        xai_tensor,
        IMG_SIZE,
        method="bilinear"
    )

    xai_map_resized = (
        xai_tensor.numpy()[:, :, 0]
    )

    flattened = (
        xai_map_resized.reshape(-1)
    )

    ranking = np.argsort(
        flattened
    )[::-1]

    return xai_map_resized, ranking


def create_masked_batch(
    image,
    ranking
):
    """
    Create 11 masked versions of one image.

    Masking is performed on the RAW image.
    The masked batch is preprocessed only afterward.
    """

    height, width = IMG_SIZE

    total_pixels = (
        height * width
    )

    mean_rgb = np.mean(
        image,
        axis=(0, 1),
        keepdims=True
    )

    masked_images = []

    for percentage in MASK_PERCENTAGES:

        masked = image.copy()

        if percentage > 0:

            pixels_to_mask = int(
                total_pixels
                * percentage
                / 100
            )

            if pixels_to_mask > 0:

                selected_pixels = (
                    ranking[:pixels_to_mask]
                )

                flat_masked = (
                    masked.reshape(
                        -1,
                        3
                    )
                )

                flat_masked[
                    selected_pixels
                ] = mean_rgb.reshape(
                    1,
                    3
                )

                masked = flat_masked.reshape(
                    height,
                    width,
                    3
                )

        masked_images.append(
            masked
        )

    return np.stack(
        masked_images,
        axis=0
    ).astype(np.float32)


def predict_confidences(
    model,
    masked_images,
    target_class
):
    """
    Predict target-class confidence for
    all masking levels in one batch.
    """

    preprocessed = preprocess_batch(
        masked_images
    )

    predictions = model.predict(
        preprocessed,
        verbose=0
    )

    confidences = predictions[
        :,
        target_class
    ]

    return confidences.astype(
        np.float32
    )


def compute_auc(
    confidences
):
    """
    Compute deletion AUC.

    Lower AUC means confidence decreases
    more rapidly as important pixels are removed.

    This metric is reported descriptively;
    it is not interpreted as a method winner
    from this single evaluation alone.
    """

    x = np.array(
        MASK_PERCENTAGES,
        dtype=np.float32
    )

    y = np.asarray(
        confidences,
        dtype=np.float32
    )

    return float(
        np.trapz(
            y,
            x
        )
    )


# ============================================================
# Load selected image metadata
# ============================================================

print("=" * 70)
print("Multi-image Deletion Test")
print("ResNet50 Fine-Tuned")
print("=" * 70)

print()
print("Loading selected image metadata...")

with open(
    SELECTION_PATH,
    "r",
    encoding="utf-8"
) as file:

    selected_data = json.load(
        file
    )

selected_images = selected_data["images"]

# ============================================================
# Load model
# ============================================================

print()
print("Loading fine-tuned ResNet50 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")

print(
    f"Model input shape: "
    f"{model.input_shape}"
)

print(
    f"Model output shape: "
    f"{model.output_shape}"
)


# ============================================================
# Prepare output directory
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# Evaluation containers
# ============================================================

gradcam_results = []
scorecam_results = []

processed_count = 0


# ============================================================
# Main evaluation loop
# ============================================================

for item in selected_images:

    filename = item["filename"]
    class_name = item["class_name"]
    relative_path = item["relative_path"]

    image_path = os.path.join(
        DATASET_DIR,
        "validation",
        relative_path
    )

    image_id = os.path.splitext(
        filename
    )[0]

    map_dir = os.path.join(
        RAW_MAP_DIR,
        image_id
    )

    gradcam_path = os.path.join(
        map_dir,
        "gradcam_raw.npy"
    )

    scorecam_path = os.path.join(
        map_dir,
        "scorecam_raw.npy"
    )

    print()
    print("-" * 70)
    print(
        f"Image {processed_count + 1}/"
        f"{len(selected_images)}"
    )
    print(
        f"File: {filename}"
    )
    print(
        f"Ground truth: {class_name}"
    )

    # --------------------------------------------------------
    # Load raw image
    # --------------------------------------------------------

    image = load_image(
        image_path
    )

    original_batch = (
        preprocess_batch(
            image[np.newaxis, ...]
        )
    )

    original_prediction = (
        model.predict(
            original_batch,
            verbose=0
        )[0]
    )

    predicted_class_index = int(
        np.argmax(
            original_prediction
        )
    )

    predicted_class = (
        CLASS_NAMES[
            predicted_class_index
        ]
    )

    original_confidence = float(
        original_prediction[
            predicted_class_index
        ]
    )

    correct_prediction = (
        predicted_class
        == class_name
    )

    print(
        f"Prediction: "
        f"{predicted_class}"
    )

    print(
        f"Confidence: "
        f"{original_confidence * 100:.2f}%"
    )

    print(
        f"Correct: "
        f"{correct_prediction}"
    )

    # --------------------------------------------------------
    # Load raw XAI maps
    # --------------------------------------------------------

    gradcam_raw = np.load(
        gradcam_path
    )

    scorecam_raw = np.load(
        scorecam_path
    )

    gradcam_map = normalize_map(
        gradcam_raw
    )

    scorecam_map = normalize_map(
        scorecam_raw
    )

    # --------------------------------------------------------
    # Evaluate Grad-CAM
    # --------------------------------------------------------

    print()
    print("Evaluating Grad-CAM...")

    _, gradcam_ranking = (
        rank_pixels(
            gradcam_map
        )
    )

    gradcam_masked_images = (
        create_masked_batch(
            image,
            gradcam_ranking
        )
    )

    gradcam_confidences = (
        predict_confidences(
            model,
            gradcam_masked_images,
            predicted_class_index
        )
    )

    gradcam_auc = compute_auc(
        gradcam_confidences
    )

    print(
        f"Grad-CAM AUC: "
        f"{gradcam_auc:.6f}"
    )

    # --------------------------------------------------------
    # Evaluate Score-CAM
    # --------------------------------------------------------

    print()
    print("Evaluating Score-CAM...")

    _, scorecam_ranking = (
        rank_pixels(
            scorecam_map
        )
    )

    scorecam_masked_images = (
        create_masked_batch(
            image,
            scorecam_ranking
        )
    )

    scorecam_confidences = (
        predict_confidences(
            model,
            scorecam_masked_images,
            predicted_class_index
        )
    )

    scorecam_auc = compute_auc(
        scorecam_confidences
    )

    print(
        f"Score-CAM AUC: "
        f"{scorecam_auc:.6f}"
    )

    # --------------------------------------------------------
    # Store Grad-CAM result
    # --------------------------------------------------------

    gradcam_results.append(
        {
            "filename": filename,
            "image_id": image_id,
            "ground_truth": class_name,
            "predicted_class": predicted_class,
            "predicted_class_index": (
                predicted_class_index
            ),
            "original_confidence": (
                original_confidence
            ),
            "correct_prediction": (
                correct_prediction
            ),
            "mask_percentages": (
                MASK_PERCENTAGES
            ),
            "confidences": (
                gradcam_confidences
                .tolist()
            ),
            "auc": gradcam_auc
        }
    )

    # --------------------------------------------------------
    # Store Score-CAM result
    # --------------------------------------------------------

    scorecam_results.append(
        {
            "filename": filename,
            "image_id": image_id,
            "ground_truth": class_name,
            "predicted_class": predicted_class,
            "predicted_class_index": (
                predicted_class_index
            ),
            "original_confidence": (
                original_confidence
            ),
            "correct_prediction": (
                correct_prediction
            ),
            "mask_percentages": (
                MASK_PERCENTAGES
            ),
            "confidences": (
                scorecam_confidences
                .tolist()
            ),
            "auc": scorecam_auc
        }
    )

    processed_count += 1


# ============================================================
# Save per-image results
# ============================================================

gradcam_results_path = os.path.join(
    OUTPUT_DIR,
    "gradcam_results.json"
)

scorecam_results_path = os.path.join(
    OUTPUT_DIR,
    "scorecam_results.json"
)

with open(
    gradcam_results_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        {
            "method": "Grad-CAM",
            "images_evaluated": (
                len(gradcam_results)
            ),
            "mask_percentages": (
                MASK_PERCENTAGES
            ),
            "results": gradcam_results
        },
        file,
        indent=4
    )


with open(
    scorecam_results_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        {
            "method": "Score-CAM",
            "images_evaluated": (
                len(scorecam_results)
            ),
            "mask_percentages": (
                MASK_PERCENTAGES
            ),
            "results": scorecam_results
        },
        file,
        indent=4
    )


# ============================================================
# Aggregate statistics
# ============================================================

def calculate_summary(results):
    auc_values = np.array(
        [
            item["auc"]
            for item in results
        ],
        dtype=np.float32
    )

    correct_items = [
        item
        for item in results
        if item["correct_prediction"]
    ]

    incorrect_items = [
        item
        for item in results
        if not item["correct_prediction"]
    ]

    summary = {
        "n_images": int(
            len(results)
        ),
        "mean_auc": float(
            np.mean(auc_values)
        ),
        "std_auc": float(
            np.std(
                auc_values,
                ddof=1
            )
        ),
        "min_auc": float(
            np.min(auc_values)
        ),
        "max_auc": float(
            np.max(auc_values)
        ),
        "n_correct_predictions": int(
            len(correct_items)
        ),
        "n_incorrect_predictions": int(
            len(incorrect_items)
        )
    }

    if correct_items:

        correct_auc = np.array(
            [
                item["auc"]
                for item in correct_items
            ],
            dtype=np.float32
        )

        summary[
            "mean_auc_correct_only"
        ] = float(
            np.mean(
                correct_auc
            )
        )

        summary[
            "std_auc_correct_only"
        ] = float(
            np.std(
                correct_auc,
                ddof=1
            )
        )

    return summary


def calculate_mean_curve(results):
    curves = np.array(
        [
            item["confidences"]
            for item in results
        ],
        dtype=np.float32
    )

    return {
        str(
            percentage
        ): {
            "mean_confidence": float(
                np.mean(
                    curves[:, index]
                )
            ),
            "std_confidence": float(
                np.std(
                    curves[:, index],
                    ddof=1
                )
            )
        }
        for index, percentage
        in enumerate(
            MASK_PERCENTAGES
        )
    }


# ============================================================
# Overall summary
# ============================================================

summary = {
    "model": (
        "ResNet50 Fine-Tuned"
    ),
    "dataset": {
        "total_images": (
            len(selected_images)
        ),
        "classes": {
            class_name: sum(
                1
                for item in selected_images
                if item["class_name"]
                == class_name
            )
            for class_name in CLASS_NAMES
        }
    },
    "mask_percentages": (
        MASK_PERCENTAGES
    ),
    "Grad-CAM": calculate_summary(
        gradcam_results
    ),
    "Score-CAM": calculate_summary(
        scorecam_results
    ),
    "mean_curve": {
        "Grad-CAM": calculate_mean_curve(
            gradcam_results
        ),
        "Score-CAM": calculate_mean_curve(
            scorecam_results
        )
    }
}


summary_path = os.path.join(
    OUTPUT_DIR,
    "summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


# ============================================================
# Per-class summary
# ============================================================

per_class_summary = {}

for class_name in CLASS_NAMES:

    gradcam_class_results = [
        item
        for item in gradcam_results
        if item["ground_truth"]
        == class_name
    ]

    scorecam_class_results = [
        item
        for item in scorecam_results
        if item["ground_truth"]
        == class_name
    ]

    per_class_summary[
        class_name
    ] = {
        "n_images": len(
            gradcam_class_results
        ),
        "Grad-CAM": calculate_summary(
            gradcam_class_results
        ),
        "Score-CAM": calculate_summary(
            scorecam_class_results
        ),
        "mean_curve": {
            "Grad-CAM": calculate_mean_curve(
                gradcam_class_results
            ),
            "Score-CAM": calculate_mean_curve(
                scorecam_class_results
            )
        }
    }


per_class_path = os.path.join(
    OUTPUT_DIR,
    "per_class_summary.json"
)

with open(
    per_class_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        per_class_summary,
        file,
        indent=4
    )


# ============================================================
# Final console summary
# ============================================================

print()
print("=" * 70)
print("Multi-image Deletion Test completed.")
print("=" * 70)

print()
print(
    f"Images evaluated: "
    f"{processed_count}"
)

print()
print("Grad-CAM:")
print(
    f"  Mean AUC: "
    f"{summary['Grad-CAM']['mean_auc']:.6f}"
)
print(
    f"  SD AUC:   "
    f"{summary['Grad-CAM']['std_auc']:.6f}"
)

print()
print("Score-CAM:")
print(
    f"  Mean AUC: "
    f"{summary['Score-CAM']['mean_auc']:.6f}"
)
print(
    f"  SD AUC:   "
    f"{summary['Score-CAM']['std_auc']:.6f}"
)

print()
print("Output files:")
print(
    f"  {gradcam_results_path}"
)
print(
    f"  {scorecam_results_path}"
)
print(
    f"  {summary_path}"
)
print(
    f"  {per_class_path}"
)

print()
print("=" * 70)