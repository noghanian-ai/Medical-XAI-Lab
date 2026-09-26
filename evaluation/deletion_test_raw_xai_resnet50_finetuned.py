import os
import json
import cv2
import numpy as np
import tensorflow as tf

# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/resnet50_3class_finetuned.keras"

IMAGE_PATH = "images/Lung_Opacity/Lung_Opacity-1000.png"

RAW_XAI_DIR = (
    "outputs/Raw_XAI_Maps_ResNet50_FineTuned"
)

OUTPUT_DIR = (
    "outputs/DeletionTest_Raw_XAI_ResNet50_FineTuned"
)

CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal"]
TARGET_CLASS = "Lung_Opacity"
TARGET_CLASS_INDEX = CLASS_NAMES.index(TARGET_CLASS)

IMG_SIZE = (224, 224)

MASK_PERCENTAGES = [
    0, 10, 20, 30, 40,
    50, 60, 70, 80,
    90, 100
]


# ============================================================
# Utility functions
# ============================================================

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMG_SIZE)

    return image


def preprocess_for_resnet50(image):
    image = image.astype(np.float32)
    image = tf.keras.applications.resnet50.preprocess_input(
        image
    )
    return image


def normalize_map(xai_map):
    xai_map = np.asarray(xai_map, dtype=np.float32)

    xai_map = np.maximum(xai_map, 0)

    min_value = np.min(xai_map)
    max_value = np.max(xai_map)

    if max_value > min_value:
        xai_map = (
            (xai_map - min_value)
            / (max_value - min_value)
        )
    else:
        xai_map = np.zeros_like(xai_map)

    return xai_map


def resize_xai_map(xai_map):
    return cv2.resize(
        xai_map,
        IMG_SIZE,
        interpolation=cv2.INTER_LINEAR
    )


def create_masked_image(
    original_image,
    importance_map,
    percentage
):
    height, width = importance_map.shape

    total_pixels = height * width

    pixels_to_mask = int(
        total_pixels * percentage / 100.0
    )

    masked_image = original_image.copy()

    if pixels_to_mask <= 0:
        return masked_image

    if pixels_to_mask >= total_pixels:
        mask = np.ones(
            (height, width),
            dtype=bool
        )
    else:
        flat_indices = np.argsort(
            importance_map.reshape(-1)
        )[::-1]

        selected_indices = flat_indices[
            :pixels_to_mask
        ]

        mask = np.zeros(
            total_pixels,
            dtype=bool
        )

        mask[selected_indices] = True
        mask = mask.reshape(height, width)

    resized_mask = cv2.resize(
        mask.astype(np.uint8),
        IMG_SIZE,
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    mean_pixel = np.mean(
        original_image,
        axis=(0, 1),
        keepdims=True
    )

    masked_image[resized_mask] = mean_pixel

    return masked_image


def predict_target_confidence(
    model,
    image
):
    processed = preprocess_for_resnet50(
        image
    )

    processed = np.expand_dims(
        processed,
        axis=0
    )

    prediction = model.predict(
        processed,
        verbose=0
    )[0]

    return float(
        prediction[TARGET_CLASS_INDEX]
    )


def run_deletion_test(
    model,
    original_image,
    raw_xai_map,
    method_name
):
    normalized_map = normalize_map(
        raw_xai_map
    )

    results = []

    print()
    print("=" * 60)
    print(f"Deletion Test: {method_name}")
    print("=" * 60)

    for percentage in MASK_PERCENTAGES:

        masked_image = create_masked_image(
            original_image,
            normalized_map,
            percentage
        )

        confidence = predict_target_confidence(
            model,
            masked_image
        )

        result = {
            "masked_percentage": percentage,
            "target_class": TARGET_CLASS,
            "confidence": confidence
        }

        results.append(result)

        print(
            f"{percentage:3d}% masked -> "
            f"{confidence * 100:.2f}%"
        )

    percentages = np.array(
        [r["masked_percentage"] for r in results],
        dtype=np.float32
    )

    confidences = np.array(
        [r["confidence"] for r in results],
        dtype=np.float32
    )

    auc = float(
        np.trapz(
            confidences,
            percentages
        )
    )

    return {
        "method": method_name,
        "target_class": TARGET_CLASS,
        "results": results,
        "auc": auc
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Raw XAI Deletion Test")
    print("ResNet50 Fine-Tuned")
    print("=" * 60)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print()
    print("Loading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully.")

    print()
    print("Loading image...")

    original_image = load_and_preprocess_image(
        IMAGE_PATH
    )

    print(
        f"Image shape: {original_image.shape}"
    )

    print()
    print("Running original prediction...")

    original_confidence = (
        predict_target_confidence(
            model,
            original_image
        )
    )

    print(
        f"Target class: {TARGET_CLASS}"
    )

    print(
        f"Original confidence: "
        f"{original_confidence * 100:.2f}%"
    )

    gradcam_path = os.path.join(
        RAW_XAI_DIR,
        "gradcam_raw.npy"
    )

    scorecam_path = os.path.join(
        RAW_XAI_DIR,
        "scorecam_raw.npy"
    )

    print()
    print("Loading raw XAI maps...")

    gradcam_raw = np.load(
        gradcam_path
    )

    scorecam_raw = np.load(
        scorecam_path
    )

    print(
        f"Grad-CAM raw shape: "
        f"{gradcam_raw.shape}"
    )

    print(
        f"Score-CAM raw shape: "
        f"{scorecam_raw.shape}"
    )

    gradcam_results = run_deletion_test(
        model,
        original_image,
        gradcam_raw,
        "Grad-CAM"
    )

    scorecam_results = run_deletion_test(
        model,
        original_image,
        scorecam_raw,
        "Score-CAM"
    )

    summary = {
        "model": MODEL_PATH,
        "image": IMAGE_PATH,
        "target_class": TARGET_CLASS,
        "original_confidence": original_confidence,
        "mask_percentages": MASK_PERCENTAGES,
        "gradcam_auc": gradcam_results["auc"],
        "scorecam_auc": scorecam_results["auc"]
    }

    gradcam_output = os.path.join(
        OUTPUT_DIR,
        "gradcam_raw_deletion_results.json"
    )

    scorecam_output = os.path.join(
        OUTPUT_DIR,
        "scorecam_raw_deletion_results.json"
    )

    summary_output = os.path.join(
        OUTPUT_DIR,
        "deletion_test_raw_summary.json"
    )

    with open(
        gradcam_output,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            gradcam_results,
            f,
            indent=4
        )

    with open(
        scorecam_output,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            scorecam_results,
            f,
            indent=4
        )

    with open(
        summary_output,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=4
        )

    print()
    print("=" * 60)
    print("Deletion Test Completed")
    print("=" * 60)

    print()
    print(
        f"Grad-CAM AUC: "
        f"{gradcam_results['auc']:.6f}"
    )

    print(
        f"Score-CAM AUC: "
        f"{scorecam_results['auc']:.6f}"
    )

    print()
    print("Saved:")
    print(gradcam_output)
    print(scorecam_output)
    print(summary_output)


if __name__ == "__main__":
    main()