import os
import json
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

GRADCAM_HEATMAP_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "GradCAM_ResNet50_FineTuned",
    "heatmap.png"
)

SCORECAM_HEATMAP_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "ScoreCAM_ResNet50_FineTuned",
    "heatmap.png"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "DeletionTest_ResNet50_FineTuned"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = [
    "COVID",
    "Lung_Opacity",
    "Normal"
]

TARGET_CLASS = "Lung_Opacity"
TARGET_CLASS_INDEX = CLASS_NAMES.index(TARGET_CLASS)

DELETION_PERCENTAGES = [
    0,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100
]

PREDICTION_BATCH_SIZE = 4


# ============================================================
# Utility functions
# ============================================================

def load_image(image_path):
    """
    Load an RGB image and resize it to 224x224.
    """
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image:\n{image_path}"
        )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    return image.astype(np.float32)


def load_heatmap(heatmap_path):
    """
    Load an existing heatmap visualization and convert it
    into a normalized grayscale importance map.

    Note:
    The stored heatmap is a visualization PNG rather than
    the original raw XAI array. Therefore this represents
    the intensity of the saved visualization.
    """
    heatmap = cv2.imread(heatmap_path)

    if heatmap is None:
        raise FileNotFoundError(
            f"Could not load heatmap:\n{heatmap_path}"
        )

    heatmap = cv2.resize(
        heatmap,
        (224, 224),
        interpolation=cv2.INTER_LINEAR
    )

    gray = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2GRAY
    )

    gray = gray.astype(np.float32)

    min_value = np.min(gray)
    max_value = np.max(gray)

    if max_value > min_value:
        gray = (
            (gray - min_value)
            / (max_value - min_value)
        )
    else:
        gray = np.zeros_like(gray)

    return gray


def create_masked_image(
        image,
        importance_map,
        deletion_percentage
):
    """
    Mask the most important pixels according to the
    XAI importance map.

    A per-channel mean pixel value is used as the
    neutral masking baseline instead of black pixels.
    """

    masked = image.copy()

    total_pixels = importance_map.size

    number_to_delete = int(
        total_pixels * deletion_percentage / 100.0
    )

    if number_to_delete <= 0:
        return masked

    if number_to_delete >= total_pixels:
        number_to_delete = total_pixels

    flat_importance = importance_map.reshape(-1)

    # Indices of pixels with highest importance.
    important_indices = np.argpartition(
        flat_importance,
        -number_to_delete
    )[-number_to_delete:]

    flat_masked = masked.reshape(-1, 3)

    # Mean RGB value of the original image.
    baseline = np.mean(
        image.reshape(-1, 3),
        axis=0
    )

    flat_masked[important_indices] = baseline

    return flat_masked.reshape(image.shape)


def predict_target_confidence(
        model,
        image
):
    """
    Predict the target-class confidence for one image.
    """

    batch = np.expand_dims(image, axis=0)

    batch = tf.keras.applications.resnet50.preprocess_input(
        batch
    )

    prediction = model.predict(
        batch,
        verbose=0
    )

    return float(
        prediction[0, TARGET_CLASS_INDEX]
    )


def run_deletion_test(
        model,
        original_image,
        importance_map,
        method_name
):
    """
    Run deletion test for one XAI method.
    """

    results = []

    print()
    print("=" * 60)
    print(f"Running deletion test: {method_name}")
    print("=" * 60)

    for percentage in DELETION_PERCENTAGES:

        masked_image = create_masked_image(
            original_image,
            importance_map,
            percentage
        )

        confidence = predict_target_confidence(
            model,
            masked_image
        )

        result = {
            "deletion_percentage": percentage,
            "remaining_percentage": 100 - percentage,
            "target_class": TARGET_CLASS,
            "target_confidence": confidence
        }

        results.append(result)

        print(
            f"Deleted: {percentage:3d}%"
            f" | Remaining: {100 - percentage:3d}%"
            f" | {TARGET_CLASS} confidence:"
            f" {confidence:.6f}"
            f" ({confidence * 100:.2f}%)"
        )

    return results


def calculate_auc(results):
    """
    Calculate area under the deletion curve.

    Lower AUC generally means confidence decreases more
    rapidly when important regions are removed.
    """

    x = np.array(
        [r["deletion_percentage"] for r in results],
        dtype=np.float32
    )

    y = np.array(
        [r["target_confidence"] for r in results],
        dtype=np.float32
    )

    return float(np.trapz(y, x))


def save_results(results, method_name):
    """
    Save deletion results as JSON.
    """

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{method_name.lower()}_deletion_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            indent=4
        )

    return output_path


def plot_deletion_curves(
        gradcam_results,
        scorecam_results
):
    """
    Plot Grad-CAM and Score-CAM deletion curves.
    """

    gradcam_x = [
        r["deletion_percentage"]
        for r in gradcam_results
    ]

    gradcam_y = [
        r["target_confidence"]
        for r in gradcam_results
    ]

    scorecam_x = [
        r["deletion_percentage"]
        for r in scorecam_results
    ]

    scorecam_y = [
        r["target_confidence"]
        for r in scorecam_results
    ]

    plt.figure(figsize=(9, 6))

    plt.plot(
        gradcam_x,
        gradcam_y,
        marker="o",
        label="Grad-CAM"
    )

    plt.plot(
        scorecam_x,
        scorecam_y,
        marker="s",
        label="Score-CAM"
    )

    plt.xlabel(
        "Deleted Pixels (%)"
    )

    plt.ylabel(
        "Target Class Confidence"
    )

    plt.title(
        "Deletion Test - ResNet50 Fine-Tuned"
    )

    plt.ylim(0, 1.05)

    plt.grid(True, alpha=0.3)

    plt.legend()

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "deletion_curves.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()

    return output_path


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("ResNet50 Fine-Tuned - Deletion Test")
    print("=" * 60)

    print()
    print("Loading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully.")

    print()
    print("Loading target image...")

    original_image = load_image(
        IMAGE_PATH
    )

    print(
        f"Image shape: {original_image.shape}"
    )

    print()
    print("Loading Grad-CAM heatmap...")

    gradcam_heatmap = load_heatmap(
        GRADCAM_HEATMAP_PATH
    )

    print("Grad-CAM heatmap loaded.")

    print()
    print("Loading Score-CAM heatmap...")

    scorecam_heatmap = load_heatmap(
        SCORECAM_HEATMAP_PATH
    )

    print("Score-CAM heatmap loaded.")

    print()
    print("Checking original prediction...")

    original_confidence = predict_target_confidence(
        model,
        original_image
    )

    print(
        f"Original {TARGET_CLASS} confidence:"
        f" {original_confidence:.6f}"
        f" ({original_confidence * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    gradcam_results = run_deletion_test(
        model,
        original_image,
        gradcam_heatmap,
        "Grad-CAM"
    )

    # --------------------------------------------------------
    # Score-CAM
    # --------------------------------------------------------

    scorecam_results = run_deletion_test(
        model,
        original_image,
        scorecam_heatmap,
        "Score-CAM"
    )

    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------

    gradcam_auc = calculate_auc(
        gradcam_results
    )

    scorecam_auc = calculate_auc(
        scorecam_results
    )

    print()
    print("=" * 60)
    print("Deletion Test Summary")
    print("=" * 60)

    print(
        f"Grad-CAM AUC : {gradcam_auc:.6f}"
    )

    print(
        f"Score-CAM AUC: {scorecam_auc:.6f}"
    )

    # --------------------------------------------------------
    # Save individual results
    # --------------------------------------------------------

    gradcam_json = save_results(
        gradcam_results,
        "GradCAM"
    )

    scorecam_json = save_results(
        scorecam_results,
        "ScoreCAM"
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = {
        "model": "ResNet50 Fine-Tuned",
        "target_image": IMAGE_PATH,
        "target_class": TARGET_CLASS,
        "target_class_index": TARGET_CLASS_INDEX,
        "original_target_confidence": original_confidence,
        "gradcam_auc": gradcam_auc,
        "scorecam_auc": scorecam_auc,
        "deletion_percentages": DELETION_PERCENTAGES,
        "masking_baseline": "per-channel image mean",
        "note": (
            "Importance maps were derived from the saved "
            "heatmap PNG visualizations rather than raw "
            "XAI arrays."
        )
    }

    summary_path = os.path.join(
        OUTPUT_DIR,
        "deletion_test_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_path = plot_deletion_curves(
        gradcam_results,
        scorecam_results
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Deletion test completed successfully.")
    print("=" * 60)

    print()
    print("Saved files:")

    print(gradcam_json)
    print(scorecam_json)
    print(summary_path)
    print(plot_path)


if __name__ == "__main__":
    main()