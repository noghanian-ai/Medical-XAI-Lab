import os
import json
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

INPUT_DIR = (
    "outputs/DeletionTest_Raw_XAI_ResNet50_FineTuned"
)

OUTPUT_DIR = INPUT_DIR

GRADCAM_FILE = os.path.join(
    INPUT_DIR,
    "gradcam_raw_deletion_results.json"
)

SCORECAM_FILE = os.path.join(
    INPUT_DIR,
    "scorecam_raw_deletion_results.json"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "raw_deletion_curves.png"
)


# ============================================================
# Load results
# ============================================================

def load_results(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


gradcam = load_results(GRADCAM_FILE)
scorecam = load_results(SCORECAM_FILE)


# ============================================================
# Extract data
# ============================================================

gradcam_percentages = [
    item["masked_percentage"]
    for item in gradcam["results"]
]

gradcam_confidences = [
    item["confidence"] * 100
    for item in gradcam["results"]
]

scorecam_percentages = [
    item["masked_percentage"]
    for item in scorecam["results"]
]

scorecam_confidences = [
    item["confidence"] * 100
    for item in scorecam["results"]
]


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    gradcam_percentages,
    gradcam_confidences,
    marker="o",
    linewidth=2,
    label="Grad-CAM"
)

plt.plot(
    scorecam_percentages,
    scorecam_confidences,
    marker="o",
    linewidth=2,
    label="Score-CAM"
)

plt.xlabel(
    "Percentage of Most Important Regions Masked"
)

plt.ylabel(
    "Lung_Opacity Confidence (%)"
)

plt.title(
    "Deletion Curves — ResNet50 Fine-Tuned"
)

plt.xlim(0, 100)
plt.ylim(0, 105)

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("=" * 60)
print("Raw XAI Deletion Curve")
print("=" * 60)

print()
print("Grad-CAM AUC:")
print(f"{gradcam['auc']:.6f}")

print()
print("Score-CAM AUC:")
print(f"{scorecam['auc']:.6f}")

print()
print("Saved:")
print(OUTPUT_FILE)