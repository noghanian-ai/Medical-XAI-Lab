import os
import cv2
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GRADCAM_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "GradCAM_ResNet50_FineTuned"
)

SCORECAM_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "ScoreCAM_ResNet50_FineTuned"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "XAI_Comparison_ResNet50_FineTuned"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Load images
# ============================================================

original = cv2.imread(
    os.path.join(GRADCAM_DIR, "original.png")
)

gradcam_heatmap = cv2.imread(
    os.path.join(GRADCAM_DIR, "heatmap.png")
)

gradcam_overlay = cv2.imread(
    os.path.join(GRADCAM_DIR, "overlay.png")
)

scorecam_heatmap = cv2.imread(
    os.path.join(SCORECAM_DIR, "heatmap.png")
)

scorecam_overlay = cv2.imread(
    os.path.join(SCORECAM_DIR, "overlay.png")
)


# ============================================================
# Validate files
# ============================================================

images = {
    "Original": original,
    "Grad-CAM Heatmap": gradcam_heatmap,
    "Grad-CAM Overlay": gradcam_overlay,
    "Score-CAM Heatmap": scorecam_heatmap,
    "Score-CAM Overlay": scorecam_overlay,
}

for name, image in images.items():
    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {name}"
        )


# ============================================================
# Convert BGR → RGB
# ============================================================

images_rgb = {}

for name, image in images.items():
    images_rgb[name] = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# Create comparison figure
# ============================================================

fig, axes = plt.subplots(
    1,
    5,
    figsize=(20, 4)
)

for ax, (name, image) in zip(
    axes,
    images_rgb.items()
):
    ax.imshow(image)
    ax.set_title(name)
    ax.axis("off")


fig.suptitle(
    "Grad-CAM vs Score-CAM — ResNet50 Fine-Tuned",
    fontsize=16
)

plt.tight_layout()


# ============================================================
# Save comparison
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "gradcam_vs_scorecam.png"
)

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print()
print("XAI comparison completed successfully.")
print()
print("Saved file:")
print(output_path)