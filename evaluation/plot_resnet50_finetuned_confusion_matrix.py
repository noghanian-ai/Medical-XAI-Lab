import os

import numpy as np
import matplotlib.pyplot as plt

from config import BASE_DIR, CLASS_NAMES


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

INPUT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "Evaluation_ResNet50_FineTuned",
    "confusion_matrix.npy"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "Evaluation_ResNet50_FineTuned"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)


# ---------------------------------------------------------
# Load confusion matrix
# ---------------------------------------------------------

cm = np.load(INPUT_PATH)


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title("ResNet50 Fine-Tuned - Confusion Matrix")

plt.colorbar()

plt.xticks(
    np.arange(len(CLASS_NAMES)),
    CLASS_NAMES,
    rotation=45,
    ha="right"
)

plt.yticks(
    np.arange(len(CLASS_NAMES)),
    CLASS_NAMES
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")


# Add values to cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print()
print("Confusion matrix visualization created.")
print()
print("Saved to:")
print(OUTPUT_PATH)