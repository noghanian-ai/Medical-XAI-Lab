import os
import numpy as np
import matplotlib.pyplot as plt

from config import CLASS_NAMES, BASE_DIR


# =========================
# Paths
# =========================

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "Evaluation_ResNet50"
)

CM_PATH = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.npy"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)


# =========================
# Load Confusion Matrix
# =========================

cm = np.load(CM_PATH)


# =========================
# Plot
# =========================

fig, ax = plt.subplots(
    figsize=(7, 6)
)

im = ax.imshow(cm)

ax.set_title(
    "ResNet50 Confusion Matrix"
)

ax.set_xlabel(
    "Predicted Class"
)

ax.set_ylabel(
    "True Class"
)

ax.set_xticks(
    np.arange(len(CLASS_NAMES))
)

ax.set_yticks(
    np.arange(len(CLASS_NAMES))
)

ax.set_xticklabels(
    CLASS_NAMES,
    rotation=30,
    ha="right"
)

ax.set_yticklabels(
    CLASS_NAMES
)


# =========================
# Add Values
# =========================

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )


fig.colorbar(im, ax=ax)

fig.tight_layout()

fig.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# =========================
# Final Message
# =========================

print("Confusion matrix generated successfully.")
print()
print("Saved to:")
print(OUTPUT_PATH)