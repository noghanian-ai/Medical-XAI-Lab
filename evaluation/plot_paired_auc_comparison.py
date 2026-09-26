import json
import os

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "XAI_Evaluation_Dataset",
    "Deletion_Test",
)

STATISTICS_DIR = os.path.join(
    RESULTS_DIR,
    "Statistical_Analysis",
)

GRADCAM_FILE = os.path.join(
    RESULTS_DIR,
    "gradcam_results.json",
)

SCORECAM_FILE = os.path.join(
    RESULTS_DIR,
    "scorecam_results.json",
)

OUTPUT_FILE = os.path.join(
    STATISTICS_DIR,
    "paired_auc_comparison.png",
)


# ============================================================
# Load results
# ============================================================

with open(GRADCAM_FILE, "r", encoding="utf-8") as f:
    gradcam_results = json.load(f)

with open(SCORECAM_FILE, "r", encoding="utf-8") as f:
    scorecam_results = json.load(f)


# ============================================================
# Normalize JSON structure
# ============================================================

def extract_records(data):
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["results", "images", "records", "data"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError("Unable to find result records in JSON file.")


gradcam_records = extract_records(gradcam_results)
scorecam_records = extract_records(scorecam_results)


# ============================================================
# Build lookup dictionaries
# ============================================================

def get_filename(record):
    for key in ["filename", "image", "image_name", "file"]:
        if key in record:
            return record[key]

    raise KeyError("Filename field not found in result record.")


def get_auc(record):
    for key in ["auc", "AUC", "deletion_auc"]:
        if key in record:
            return float(record[key])

    raise KeyError("AUC field not found in result record.")


gradcam_by_file = {
    get_filename(record): record
    for record in gradcam_records
}

scorecam_by_file = {
    get_filename(record): record
    for record in scorecam_records
}


common_files = sorted(
    set(gradcam_by_file.keys()) &
    set(scorecam_by_file.keys())
)


if not common_files:
    raise RuntimeError("No matching image filenames found.")


# ============================================================
# Keep class order consistent with evaluation dataset
# ============================================================

class_order = {
    "COVID": 0,
    "Lung_Opacity": 1,
    "Normal": 2,
}


def get_class(filename):
    for class_name in class_order:
        if filename.startswith(class_name):
            return class_name

    return "Unknown"


common_files = sorted(
    common_files,
    key=lambda filename: (
        class_order.get(get_class(filename), 99),
        filename,
    ),
)


# ============================================================
# Extract values
# ============================================================

filenames = []
classes = []
gradcam_auc = []
scorecam_auc = []

for filename in common_files:
    filenames.append(filename)
    classes.append(get_class(filename))

    gradcam_auc.append(
        get_auc(gradcam_by_file[filename])
    )

    scorecam_auc.append(
        get_auc(scorecam_by_file[filename])
    )


gradcam_auc = np.array(gradcam_auc)
scorecam_auc = np.array(scorecam_auc)

x = np.arange(len(filenames))


# ============================================================
# Identify incorrect prediction
# ============================================================

incorrect_indices = []

for index, filename in enumerate(filenames):
    record = gradcam_by_file[filename]

    correct = record.get("correct")

    if correct is False:
        incorrect_indices.append(index)


# ============================================================
# Create output directory
# ============================================================

os.makedirs(STATISTICS_DIR, exist_ok=True)


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(16, 7))

for index in range(len(filenames)):
    ax.plot(
        [x[index], x[index]],
        [gradcam_auc[index], scorecam_auc[index]],
        linewidth=0.8,
        alpha=0.5,
    )

ax.plot(
    x,
    gradcam_auc,
    marker="o",
    markersize=4,
    linewidth=1.5,
    label="Grad-CAM",
)

ax.plot(
    x,
    scorecam_auc,
    marker="s",
    markersize=4,
    linewidth=1.5,
    label="Score-CAM",
)


# Highlight incorrectly classified image(s)
for index in incorrect_indices:
    ax.scatter(
        x[index],
        gradcam_auc[index],
        marker="x",
        s=80,
        linewidths=2,
    )

    ax.scatter(
        x[index],
        scorecam_auc[index],
        marker="x",
        s=80,
        linewidths=2,
    )


# ============================================================
# Class boundaries
# ============================================================

class_boundaries = []

for index in range(1, len(classes)):
    if classes[index] != classes[index - 1]:
        class_boundaries.append(index - 0.5)

for boundary in class_boundaries:
    ax.axvline(
        boundary,
        linewidth=1,
        linestyle="--",
        alpha=0.4,
    )


# ============================================================
# Axis labels
# ============================================================

ax.set_xlabel("Evaluation image")
ax.set_ylabel("Deletion AUC")
ax.set_title(
    "Paired Deletion AUC Comparison: Grad-CAM vs Score-CAM"
)

ax.set_xticks(x)
ax.set_xticklabels(
    [
        filename.replace(".png", "")
        for filename in filenames
    ],
    rotation=90,
    fontsize=7,
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25,
)

plt.tight_layout()


# ============================================================
# Save
# ============================================================

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


print("=" * 70)
print("Paired AUC comparison plot created successfully.")
print("=" * 70)
print(f"Images: {len(filenames)}")
print(f"Output: {OUTPUT_FILE}")

if incorrect_indices:
    print()
    print("Incorrectly classified image(s):")

    for index in incorrect_indices:
        print(
            f"  {filenames[index]} "
            f"(Grad-CAM AUC={gradcam_auc[index]:.4f}, "
            f"Score-CAM AUC={scorecam_auc[index]:.4f})"
        )

print("=" * 70)