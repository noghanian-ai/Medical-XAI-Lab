import os
import json

import numpy as np
import matplotlib.pyplot as plt

from config import BASE_DIR


# ---------------------------------------------------------
# Input
# ---------------------------------------------------------

INPUT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "model_comparison_three_models.json"
)


# ---------------------------------------------------------
# Output
# ---------------------------------------------------------

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "Model_Comparison"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "three_model_metrics_comparison.png"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Load comparison data
# ---------------------------------------------------------

with open(
    INPUT_PATH,
    "r",
    encoding="utf-8"
) as f:
    comparison = json.load(f)


models = list(comparison["models"].keys())

display_names = [
    "Custom VGG16-like CNN",
    "ResNet50 Frozen",
    "ResNet50 Fine-Tuned"
]

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1_score"
]

metric_labels = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]


# ---------------------------------------------------------
# Extract values
# ---------------------------------------------------------

values = []

for model in models:
    model_values = [
        comparison["models"][model][metric] * 100
        for metric in metrics
    ]
    values.append(model_values)

values = np.array(values)


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

x = np.arange(len(metrics))
width = 0.24

plt.figure(figsize=(10, 6))

for i, model_values in enumerate(values):
    plt.bar(
        x + (i - 1) * width,
        model_values,
        width,
        label=display_names[i]
    )


plt.title(
    "Three-Model Performance Comparison"
)

plt.xlabel("Evaluation Metric")
plt.ylabel("Percentage (%)")

plt.xticks(
    x,
    metric_labels
)

plt.ylim(90, 100)

plt.legend()

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# Finished
# ---------------------------------------------------------

print()
print("Three-model comparison chart created.")
print()
print("Saved to:")
print(OUTPUT_PATH)