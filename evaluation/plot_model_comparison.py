import json
import os

import matplotlib.pyplot as plt

from config import BASE_DIR


METRICS_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "model_comparison.json"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "Model_Comparison"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "model_metrics_comparison.png"
)


with open(METRICS_PATH, "r", encoding="utf-8") as file:
    comparison = json.load(file)


models = list(comparison["models"].keys())

display_names = [
    "Custom VGG16-like CNN",
    "ResNet50 Transfer Learning"
]

metric_keys = [
    "accuracy",
    "precision",
    "recall",
    "f1_score"
]

metric_labels = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score"
]


values = []

for model in models:
    model_values = [
        comparison["models"][model]["metrics"][metric]
        for metric in metric_keys
    ]
    values.append(model_values)


x = range(len(metric_keys))

width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width / 2 for i in x],
    values[0],
    width=width,
    label=display_names[0]
)

plt.bar(
    [i + width / 2 for i in x],
    values[1],
    width=width,
    label=display_names[1]
)

plt.xticks(x, metric_labels)

plt.ylim(0.85, 1.0)

plt.ylabel("Score")

plt.title(
    "Model Performance Comparison"
)

plt.legend()

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print()
print("Model comparison plot created.")
print()
print("Saved to:")
print(OUTPUT_PATH)