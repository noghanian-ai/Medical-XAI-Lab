import json
import os

from config import BASE_DIR


BASELINE_METRICS_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "Evaluation",
    "metrics.json"
)

RESNET50_METRICS_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "Evaluation_ResNet50",
    "metrics.json"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "model_comparison.json"
)


def load_metrics(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


baseline_metrics = load_metrics(BASELINE_METRICS_PATH)
resnet50_metrics = load_metrics(RESNET50_METRICS_PATH)


comparison = {
    "dataset": {
        "validation_images": 600,
        "classes": 3,
        "class_names": [
            "COVID",
            "Lung_Opacity",
            "Normal"
        ]
    },
    "models": {
        "Custom_VGG16_like_CNN": {
            "description": "Custom VGG16-like CNN trained from scratch",
            "metrics": baseline_metrics
        },
        "ResNet50_Transfer_Learning": {
            "description": "ResNet50 pretrained on ImageNet with frozen backbone",
            "metrics": resnet50_metrics
        }
    }
}


with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(comparison, file, indent=4)


print()
print("Model comparison completed.")
print()
print("Models:")
print("- Custom VGG16-like CNN")
print("- ResNet50 Transfer Learning")
print()
print("Validation images:", comparison["dataset"]["validation_images"])
print("Classes:", comparison["dataset"]["classes"])
print()
print("Comparison saved to:")
print(OUTPUT_PATH)