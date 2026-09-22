import os
import json

from config import BASE_DIR


# ---------------------------------------------------------
# Input files
# ---------------------------------------------------------

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

RESNET50_FINETUNED_METRICS_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "Evaluation_ResNet50_FineTuned",
    "metrics.json"
)


# ---------------------------------------------------------
# Output
# ---------------------------------------------------------

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "model_comparison_three_models.json"
)


# ---------------------------------------------------------
# Load metrics
# ---------------------------------------------------------

with open(
    BASELINE_METRICS_PATH,
    "r",
    encoding="utf-8"
) as f:
    baseline = json.load(f)


with open(
    RESNET50_METRICS_PATH,
    "r",
    encoding="utf-8"
) as f:
    resnet50 = json.load(f)


with open(
    RESNET50_FINETUNED_METRICS_PATH,
    "r",
    encoding="utf-8"
) as f:
    resnet50_finetuned = json.load(f)


# ---------------------------------------------------------
# Build comparison
# ---------------------------------------------------------

comparison = {
    "validation_images": 600,
    "classes": 3,
    "class_names": [
        "COVID",
        "Lung_Opacity",
        "Normal"
    ],
    "models": {
        "Custom_VGG16_like_CNN": {
            "description": (
                "Custom VGG16-like CNN trained from scratch"
            ),
            "accuracy": baseline["accuracy"],
            "precision": baseline["precision"],
            "recall": baseline["recall"],
            "f1_score": baseline["f1_score"]
        },
        "ResNet50_Transfer_Learning": {
            "description": (
                "ResNet50 pretrained on ImageNet "
                "with frozen backbone"
            ),
            "accuracy": resnet50["accuracy"],
            "precision": resnet50["precision"],
            "recall": resnet50["recall"],
            "f1_score": resnet50["f1_score"]
        },
        "ResNet50_Fine_Tuned": {
            "description": (
                "ResNet50 pretrained on ImageNet "
                "with the final layers fine-tuned"
            ),
            "accuracy": resnet50_finetuned["accuracy"],
            "precision": resnet50_finetuned["precision"],
            "recall": resnet50_finetuned["recall"],
            "f1_score": resnet50_finetuned["f1_score"]
        }
    }
}


# ---------------------------------------------------------
# Save comparison
# ---------------------------------------------------------

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        comparison,
        f,
        indent=4
    )


# ---------------------------------------------------------
# Print summary
# ---------------------------------------------------------

print()
print("=" * 65)
print("Three-Model Comparison")
print("=" * 65)

for model_name, metrics in comparison["models"].items():
    print()
    print(model_name)
    print(
        f"  Accuracy : {metrics['accuracy']:.4f}"
    )
    print(
        f"  Precision: {metrics['precision']:.4f}"
    )
    print(
        f"  Recall   : {metrics['recall']:.4f}"
    )
    print(
        f"  F1 Score : {metrics['f1_score']:.4f}"
    )

print()
print("Comparison saved to:")
print(OUTPUT_PATH)