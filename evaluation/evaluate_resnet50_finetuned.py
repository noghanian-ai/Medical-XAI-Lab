import os
import json

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from training.data_loader import load_dataset
from config import BASE_DIR, CLASS_NAMES


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "resnet50_3class_finetuned.keras"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "Evaluation_ResNet50_FineTuned"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Load validation dataset
# ---------------------------------------------------------

_, val_ds = load_dataset()


def preprocess_resnet50(images, labels):
    images = tf.keras.applications.resnet50.preprocess_input(
        tf.cast(images, tf.float32)
    )
    return images, labels


val_ds = val_ds.map(
    preprocess_resnet50,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------
# Load fine-tuned model
# ---------------------------------------------------------

print()
print("Loading fine-tuned ResNet50 model...")
print(MODEL_PATH)

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ---------------------------------------------------------
# Generate predictions
# ---------------------------------------------------------

y_true = []
y_pred = []


for images, labels in val_ds:
    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ---------------------------------------------------------
# Calculate metrics
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


# ---------------------------------------------------------
# Classification report
# ---------------------------------------------------------

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred
)


# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

print()
print("=" * 60)
print("ResNet50 Fine-Tuned Evaluation")
print("=" * 60)

print()
print(f"Validation images: {len(y_true)}")

print()
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print()
print("Classification Report:")
print(report)

print()
print("Confusion Matrix:")
print(cm)


# ---------------------------------------------------------
# Save metrics
# ---------------------------------------------------------

metrics = {
    "model": "ResNet50 Fine-Tuned",
    "validation_images": int(len(y_true)),
    "classes": len(CLASS_NAMES),
    "class_names": CLASS_NAMES,
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1)
}


metrics_path = os.path.join(
    OUTPUT_DIR,
    "metrics.json"
)


with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metrics,
        f,
        indent=4
    )


# ---------------------------------------------------------
# Save classification report
# ---------------------------------------------------------

report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:
    f.write(report)


# ---------------------------------------------------------
# Save confusion matrix
# ---------------------------------------------------------

cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.npy"
)


np.save(
    cm_path,
    cm
)


# ---------------------------------------------------------
# Finished
# ---------------------------------------------------------

print()
print("Evaluation completed.")

print()
print("Outputs saved to:")
print(OUTPUT_DIR)

print()
print("Files:")
print("- metrics.json")
print("- classification_report.txt")
print("- confusion_matrix.npy")