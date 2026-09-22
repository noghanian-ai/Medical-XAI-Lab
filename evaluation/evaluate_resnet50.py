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
from config import CLASS_NAMES, BASE_DIR


# =========================
# Paths
# =========================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "resnet50_3class.keras"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "Evaluation_ResNet50"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# Load Model
# =========================

print("Loading ResNet50 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# =========================
# Load Validation Dataset
# =========================

print()
print("Loading validation dataset...")

_, val_ds = load_dataset()


# =========================
# ResNet50 Preprocessing
# =========================

def preprocess_resnet50(images, labels):

    images = tf.keras.applications.resnet50.preprocess_input(
        tf.cast(images, tf.float32)
    )

    return images, labels


val_ds = val_ds.map(
    preprocess_resnet50,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.prefetch(
    tf.data.AUTOTUNE
)


# =========================
# Predictions
# =========================

print()
print("Running predictions...")

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

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_classes
    )


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# =========================
# Metrics
# =========================

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


# =========================
# Print Results
# =========================

print()
print("========================================")
print("ResNet50 Evaluation Results")
print("========================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# =========================
# Classification Report
# =========================

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print()
print("Classification Report")
print("----------------------------------------")
print(report)


# =========================
# Confusion Matrix
# =========================

cm = confusion_matrix(
    y_true,
    y_pred
)

print()
print("Confusion Matrix")
print("----------------------------------------")
print(cm)


# =========================
# Save Metrics
# =========================

metrics = {
    "model": "ResNet50",
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


# =========================
# Save Classification Report
# =========================

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


# =========================
# Save Confusion Matrix
# =========================

cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.npy"
)

np.save(
    cm_path,
    cm
)


# =========================
# Final Message
# =========================

print()
print("Evaluation completed.")

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("Saved files:")
print(metrics_path)
print(report_path)
print(cm_path)