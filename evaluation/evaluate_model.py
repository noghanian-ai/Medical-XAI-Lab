import tensorflow as tf
import numpy as np
import os


from evaluation.metrics import (
    calculate_metrics,
    print_report,
    save_metrics,
    save_classification_report
)


from evaluation.confusion_matrix import (
    plot_confusion_matrix
)


from training.data_loader import load_dataset


from config import (
    MODEL_PATH,
    CLASS_NAMES,
    BASE_DIR
)


import sys
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(BASE_DIR)


from evaluation.metrics import (
    calculate_metrics,
    print_report,
    save_metrics,
    save_classification_report
)

from evaluation.confusion_matrix import (
    plot_confusion_matrix
)

from training.data_loader import load_dataset

from config import MODEL_PATH, CLASS_NAMES

# ==========================
# Load Model
# ==========================

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully")



# ==========================
# Load Validation Dataset
# ==========================

_, val_ds = load_dataset()



# ==========================
# Prediction
# ==========================

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



print("Prediction completed")



# ==========================
# Output Directory
# ==========================

output_dir = os.path.join(
    BASE_DIR,
    "outputs",
    "Evaluation"
)


os.makedirs(
    output_dir,
    exist_ok=True
)



# ==========================
# Metrics
# ==========================


results = calculate_metrics(
    y_true,
    y_pred
)


metrics_path = os.path.join(
    output_dir,
    "metrics.json"
)


save_metrics(
    results,
    metrics_path
)


print("\nEvaluation Results")

for key, value in results.items():

    print(
        f"{key}: {value:.4f}"
    )



# Classification Report

print_report(
    y_true,
    y_pred,
    CLASS_NAMES
)


report_path = os.path.join(
    output_dir,
    "classification_report.txt"
)


save_classification_report(
    y_true,
    y_pred,
    CLASS_NAMES,
    report_path
)

# ==========================
# Confusion Matrix
# ==========================

output_dir = os.path.join(
    BASE_DIR,
    "outputs",
    "Evaluation"
)


os.makedirs(
    output_dir,
    exist_ok=True
)



cm_path = os.path.join(
    output_dir,
    "confusion_matrix.png"
)



plot_confusion_matrix(
    y_true,
    y_pred,
    CLASS_NAMES,
    cm_path
)



print(
    "Confusion matrix saved:"
)

print(cm_path)