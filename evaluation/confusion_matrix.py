import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


def plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        save_path):

    cm = confusion_matrix(
        y_true,
        y_pred
    )


    plt.figure(figsize=(6,6))

    plt.imshow(cm)

    plt.title(
        "Confusion Matrix"
    )

    plt.colorbar()


    plt.xticks(
        np.arange(len(class_names)),
        class_names,
        rotation=45
    )

    plt.yticks(
        np.arange(len(class_names)),
        class_names
    )


    for i in range(len(class_names)):
        for j in range(len(class_names)):

            plt.text(
                j,
                i,
                cm[i,j],
                ha="center",
                va="center"
            )


    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "True"
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.close()