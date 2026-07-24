
import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def calculate_metrics(y_true, y_pred):

    results = {}

    results["accuracy"] = accuracy_score(
        y_true,
        y_pred
    )

    results["precision"] = precision_score(
        y_true,
        y_pred,
        average="weighted"
    )

    results["recall"] = recall_score(
        y_true,
        y_pred,
        average="weighted"
    )

    results["f1_score"] = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    return results



def print_report(y_true, y_pred, class_names):

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names
        )
    )
    
    
    
def save_metrics(results, save_path):

    with open(save_path, "w") as f:
        json.dump(
            results,
            f,
            indent=4
        )    
        
        
        
def save_classification_report(
        y_true,
        y_pred,
        class_names,
        save_path):

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )


    with open(save_path, "w") as f:
        f.write(report)