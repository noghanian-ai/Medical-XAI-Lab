import os
import cv2
import json


def save_original(output_folder, original):

    cv2.imwrite(
        os.path.join(
            output_folder,
            "original.png"
        ),
        original
    )


def save_heatmap(output_folder, heatmap):

    cv2.imwrite(
        os.path.join(
            output_folder,
            "heatmap.png"
        ),
        heatmap
    )


def save_overlay(output_folder, overlay):

    cv2.imwrite(
        os.path.join(
            output_folder,
            "overlay.png"
        ),
        overlay
    )


def save_prediction(output_folder, prediction, confidence):

    prediction_data = {
        "prediction": prediction,
        "confidence": float(confidence)
    }

    with open(
        os.path.join(
            output_folder,
            "prediction.json"
        ),
        "w"
    ) as f:

        json.dump(
            prediction_data,
            f,
            indent=4
        )