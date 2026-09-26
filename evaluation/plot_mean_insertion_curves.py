import json
import os

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INSERTION_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "XAI_Evaluation_Dataset",
    "Insertion_Test"
)

GRADCAM_PATH = os.path.join(
    INSERTION_DIR,
    "gradcam_results.json"
)

SCORECAM_PATH = os.path.join(
    INSERTION_DIR,
    "scorecam_results.json"
)

OUTPUT_DIR = os.path.join(
    INSERTION_DIR,
    "Mean_Insertion_Curves"
)

MASK_PERCENTAGES = np.array(
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    dtype=float
)


def load_results(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_statistics(results):
    confidence_matrix = np.array(
        [item["confidences"] for item in results],
        dtype=float
    )

    mean_confidence = np.mean(
        confidence_matrix,
        axis=0
    )

    std_confidence = np.std(
        confidence_matrix,
        axis=0,
        ddof=1
    )

    return (
        confidence_matrix,
        mean_confidence,
        std_confidence
    )


def calculate_class_statistics(results):
    classes = sorted(
        set(item["ground_truth"] for item in results)
    )

    statistics = {}

    for class_name in classes:
        class_results = [
            item
            for item in results
            if item["ground_truth"] == class_name
        ]

        confidence_matrix = np.array(
            [item["confidences"] for item in class_results],
            dtype=float
        )

        statistics[class_name] = {
            "n": len(class_results),
            "mean": np.mean(
                confidence_matrix,
                axis=0
            ),
            "std": np.std(
                confidence_matrix,
                axis=0,
                ddof=1
            )
        }

    return statistics


def plot_overall_curve(
    gradcam_mean,
    gradcam_std,
    scorecam_mean,
    scorecam_std
):
    plt.figure(figsize=(10, 6))

    plt.plot(
        MASK_PERCENTAGES,
        gradcam_mean,
        marker="o",
        label="Grad-CAM"
    )

    plt.fill_between(
        MASK_PERCENTAGES,
        np.maximum(
            gradcam_mean - gradcam_std,
            0
        ),
        np.minimum(
            gradcam_mean + gradcam_std,
            1
        ),
        alpha=0.15
    )

    plt.plot(
        MASK_PERCENTAGES,
        scorecam_mean,
        marker="o",
        label="Score-CAM"
    )

    plt.fill_between(
        MASK_PERCENTAGES,
        np.maximum(
            scorecam_mean - scorecam_std,
            0
        ),
        np.minimum(
            scorecam_mean + scorecam_std,
            1
        ),
        alpha=0.15
    )

    plt.xlabel("Masked pixels (%)")
    plt.ylabel("Target-class confidence")
    plt.title(
        "Mean Insertion Curves: Grad-CAM vs Score-CAM"
    )

    plt.ylim(0, 1.05)
    plt.xlim(0, 100)
    plt.xticks(MASK_PERCENTAGES)

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "overall_mean_insertion_curve.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


def plot_class_curves(
    gradcam_class_stats,
    scorecam_class_stats
):
    class_names = [
        "COVID",
        "Lung_Opacity",
        "Normal"
    ]

    for class_name in class_names:

        if (
            class_name not in gradcam_class_stats
            or class_name not in scorecam_class_stats
        ):
            continue

        gradcam = gradcam_class_stats[class_name]
        scorecam = scorecam_class_stats[class_name]

        plt.figure(figsize=(10, 6))

        plt.plot(
            MASK_PERCENTAGES,
            gradcam["mean"],
            marker="o",
            label="Grad-CAM"
        )

        plt.fill_between(
            MASK_PERCENTAGES,
            np.maximum(
                gradcam["mean"] - gradcam["std"],
                0
            ),
            np.minimum(
                gradcam["mean"] + gradcam["std"],
                1
            ),
            alpha=0.15
        )

        plt.plot(
            MASK_PERCENTAGES,
            scorecam["mean"],
            marker="o",
            label="Score-CAM"
        )

        plt.fill_between(
            MASK_PERCENTAGES,
            np.maximum(
                scorecam["mean"] - scorecam["std"],
                0
            ),
            np.minimum(
                scorecam["mean"] + scorecam["std"],
                1
            ),
            alpha=0.15
        )

        plt.xlabel("Masked pixels (%)")
        plt.ylabel("Target-class confidence")

        plt.title(
            f"Mean Insertion Curve — {class_name}"
        )

        plt.ylim(0, 1.05)
        plt.xlim(0, 100)
        plt.xticks(MASK_PERCENTAGES)

        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{class_name}_mean_insertion_curve.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


def save_statistics(
    gradcam_data,
    scorecam_data,
    gradcam_class_stats,
    scorecam_class_stats
):
    output = {
        "mask_percentages": MASK_PERCENTAGES.astype(int).tolist(),
        "overall": {
            "Grad-CAM": {
                "mean_confidence": (
                    gradcam_data[1].tolist()
                ),
                "std_confidence": (
                    gradcam_data[2].tolist()
                )
            },
            "Score-CAM": {
                "mean_confidence": (
                    scorecam_data[1].tolist()
                ),
                "std_confidence": (
                    scorecam_data[2].tolist()
                )
            }
        },
        "classes": {}
    }

    for class_name in [
        "COVID",
        "Lung_Opacity",
        "Normal"
    ]:

        output["classes"][class_name] = {
            "Grad-CAM": {
                "n": gradcam_class_stats[class_name]["n"],
                "mean_confidence": (
                    gradcam_class_stats[class_name]["mean"].tolist()
                ),
                "std_confidence": (
                    gradcam_class_stats[class_name]["std"].tolist()
                )
            },
            "Score-CAM": {
                "n": scorecam_class_stats[class_name]["n"],
                "mean_confidence": (
                    scorecam_class_stats[class_name]["mean"].tolist()
                ),
                "std_confidence": (
                    scorecam_class_stats[class_name]["std"].tolist()
                )
            }
        }

    output_path = os.path.join(
        OUTPUT_DIR,
        "mean_insertion_statistics.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            output,
            file,
            indent=2
        )

    return output_path


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("Loading insertion-test results...")

    gradcam_data_raw = load_results(
        GRADCAM_PATH
    )

    scorecam_data_raw = load_results(
        SCORECAM_PATH
    )

    print(
        f"Grad-CAM images: "
        f"{gradcam_data_raw['images_evaluated']}"
    )

    print(
        f"Score-CAM images: "
        f"{scorecam_data_raw['images_evaluated']}"
    )

    gradcam_stats = calculate_statistics(
        gradcam_data_raw["results"]
    )

    scorecam_stats = calculate_statistics(
        scorecam_data_raw["results"]
    )

    gradcam_class_stats = calculate_class_statistics(
        gradcam_data_raw["results"]
    )

    scorecam_class_stats = calculate_class_statistics(
        scorecam_data_raw["results"]
    )

    overall_plot = plot_overall_curve(
        gradcam_stats[1],
        gradcam_stats[2],
        scorecam_stats[1],
        scorecam_stats[2]
    )

    plot_class_curves(
        gradcam_class_stats,
        scorecam_class_stats
    )

    statistics_path = save_statistics(
        gradcam_stats,
        scorecam_stats,
        gradcam_class_stats,
        scorecam_class_stats
    )

    print()
    print("Mean insertion curve analysis completed.")
    print()
    print("Overall plot:")
    print(overall_plot)
    print()
    print("Statistics:")
    print(statistics_path)
    print()
    print("Class plots:")
    print(
        os.path.join(
            OUTPUT_DIR,
            "COVID_mean_insertion_curve.png"
        )
    )
    print(
        os.path.join(
            OUTPUT_DIR,
            "Lung_Opacity_mean_insertion_curve.png"
        )
    )
    print(
        os.path.join(
            OUTPUT_DIR,
            "Normal_mean_insertion_curve.png"
        )
    )


if __name__ == "__main__":
    main()
