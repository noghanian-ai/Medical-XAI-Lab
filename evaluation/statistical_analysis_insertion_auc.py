import json
import os

import numpy as np
from scipy import stats


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

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
    "Statistical_Analysis"
)

OUTPUT_JSON = os.path.join(
    OUTPUT_DIR,
    "insertion_auc_statistical_analysis.json"
)


# ============================================================
# Configuration
# ============================================================

CLASS_NAMES = [
    "COVID",
    "Lung_Opacity",
    "Normal"
]


# ============================================================
# Load results
# ============================================================

def load_results(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["results"]


# ============================================================
# Build paired dataset
# ============================================================

def build_paired_dataset(gradcam_results, scorecam_results):
    gradcam_by_filename = {
        item["filename"]: item
        for item in gradcam_results
    }

    scorecam_by_filename = {
        item["filename"]: item
        for item in scorecam_results
    }

    common_filenames = sorted(
        set(gradcam_by_filename)
        & set(scorecam_by_filename)
    )

    paired = []

    for filename in common_filenames:
        gradcam = gradcam_by_filename[filename]
        scorecam = scorecam_by_filename[filename]

        if gradcam["ground_truth"] != scorecam["ground_truth"]:
            raise ValueError(
                f"Ground-truth mismatch for {filename}"
            )

        if gradcam["predicted_class"] != scorecam["predicted_class"]:
            raise ValueError(
                f"Predicted-class mismatch for {filename}"
            )

        paired.append(
            {
                "filename": filename,
                "ground_truth": gradcam["ground_truth"],
                "predicted_class": gradcam["predicted_class"],
                "correct_prediction": gradcam[
                    "correct_prediction"
                ],
                "gradcam_auc": float(gradcam["auc"]),
                "scorecam_auc": float(scorecam["auc"]),
                "difference_scorecam_minus_gradcam": (
                    float(scorecam["auc"])
                    - float(gradcam["auc"])
                ),
            }
        )

    return paired


# ============================================================
# Bootstrap confidence interval
# ============================================================

def bootstrap_mean_difference_ci(
    differences,
    n_bootstrap=10000,
    confidence_level=0.95,
    seed=42
):
    differences = np.asarray(
        differences,
        dtype=float
    )

    rng = np.random.default_rng(seed)

    n = len(differences)

    bootstrap_means = np.empty(
        n_bootstrap,
        dtype=float
    )

    for i in range(n_bootstrap):
        sample = rng.choice(
            differences,
            size=n,
            replace=True
        )

        bootstrap_means[i] = np.mean(sample)

    alpha = 1.0 - confidence_level

    lower = np.percentile(
        bootstrap_means,
        100.0 * (alpha / 2.0)
    )

    upper = np.percentile(
        bootstrap_means,
        100.0 * (1.0 - alpha / 2.0)
    )

    return float(lower), float(upper)


# ============================================================
# Paired analysis
# ============================================================

def analyze_paired_auc(paired_results):
    gradcam_auc = np.array(
        [
            item["gradcam_auc"]
            for item in paired_results
        ],
        dtype=float
    )

    scorecam_auc = np.array(
        [
            item["scorecam_auc"]
            for item in paired_results
        ],
        dtype=float
    )

    differences = (
        scorecam_auc
        - gradcam_auc
    )

    mean_gradcam = float(
        np.mean(gradcam_auc)
    )

    mean_scorecam = float(
        np.mean(scorecam_auc)
    )

    median_gradcam = float(
        np.median(gradcam_auc)
    )

    median_scorecam = float(
        np.median(scorecam_auc)
    )

    mean_difference = float(
        np.mean(differences)
    )

    median_difference = float(
        np.median(differences)
    )

    std_difference = float(
        np.std(
            differences,
            ddof=1
        )
    )

    bootstrap_ci = (
        bootstrap_mean_difference_ci(
            differences
        )
    )

    # Paired t-test
    t_statistic, t_pvalue = (
        stats.ttest_rel(
            scorecam_auc,
            gradcam_auc
        )
    )

    # Wilcoxon signed-rank test
    #
    # zero_method="wilcox" excludes exact zero
    # differences, which is appropriate for the
    # signed-rank calculation.
    try:
        wilcoxon_statistic, wilcoxon_pvalue = (
            stats.wilcoxon(
                scorecam_auc,
                gradcam_auc,
                zero_method="wilcox",
                alternative="two-sided",
                method="auto"
            )
        )

        wilcoxon_result = {
            "statistic": float(
                wilcoxon_statistic
            ),
            "p_value": float(
                wilcoxon_pvalue
            ),
        }

    except ValueError as error:
        wilcoxon_result = {
            "statistic": None,
            "p_value": None,
            "error": str(error),
        }

    # Cohen's dz for paired samples
    if std_difference > 0:
        cohens_dz = (
            mean_difference
            / std_difference
        )
    else:
        cohens_dz = 0.0

    # Direction counts
    scorecam_lower = int(
        np.sum(differences < 0)
    )

    equal = int(
        np.sum(differences == 0)
    )

    scorecam_higher = int(
        np.sum(differences > 0)
    )

    return {
        "n": int(len(paired_results)),
        "mean_auc": {
            "Grad-CAM": mean_gradcam,
            "Score-CAM": mean_scorecam,
        },
        "median_auc": {
            "Grad-CAM": median_gradcam,
            "Score-CAM": median_scorecam,
        },
        "paired_difference": {
            "definition": "Score-CAM minus Grad-CAM",
            "mean": mean_difference,
            "median": median_difference,
            "std": std_difference,
            "bootstrap_95_percent_ci": {
                "lower": bootstrap_ci[0],
                "upper": bootstrap_ci[1],
            },
        },
        "paired_t_test": {
            "t_statistic": float(
                t_statistic
            ),
            "p_value": float(
                t_pvalue
            ),
        },
        "wilcoxon_signed_rank_test": wilcoxon_result,
        "effect_size": {
            "name": "Cohen's dz",
            "value": float(
                cohens_dz
            ),
        },
        "direction_counts": {
            "Score-CAM_higher_AUC": scorecam_higher,
            "equal": equal,
            "Score-CAM_lower_AUC": scorecam_lower,
        },
    }


# ============================================================
# Class-specific analysis
# ============================================================

def analyze_by_class(paired_results):
    class_results = {}

    for class_name in CLASS_NAMES:
        subset = [
            item
            for item in paired_results
            if item["ground_truth"] == class_name
        ]

        if not subset:
            continue

        class_results[class_name] = (
            analyze_paired_auc(subset)
        )

    return class_results


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Statistical analysis of XAI insertion-test AUC")
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("\nLoading results...")

    gradcam_results = load_results(
        GRADCAM_PATH
    )

    scorecam_results = load_results(
        SCORECAM_PATH
    )

    print(
        f"Grad-CAM results: {len(gradcam_results)}"
    )

    print(
        f"Score-CAM results: {len(scorecam_results)}"
    )

    paired_results = build_paired_dataset(
        gradcam_results,
        scorecam_results
    )

    print(
        f"Paired images: {len(paired_results)}"
    )

    # --------------------------------------------------------
    # All images
    # --------------------------------------------------------

    all_results = analyze_paired_auc(
        paired_results
    )

    # --------------------------------------------------------
    # Correctly classified images
    # --------------------------------------------------------

    correct_results = [
        item
        for item in paired_results
        if item["correct_prediction"]
    ]

    incorrect_results = [
        item
        for item in paired_results
        if not item["correct_prediction"]
    ]

    correct_analysis = analyze_paired_auc(
        correct_results
    )

    # --------------------------------------------------------
    # Correctly classified results by class
    # --------------------------------------------------------

    correct_by_class = analyze_by_class(
        correct_results
    )

    # --------------------------------------------------------
    # All-image results by class
    # --------------------------------------------------------

    all_by_class = analyze_by_class(
        paired_results
    )

    # --------------------------------------------------------
    # Incorrect predictions
    # --------------------------------------------------------

    incorrect_details = []

    for item in incorrect_results:
        incorrect_details.append(
            {
                "filename": item["filename"],
                "ground_truth": item[
                    "ground_truth"
                ],
                "predicted_class": item[
                    "predicted_class"
                ],
                "gradcam_auc": item[
                    "gradcam_auc"
                ],
                "scorecam_auc": item[
                    "scorecam_auc"
                ],
                "difference_scorecam_minus_gradcam": item[
                    "difference_scorecam_minus_gradcam"
                ],
            }
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    report = {
        "analysis": {
            "description": (
                "Paired comparison of Grad-CAM and "
                "Score-CAM insertion-test AUC values."
            ),
            "auc_interpretation": (
                "Higher insertion AUC indicates faster "
                "confidence reduction under the specified "
                "masking procedure. This metric alone does "
                "not establish superiority of one "
                "explanation method."
            ),
            "difference_definition": (
                "Score-CAM minus Grad-CAM"
            ),
            "bootstrap_iterations": 10000,
            "bootstrap_seed": 42,
            "confidence_level": 0.95,
        },
        "dataset": {
            "total_paired_images": len(
                paired_results
            ),
            "correctly_classified_images": len(
                correct_results
            ),
            "incorrectly_classified_images": len(
                incorrect_results
            ),
        },
        "all_30_images": all_results,
        "correctly_classified_images": (
            correct_analysis
        ),
        "all_images_by_class": all_by_class,
        "correctly_classified_images_by_class": (
            correct_by_class
        ),
        "incorrect_predictions": incorrect_details,
        "paired_results": paired_results,
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Dataset summary")
    print("=" * 70)

    print(
        f"Total paired images: "
        f"{len(paired_results)}"
    )

    print(
        f"Correctly classified: "
        f"{len(correct_results)}"
    )

    print(
        f"Incorrectly classified: "
        f"{len(incorrect_results)}"
    )

    print("\n" + "=" * 70)
    print("All 30 images")
    print("=" * 70)

    print(
        f"Mean Grad-CAM AUC: "
        f"{all_results['mean_auc']['Grad-CAM']:.4f}"
    )

    print(
        f"Mean Score-CAM AUC: "
        f"{all_results['mean_auc']['Score-CAM']:.4f}"
    )

    print(
        f"Mean difference "
        f"(Score-CAM - Grad-CAM): "
        f"{all_results['paired_difference']['mean']:.4f}"
    )

    print(
        "Bootstrap 95% CI: "
        f"["
        f"{all_results['paired_difference']['bootstrap_95_percent_ci']['lower']:.4f}, "
        f"{all_results['paired_difference']['bootstrap_95_percent_ci']['upper']:.4f}"
        f"]"
    )

    print(
        f"Paired t-test p-value: "
        f"{all_results['paired_t_test']['p_value']:.6f}"
    )

    print(
        f"Wilcoxon p-value: "
        f"{all_results['wilcoxon_signed_rank_test']['p_value']}"
    )

    print(
        f"Cohen's dz: "
        f"{all_results['effect_size']['value']:.4f}"
    )

    print("\n" + "=" * 70)
    print("Correctly classified images (n=29)")
    print("=" * 70)

    print(
        f"Mean Grad-CAM AUC: "
        f"{correct_analysis['mean_auc']['Grad-CAM']:.4f}"
    )

    print(
        f"Mean Score-CAM AUC: "
        f"{correct_analysis['mean_auc']['Score-CAM']:.4f}"
    )

    print(
        f"Mean difference "
        f"(Score-CAM - Grad-CAM): "
        f"{correct_analysis['paired_difference']['mean']:.4f}"
    )

    print(
        "Bootstrap 95% CI: "
        f"["
        f"{correct_analysis['paired_difference']['bootstrap_95_percent_ci']['lower']:.4f}, "
        f"{correct_analysis['paired_difference']['bootstrap_95_percent_ci']['upper']:.4f}"
        f"]"
    )

    print(
        f"Paired t-test p-value: "
        f"{correct_analysis['paired_t_test']['p_value']:.6f}"
    )

    print(
        f"Wilcoxon p-value: "
        f"{correct_analysis['wilcoxon_signed_rank_test']['p_value']}"
    )

    print(
        f"Cohen's dz: "
        f"{correct_analysis['effect_size']['value']:.4f}"
    )

    print("\n" + "=" * 70)
    print("Class-specific analysis")
    print("=" * 70)

    for class_name, result in (
        correct_by_class.items()
    ):
        print(f"\n{class_name} (n={result['n']})")

        print(
            f"  Grad-CAM mean AUC: "
            f"{result['mean_auc']['Grad-CAM']:.4f}"
        )

        print(
            f"  Score-CAM mean AUC: "
            f"{result['mean_auc']['Score-CAM']:.4f}"
        )

        print(
            f"  Mean difference: "
            f"{result['paired_difference']['mean']:.4f}"
        )

        print(
            "  Bootstrap 95% CI: "
            f"["
            f"{result['paired_difference']['bootstrap_95_percent_ci']['lower']:.4f}, "
            f"{result['paired_difference']['bootstrap_95_percent_ci']['upper']:.4f}"
            f"]"
        )

        print(
            f"  Wilcoxon p-value: "
            f"{result['wilcoxon_signed_rank_test']['p_value']}"
        )

        print(
            f"  Cohen's dz: "
            f"{result['effect_size']['value']:.4f}"
        )

    print("\n" + "=" * 70)
    print("Output")
    print("=" * 70)

    print(OUTPUT_JSON)
    print("=" * 70)


if __name__ == "__main__":
    main()