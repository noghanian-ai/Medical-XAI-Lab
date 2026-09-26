import json
import os

import numpy as np


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "XAI_Evaluation_Dataset",
    "Deletion_Test",
)

GRADCAM_FILE = os.path.join(
    RESULTS_DIR,
    "gradcam_results.json",
)

SCORECAM_FILE = os.path.join(
    RESULTS_DIR,
    "scorecam_results.json",
)

OUTPUT_DIR = os.path.join(
    RESULTS_DIR,
    "Statistical_Analysis",
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "paired_auc_difference_analysis.json",
)


# ============================================================
# Load results
# ============================================================

with open(GRADCAM_FILE, "r", encoding="utf-8") as f:
    gradcam_data = json.load(f)

with open(SCORECAM_FILE, "r", encoding="utf-8") as f:
    scorecam_data = json.load(f)


# ============================================================
# Extract records
# ============================================================

def extract_records(data):
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["results", "images", "records", "data"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError("Unable to find result records in JSON.")


gradcam_records = extract_records(gradcam_data)
scorecam_records = extract_records(scorecam_data)


# ============================================================
# Field extraction
# ============================================================

def get_filename(record):
    for key in ["filename", "image", "image_name", "file"]:
        if key in record:
            return record[key]

    raise KeyError("Filename field not found.")


def get_auc(record):
    for key in ["auc", "AUC", "deletion_auc"]:
        if key in record:
            return float(record[key])

    raise KeyError("AUC field not found.")


def get_correct(record):
    return record.get("correct_prediction")


def get_class(filename):
    if filename.startswith("COVID"):
        return "COVID"

    if filename.startswith("Lung_Opacity"):
        return "Lung_Opacity"

    if filename.startswith("Normal"):
        return "Normal"

    return "Unknown"


# ============================================================
# Build lookup
# ============================================================

gradcam_by_file = {
    get_filename(record): record
    for record in gradcam_records
}

scorecam_by_file = {
    get_filename(record): record
    for record in scorecam_records
}


common_files = sorted(
    set(gradcam_by_file.keys()) &
    set(scorecam_by_file.keys())
)


if not common_files:
    raise RuntimeError("No matching image files found.")


# ============================================================
# Build paired dataset
# ============================================================

records = []

for filename in common_files:

    gradcam_record = gradcam_by_file[filename]
    scorecam_record = scorecam_by_file[filename]

    gradcam_auc = get_auc(gradcam_record)
    scorecam_auc = get_auc(scorecam_record)

    difference = scorecam_auc - gradcam_auc

    records.append(
        {
            "filename": filename,
            "class": get_class(filename),
            "correct": get_correct(gradcam_record),
            "gradcam_auc": gradcam_auc,
            "scorecam_auc": scorecam_auc,
            "difference_scorecam_minus_gradcam": difference,
            "absolute_difference": abs(difference),
        }
    )


# ============================================================
# Sort by absolute difference
# ============================================================

largest_positive = sorted(
    records,
    key=lambda x: x["difference_scorecam_minus_gradcam"],
    reverse=True,
)

largest_negative = sorted(
    records,
    key=lambda x: x["difference_scorecam_minus_gradcam"],
)

largest_absolute = sorted(
    records,
    key=lambda x: x["absolute_difference"],
    reverse=True,
)


# ============================================================
# Overall statistics
# ============================================================

differences = np.array(
    [
        r["difference_scorecam_minus_gradcam"]
        for r in records
    ]
)

gradcam_auc = np.array(
    [
        r["gradcam_auc"]
        for r in records
    ]
)

scorecam_auc = np.array(
    [
        r["scorecam_auc"]
        for r in records
    ]
)


def summarize(values):
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


overall_summary = {
    "gradcam_auc": summarize(gradcam_auc),
    "scorecam_auc": summarize(scorecam_auc),
    "difference_scorecam_minus_gradcam": summarize(differences),
    "scorecam_higher_count": int(np.sum(differences > 0)),
    "gradcam_higher_count": int(np.sum(differences < 0)),
    "equal_count": int(np.sum(differences == 0)),
}


# ============================================================
# Class-specific analysis
# ============================================================

class_summaries = {}

for class_name in ["COVID", "Lung_Opacity", "Normal"]:

    class_records = [
        r
        for r in records
        if r["class"] == class_name
    ]

    class_differences = np.array(
        [
            r["difference_scorecam_minus_gradcam"]
            for r in class_records
        ]
    )

    class_gradcam = np.array(
        [
            r["gradcam_auc"]
            for r in class_records
        ]
    )

    class_scorecam = np.array(
        [
            r["scorecam_auc"]
            for r in class_records
        ]
    )

    class_summaries[class_name] = {
        "n": len(class_records),
        "gradcam_auc": summarize(class_gradcam),
        "scorecam_auc": summarize(class_scorecam),
        "difference_scorecam_minus_gradcam": summarize(
            class_differences
        ),
        "scorecam_higher_count": int(
            np.sum(class_differences > 0)
        ),
        "gradcam_higher_count": int(
            np.sum(class_differences < 0)
        ),
        "equal_count": int(
            np.sum(class_differences == 0)
        ),
    }


# ============================================================
# Correctly classified subset
# ============================================================

correct_records = [
    r
    for r in records
    if r["correct"] is True
]

correct_differences = np.array(
    [
        r["difference_scorecam_minus_gradcam"]
        for r in correct_records
    ]
)

correct_gradcam = np.array(
    [
        r["gradcam_auc"]
        for r in correct_records
    ]
)

correct_scorecam = np.array(
    [
        r["scorecam_auc"]
        for r in correct_records
    ]
)

correct_summary = {
    "n": len(correct_records),
    "gradcam_auc": summarize(correct_gradcam),
    "scorecam_auc": summarize(correct_scorecam),
    "difference_scorecam_minus_gradcam": summarize(
        correct_differences
    ),
    "scorecam_higher_count": int(
        np.sum(correct_differences > 0)
    ),
    "gradcam_higher_count": int(
        np.sum(correct_differences < 0)
    ),
    "equal_count": int(
        np.sum(correct_differences == 0)
    ),
}


# ============================================================
# Top differences
# ============================================================

def compact_record(record):
    return {
        "filename": record["filename"],
        "class": record["class"],
        "correct": record["correct"],
        "gradcam_auc": round(record["gradcam_auc"], 6),
        "scorecam_auc": round(record["scorecam_auc"], 6),
        "difference_scorecam_minus_gradcam": round(
            record["difference_scorecam_minus_gradcam"],
            6,
        ),
    }


top_positive = [
    compact_record(r)
    for r in largest_positive[:5]
]

top_negative = [
    compact_record(r)
    for r in largest_negative[:5]
]

top_absolute = [
    compact_record(r)
    for r in largest_absolute[:10]
]


# ============================================================
# Save JSON report
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

report = {
    "description": (
        "Paired comparison of deletion-test AUC values "
        "between Grad-CAM and Score-CAM."
    ),
    "difference_definition": (
        "Score-CAM AUC minus Grad-CAM AUC. "
        "Positive values indicate higher Score-CAM AUC; "
        "negative values indicate higher Grad-CAM AUC."
    ),
    "overall": overall_summary,
    "correctly_classified_only": correct_summary,
    "class_specific": class_summaries,
    "largest_positive_differences": top_positive,
    "largest_negative_differences": top_negative,
    "largest_absolute_differences": top_absolute,
    "all_pairs": [
        compact_record(r)
        for r in records
    ],
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        report,
        f,
        indent=2,
    )


# ============================================================
# Console report
# ============================================================

print("=" * 70)
print("Paired AUC Difference Analysis")
print("=" * 70)

print(f"Images analyzed: {len(records)}")
print()

print("OVERALL")
print("-" * 70)
print(
    f"Mean difference:   "
    f"{np.mean(differences):.4f}"
)
print(
    f"Median difference: "
    f"{np.median(differences):.4f}"
)
print(
    f"SD difference:     "
    f"{np.std(differences, ddof=1):.4f}"
)
print(
    f"Score-CAM higher:  "
    f"{np.sum(differences > 0)}"
)
print(
    f"Grad-CAM higher:   "
    f"{np.sum(differences < 0)}"
)
print(
    f"Equal:             "
    f"{np.sum(differences == 0)}"
)

print()
print("CLASS-SPECIFIC")
print("-" * 70)

for class_name, summary in class_summaries.items():

    diff_summary = summary[
        "difference_scorecam_minus_gradcam"
    ]

    print(
        f"{class_name}: "
        f"mean={diff_summary['mean']:.4f}, "
        f"median={diff_summary['median']:.4f}, "
        f"Score-CAM higher={summary['scorecam_higher_count']}, "
        f"Grad-CAM higher={summary['gradcam_higher_count']}"
    )


print()
print("TOP 5 POSITIVE DIFFERENCES")
print("-" * 70)

for record in top_positive:
    print(
        f"{record['filename']}: "
        f"{record['difference_scorecam_minus_gradcam']:+.4f}"
    )


print()
print("TOP 5 NEGATIVE DIFFERENCES")
print("-" * 70)

for record in top_negative:
    print(
        f"{record['filename']}: "
        f"{record['difference_scorecam_minus_gradcam']:+.4f}"
    )


print()
print("TOP 10 ABSOLUTE DIFFERENCES")
print("-" * 70)

for record in top_absolute:
    print(
        f"{record['filename']}: "
        f"{record['difference_scorecam_minus_gradcam']:+.4f}"
    )


print()
print("=" * 70)
print("Analysis saved successfully.")
print(f"Output: {OUTPUT_FILE}")
print("=" * 70)