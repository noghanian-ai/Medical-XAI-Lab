import os
import random
import json

DATASET_DIR = (
    r"C:\Users\manot\OneDrive\Desktop\Medical_Image_Classification"
    r"\Datasets\3-class-3000-covid-normal-Lung_Opacity-Separated"
)

VAL_DIR = os.path.join(DATASET_DIR, "validation")

CLASS_NAMES = [
    "COVID",
    "Lung_Opacity",
    "Normal"
]

IMAGES_PER_CLASS = 10
SEED = 42

OUTPUT_DIR = (
    "outputs/XAI_Evaluation_Dataset"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "selected_images.json"
)


def get_image_files(class_dir):
    valid_extensions = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp"
    )

    files = []

    for filename in os.listdir(class_dir):
        if filename.lower().endswith(valid_extensions):
            files.append(filename)

    return sorted(files)


def main():

    random.seed(SEED)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    selected_images = []

    print("=" * 60)
    print("XAI Evaluation Image Selection")
    print("=" * 60)

    for class_name in CLASS_NAMES:

        class_dir = os.path.join(
            VAL_DIR,
            class_name
        )

        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Validation class directory not found: "
                f"{class_dir}"
            )

        image_files = get_image_files(
            class_dir
        )

        if len(image_files) < IMAGES_PER_CLASS:
            raise ValueError(
                f"Not enough images in {class_name}. "
                f"Found {len(image_files)}."
            )

        selected = random.sample(
            image_files,
            IMAGES_PER_CLASS
        )

        selected = sorted(selected)

        print()
        print(
            f"{class_name}: "
            f"{len(selected)} images selected"
        )

        for filename in selected:

            relative_path = os.path.join(
                class_name,
                filename
            )

            selected_images.append(
                {
                    "class_name": class_name,
                    "filename": filename,
                    "relative_path": relative_path
                }
            )

            print(
                f"  {filename}"
            )

    output = {
        "seed": SEED,
        "images_per_class": IMAGES_PER_CLASS,
        "total_images": len(selected_images),
        "classes": CLASS_NAMES,
        "images": selected_images
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=4,
            ensure_ascii=False
        )

    print()
    print("=" * 60)
    print(
        f"Total selected: "
        f"{len(selected_images)}"
    )
    print()
    print("Saved:")
    print(OUTPUT_FILE)
    print("=" * 60)


if __name__ == "__main__":
    main()