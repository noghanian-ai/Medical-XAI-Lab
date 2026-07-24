# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 11:56:55 2026

@author: manot
"""

import os

from config import TRAIN_DIR, VAL_DIR, CLASS_NAMES



def count_images(folder):

    count = 0

    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):
                count += 1

    return count



def dataset_summary(dataset_dir):

    print("\nDataset:")
    print(dataset_dir)

    total = 0

    for class_name in CLASS_NAMES:

        class_path = os.path.join(
            dataset_dir,
            class_name
        )

        num_images = count_images(
            class_path
        )

        total += num_images

        print(
            f"{class_name}: {num_images}"
        )

    print(
        f"Total: {total}"
    )



print("========== TRAIN ==========")

dataset_summary(
    TRAIN_DIR
)


print("\n========== VALIDATION ==========")

dataset_summary(
    VAL_DIR
)