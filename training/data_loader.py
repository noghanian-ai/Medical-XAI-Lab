

import tensorflow as tf

from config import (
    TRAIN_DIR,
    VAL_DIR,
    IMG_SIZE,
    BATCH_SIZE
)


def load_dataset():


    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )


    val_dataset = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    return train_dataset, val_dataset



def prepare_dataset(dataset):

    AUTOTUNE = tf.data.AUTOTUNE

    dataset = dataset.prefetch(
        buffer_size=AUTOTUNE
    )

    return dataset