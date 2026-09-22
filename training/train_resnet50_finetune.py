import os

import tensorflow as tf

from training.data_loader import load_dataset
from config import BASE_DIR


MODEL_INPUT_PATH = os.path.join(
    BASE_DIR,
    "models",
    "resnet50_3class.keras"
)

MODEL_OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "models",
    "resnet50_3class_finetuned.keras"
)

EPOCHS = 10
FINE_TUNE_LAYERS = 30
LEARNING_RATE = 1e-5


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

train_ds, val_ds = load_dataset()


def preprocess_resnet50(images, labels):
    images = tf.keras.applications.resnet50.preprocess_input(
        tf.cast(images, tf.float32)
    )
    return images, labels


train_ds = train_ds.map(
    preprocess_resnet50,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    preprocess_resnet50,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------
# Load pretrained ResNet50 transfer-learning model
# ---------------------------------------------------------

print()
print("Loading ResNet50 transfer learning model...")
print(MODEL_INPUT_PATH)

model = tf.keras.models.load_model(
    MODEL_INPUT_PATH
)


# ---------------------------------------------------------
# Locate ResNet50 backbone
# ---------------------------------------------------------

base_model = None

for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        if "resnet50" in layer.name.lower():
            base_model = layer
            break


if base_model is None:
    raise RuntimeError(
        "Could not locate the ResNet50 backbone in the loaded model."
    )


# ---------------------------------------------------------
# Fine-tuning configuration
# ---------------------------------------------------------

base_model.trainable = True


# Freeze all ResNet50 layers first
for layer in base_model.layers:
    layer.trainable = False


# Unfreeze only the last N layers.
# Keep BatchNormalization layers frozen.
for layer in base_model.layers[-FINE_TUNE_LAYERS:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True


# ---------------------------------------------------------
# Compile model
# ---------------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ---------------------------------------------------------
# Print configuration
# ---------------------------------------------------------

print()
print("Fine-tuning configuration:")
print(
    "Trainable ResNet50 layers:",
    sum(layer.trainable for layer in base_model.layers)
)
print("Requested last layers:", FINE_TUNE_LAYERS)
print("Learning rate:", LEARNING_RATE)
print("Epochs:", EPOCHS)
print()

model.summary()


# ---------------------------------------------------------
# Callbacks
# ---------------------------------------------------------

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_OUTPUT_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        mode="max",
        restore_best_weights=True,
        verbose=1
    )
]


# ---------------------------------------------------------
# Train
# ---------------------------------------------------------

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ---------------------------------------------------------
# Finished
# ---------------------------------------------------------

print()
print("Fine-tuning completed.")
print()
print("Model saved to:")
print(MODEL_OUTPUT_PATH)