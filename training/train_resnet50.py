import os
import tensorflow as tf

from training.data_loader import load_dataset
from config import IMG_SIZE, CLASS_NAMES, BASE_DIR


# =========================
# Settings
# =========================

NUM_CLASSES = len(CLASS_NAMES)

MODEL_OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "models",
    "resnet50_3class.keras"
)

EPOCHS = 10


# =========================
# Load Dataset
# =========================

train_ds, val_ds = load_dataset()


# =========================
# ResNet50 Preprocessing
# =========================

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


# =========================
# Prepare Dataset
# =========================

train_ds = train_ds.prefetch(
    tf.data.AUTOTUNE
)

val_ds = val_ds.prefetch(
    tf.data.AUTOTUNE
)


# =========================
# ResNet50 Base Model
# =========================

base_model = tf.keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)

base_model.trainable = False


# =========================
# Classification Head
# =========================

inputs = tf.keras.Input(
    shape=(*IMG_SIZE, 3)
)

x = base_model(
    inputs,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs
)


# =========================
# Compile
# =========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# =========================
# Model Summary
# =========================

model.summary()


# =========================
# Callbacks
# =========================

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


# =========================
# Training
# =========================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# =========================
# Final Message
# =========================

print()
print("Training completed.")
print("Model saved to:")
print(MODEL_OUTPUT_PATH)