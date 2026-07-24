import os


# =========================
# Dataset
# =========================

DATASET_DIR = r"C:\Users\manot\OneDrive\Desktop\Medical_Image_Classification\Datasets\3-class-3000-covid-normal-Lung_Opacity-Separated"


TRAIN_DIR = os.path.join(
    DATASET_DIR,
    "train"
)


VAL_DIR = os.path.join(
    DATASET_DIR,
    "validation"
)



# =========================
# Classes
# =========================

CLASS_NAMES = [
    "COVID",
    "Lung_Opacity",
    "Normal"
]



# =========================
# Image Settings
# =========================

IMG_SIZE = (224,224)

BATCH_SIZE = 32



# =========================
# Project Paths
# =========================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)



MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "vgg16_3class-3000-V-1.keras"
)



IMAGE_DIR = os.path.join(
    BASE_DIR,
    "images"
)



OUTPUT_DIR_GRADCAM = os.path.join(
    BASE_DIR,
    "outputs",
    "GradCAM"
)



OUTPUT_DIR_SCORECAM = os.path.join(
    BASE_DIR,
    "outputs",
    "ScoreCAM"
)



# =========================
# XAI
# =========================

LAST_CONV_LAYER = "conv2d_12"