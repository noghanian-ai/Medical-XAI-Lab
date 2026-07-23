
import os


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

IMG_SIZE = (224,224)


CLASS_NAMES = [
    "COVID",
    "Lung_Opacity",
    "Normal"
]


LAST_CONV_LAYER = "conv2d_12"