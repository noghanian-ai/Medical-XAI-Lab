# Medical-XAI-Lab

## Explainable AI for Chest X-ray Classification

This project presents an Explainable Artificial Intelligence (XAI) pipeline
for chest X-ray image classification using deep learning and visual
interpretability techniques.

The goal is not only to achieve accurate classification, but also to understand
which regions of the medical images contribute to the model decision.

---

## Project Objective

Develop an interpretable deep learning system for classification of chest
X-ray images into three categories:

- Normal
- COVID-19
- Lung Opacity

---

## Model

The classification model is based on:

**Architecture:**
VGG16 Transfer Learning

**Model file:**
models/vgg16_3class-3000-V-1.keras


**Input size:**

224 × 224 pixels

**Task:**

Three-class chest X-ray image classification

---

## Explainable AI Methods

To improve model transparency, two XAI techniques were implemented:

### 1. Grad-CAM

Gradient-weighted Class Activation Mapping was used to visualize the regions
of the image that influenced the CNN prediction.

Detailed analysis:

`docs/gradcam_analysis.md`

---

### 2. Score-CAM

Score-CAM was applied as a gradient-free explanation method based on activation
map importance and prediction confidence.

Detailed analysis:

`docs/scorecam_analysis.md`

---

## Results

The model was evaluated on representative chest X-ray samples.

Summary table:

`results/evaluation_table.md`

Example predictions:

| Image | True Class | Prediction | Confidence |
|---|---|---|---|
| Normal-1000.png | Normal | Normal | 100% |
| COVID-1000.png | COVID | COVID | 96.45% |
| Lung_Opacity-1000.png | Lung_Opacity | Lung_Opacity | 96.91% |

---

## Project Structure

Medical-XAI-Lab/

├── models/
│ └── vgg16_3class-3000-V-1.keras
│
├── images/
│
├── outputs/
│ ├── GradCAM/
│ └── ScoreCAM/
│
├── docs/
│ ├── gradcam_analysis.md
│ └── scorecam_analysis.md
│
├── results/
│ └── evaluation_table.md
│
├── gradcam.py
├── scorecam.py
├── run_gradcam.py
├── run_scorecam.py
└── config.py



---

## Explainable AI Visualization Examples

### Lung Opacity Example

Original X-ray:

![Original](outputs/GradCAM/Lung_Opacity/Lung_Opacity-1000/original.png)


Grad-CAM:

![Grad-CAM](outputs/GradCAM/Lung_Opacity/Lung_Opacity-1000/overlay.png)


Score-CAM:

![Score-CAM](outputs/ScoreCAM/Lung_Opacity/Lung_Opacity-1000/overlay.png)


### COVID Example

Original X-ray:

![Original](outputs/GradCAM/COVID/COVID-1000/original.png)


Grad-CAM:

![Grad-CAM](outputs/GradCAM/COVID/COVID-1000/overlay.png)


Score-CAM:

![Score-CAM](outputs/ScoreCAM/COVID/COVID-1000/overlay.png)


### Normal Example

Original X-ray:

![Original](outputs/GradCAM/Normal/Normal-1000/original.png)


Grad-CAM:

![Grad-CAM](outputs/GradCAM/Normal/Normal-1000/overlay.png)


Score-CAM:

![Score-CAM](outputs/ScoreCAM/Normal/Normal-1000/overlay.png)


---

## Technologies

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Deep Learning
- Computer Vision
- Explainable AI

---

## Future Work

Future improvements include:

- Vision Transformer (ViT) based models
- Swin Transformer architectures
- Medical image segmentation
- Uncertainty estimation
- Advanced XAI evaluation methods

---

## Author

Medical AI Research Project