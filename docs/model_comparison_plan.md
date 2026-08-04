# Model Comparison Plan

## Objective

The objective of this experiment is to compare different deep learning
architectures for three-class chest X-ray classification and evaluate their
performance and explainability.

The three target classes are:

- COVID-19
- Lung Opacity
- Normal

---

## Baseline Model

## VGG16-based CNN

Current baseline model:

- Architecture: VGG16-based CNN
- Input size: 224 × 224 pixels
- Task: Three-class classification

Performance:

| Metric | Score |
|---|---:|
| Accuracy | 93.00% |
| Precision | 93.52% |
| Recall | 93.00% |
| F1-score | 93.02% |

Explainability methods:

- Grad-CAM
- Score-CAM

---

# Next Model

## ResNet50 Transfer Learning

Objective:

Evaluate whether residual learning improves classification performance
compared with the baseline model.

Configuration:

- Architecture: ResNet50
- Pre-trained weights: ImageNet
- Input size: 224 × 224 pixels
- Classification classes: 3

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

# Future Model Comparison

Planned architectures:

1. VGG16-based CNN
2. ResNet50 Transfer Learning
3. DenseNet121
4. Vision Transformer (ViT)
5. Swin Transformer

---

# Explainability Comparison

All models will be evaluated using:

- Grad-CAM
- Score-CAM

The comparison will focus on:

- Classification performance
- Model robustness
- Important image regions used for prediction
- Explainability quality

---

# Research Goal

The final goal is to develop an interpretable deep learning framework
for chest X-ray classification suitable for medical AI research.