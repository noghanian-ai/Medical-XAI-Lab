# Baseline Model Architecture

## Custom VGG16-like CNN

## Overview

The baseline model used in this project is a custom VGG16-like
Convolutional Neural Network (CNN) trained from scratch for three-class
chest X-ray classification.

This model is inspired by the VGG16 architecture design pattern but does
not use ImageNet pretrained weights or transfer learning.

---

## Task

Chest X-ray classification into three categories:

- COVID-19
- Lung Opacity
- Normal

---

## Input

Image size:

3 x 224 × 224 


---

## Architecture Summary

The model consists of five convolutional feature extraction blocks.

### Convolution Blocks

### Block 1


Conv2D (8 filters, 3×3)
Conv2D (8 filters, 3×3)
MaxPooling2D


Output:

8 x 112 × 112 


---

### Block 2

Conv2D (16 filters, 3×3)
Conv2D (16 filters, 3×3)
MaxPooling2D


Output:

16 x 56 × 56 


---

### Block 3

Conv2D (32 filters, 3×3)
Conv2D (32 filters, 3×3)
Conv2D (32 filters, 3×3)
MaxPooling2D


Output:

32 x 28 × 28 


---

### Block 4

Conv2D (64 filters, 3×3)
Conv2D (64 filters, 3×3)
Conv2D (64 filters, 3×3)
MaxPooling2D


Output:

64 x 14 × 14 


---

### Block 5

Conv2D (64 filters, 3×3)
Conv2D (64 filters, 3×3)
Conv2D (64 filters, 3×3)
MaxPooling2D


Output:

64 x 7 × 7 


---

## Classification Head

Flatten

Dense(4096, ReLU)

Dense(4096, ReLU)

Dense(3, Softmax)



---

## Model Parameters

Total parameters:

29,873,323


Trainable parameters:

29,873,323


Non-trainable parameters:

0


---

## Training Strategy

The model was trained from scratch using the chest X-ray dataset.

No pretrained ImageNet weights were used.

---

## Performance

Validation dataset:

600 images

COVID: 200
Lung Opacity: 200
Normal: 200


Results:

| Metric | Score |
|---|---:|
| Accuracy | 93.00% |
| Precision | 93.52% |
| Recall | 93.00% |
| F1-score | 93.02% |

---

## Explainability

The model was analyzed using:

- Grad-CAM
- Score-CAM

The last convolutional layer used for visualization:


conv2d_12


Feature map size:

64 x 7 x 7



---

## Role in Project

This model serves as the baseline architecture for comparison with
advanced transfer learning approaches such as:

- ResNet50
- DenseNet121
- Vision Transformer (ViT)
- Swin Transformer






