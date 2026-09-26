# Explainability Analysis of Fine-Tuned ResNet50

## 1. Overview

This document reports the explainability analysis of the fine-tuned ResNet50 model developed for three-class chest X-ray image classification.

The classification task consists of three classes:

- COVID
- Lung_Opacity
- Normal

The explainability analysis uses two post-hoc visual explanation methods:

1. Grad-CAM
2. Score-CAM

Both methods were applied to the same fine-tuned ResNet50 model and the same chest X-ray image in order to provide a direct qualitative comparison.

---

## 2. Model

The evaluated model is an ImageNet-pretrained ResNet50 model followed by a global average pooling layer and a three-class softmax classification head.

The model was initially trained with the ResNet50 backbone frozen and subsequently fine-tuned by unfreezing the final portion of the backbone.

### Fine-tuned model

Model file:

`models/resnet50_3class_finetuned.keras`

### Input

- Image size: 224 × 224 pixels
- Number of channels: 3
- Number of classes: 3

### Target convolutional layer

For both Grad-CAM and Score-CAM, the final convolutional feature layer selected for explanation was:

`conv5_block3_3_conv`

The output shape of this layer is:

`7 × 7 × 2048`

This layer was selected because it provides high-level spatial feature representations while retaining spatial information that can be mapped back to the input image.

---

## 3. Classification Performance

The fine-tuned ResNet50 model was evaluated on the same 600-image validation set used for the previous model comparisons.

The overall performance was:

| Metric | Fine-Tuned ResNet50 |
|---|---:|
| Accuracy | 95.67% |
| Precision | 95.80% |
| Recall | 95.67% |
| F1-score | 95.66% |

The confusion matrix was:

    [[182, 16,  2],
     [  4, 196,  0],
     [  2,   2, 196]]

The corresponding class order is:

1. COVID
2. Lung_Opacity
3. Normal

The model therefore provides a strong classification baseline for subsequent explainability analysis.

---

## 4. Grad-CAM

### 4.1 Method

Gradient-weighted Class Activation Mapping (Grad-CAM) uses the gradients of the target class with respect to the feature maps of a selected convolutional layer.

The gradients are globally averaged across the spatial dimensions to obtain a weight for each feature map. The weighted feature maps are then combined to produce a class-discriminative localization map.

For this experiment, Grad-CAM was generated with respect to:

`conv5_block3_3_conv`

The resulting activation map was normalized and resized to the input image resolution for visualization.

### 4.2 Output

The Grad-CAM analysis generated:

- Original image
- Grad-CAM heatmap
- Grad-CAM overlay
- Prediction and confidence information

Output directory:

`outputs/GradCAM_ResNet50_FineTuned/`

---

## 5. Score-CAM

### 5.1 Method

Score-CAM is a gradient-free class activation mapping method.

Instead of using gradients to determine the importance of activation maps, Score-CAM uses the activation maps themselves to construct masked versions of the input image.

The masked images are passed through the classifier and the resulting class scores are used to determine the contribution of the corresponding activation maps.

For this experiment, Score-CAM used the same target layer:

`conv5_block3_3_conv`

The target layer produced 2048 activation maps. Invalid or non-informative activation maps were excluded before the scoring procedure.

Because Score-CAM can require substantial memory when many activation maps are processed simultaneously, the implementation uses batch-based processing.

The batch-based implementation successfully processed all valid activation maps without changing the underlying Score-CAM methodology.

### 5.2 Output

The Score-CAM analysis generated:

- Original image
- Score-CAM heatmap
- Score-CAM overlay
- Prediction and confidence information

Output directory:

`outputs/ScoreCAM_ResNet50_FineTuned/`

---

## 6. Prediction Used for Explainability

The same chest X-ray image was used for both Grad-CAM and Score-CAM.

Input image:

`images/Lung_Opacity/Lung_Opacity-1000.png`

The fine-tuned ResNet50 predicted:

`Class: Lung_Opacity`

`Confidence: 100.00%`

The Grad-CAM and Score-CAM explanations therefore correspond to the same model prediction and the same input image.

---

## 7. Grad-CAM vs Score-CAM

A direct visual comparison was generated using the same input image and the same target model layer.

The comparison contains:

1. Original image
2. Grad-CAM heatmap
3. Grad-CAM overlay
4. Score-CAM heatmap
5. Score-CAM overlay

Comparison output:

`outputs/XAI_Comparison_ResNet50_FineTuned/gradcam_vs_scorecam.png`

### Qualitative comparison

Grad-CAM and Score-CAM provide two different mechanisms for identifying image regions associated with the model prediction.

Grad-CAM is gradient-based and therefore depends on the sensitivity of the target class prediction to the feature maps of the selected convolutional layer.

Score-CAM is gradient-free and instead evaluates the contribution of activation maps through forward-pass class scores.

Using both methods provides a useful cross-check of the model's spatial attention.

If both methods highlight similar anatomical regions, this provides qualitative evidence that the observed explanation is not solely dependent on one explanation mechanism.

Conversely, substantial differences between the two maps may indicate that the model's decision is distributed across multiple feature representations or that the explanation is sensitive to the selected XAI method.

---

## 8. Interpretation

The purpose of these heatmaps is to investigate which image regions contribute to the model prediction.

The highlighted regions should not automatically be interpreted as clinically meaningful lesions or as evidence that the model has learned medically valid features.

In particular, a heatmap does not establish:

- the presence of a disease;
- the location of a clinically confirmed lesion;
- causal relationships between image features and disease;
- or clinical validity of the model.

Therefore, the Grad-CAM and Score-CAM visualizations are considered model-explanation evidence rather than clinical evidence.

A stronger medical interpretation would require comparison with expert annotations, lesion segmentation masks, radiologist assessments, or other clinically validated reference information.

---

## 9. Limitations

Several limitations should be considered.

### 9.1 Single-image qualitative analysis

The current Grad-CAM and Score-CAM comparison is based on a representative chest X-ray image.

A single image cannot establish the general reliability of an explanation method.

Future analysis should evaluate explanations across a larger and systematically selected subset of the validation dataset.

### 9.2 Spatial resolution

The selected ResNet50 feature layer has a spatial resolution of only:

`7 × 7`

Consequently, the resulting explanation maps are relatively coarse compared with the original 224 × 224 image.

### 9.3 Absence of expert localization ground truth

The current dataset does not provide expert lesion localization masks for the evaluated image.

Therefore, localization quality cannot currently be measured against a clinically validated spatial reference.

### 9.4 Confidence interpretation

The model produced a confidence of 100.00% for the analyzed image.

A high softmax confidence should not be interpreted as proof of diagnostic correctness or clinical certainty.

Softmax probabilities represent the model's output distribution and may be poorly calibrated.

### 9.5 Dataset limitations

The model was developed using a limited three-class dataset derived from chest X-ray images.

Dataset composition, image acquisition conditions, class definitions, and potential dataset-specific artifacts can affect both classification performance and explainability results.

---

## 10. Reproducibility

The explainability experiments were implemented as independent scripts so that the original baseline Grad-CAM implementation remains unchanged.

### Grad-CAM

`run_gradcam_resnet50_finetuned.py`

### Score-CAM

`run_scorecam_resnet50_finetuned.py`

### Comparison

`compare_gradcam_scorecam_resnet50.py`

### Model

`models/resnet50_3class_finetuned.keras`

### Target layer

`conv5_block3_3_conv`

### Input image

`images/Lung_Opacity/Lung_Opacity-1000.png`

---

## 11. Next Step: Quantitative Explainability Evaluation

The current analysis establishes the qualitative explainability pipeline.

The next stage is to introduce quantitative evaluation of the explanations.

Potential evaluation directions include:

- insertion and deletion metrics;
- localization metrics where suitable annotations are available;
- faithfulness evaluation;
- explanation consistency;
- explanation overlap between Grad-CAM and Score-CAM;
- confidence change after masking important regions;
- comparison across multiple images and all three classes.

Quantitative evaluation will provide stronger evidence than visual inspection alone and will make the XAI component of the project more suitable for research-oriented evaluation.

---

## 12. Summary

The fine-tuned ResNet50 model achieved 95.67% validation accuracy on the three-class chest X-ray classification task.

Both Grad-CAM and Score-CAM were successfully implemented using the final convolutional feature representation of the ResNet50 backbone.

The two methods were applied to the same image and model prediction, and a direct visual comparison was generated.

This establishes a reproducible qualitative explainability pipeline for the fine-tuned ResNet50 model.

The next research step is quantitative evaluation of explanation quality and consistency across a larger image subset.