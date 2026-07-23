# Model Evaluation and XAI Results

## Chest X-ray Classification Results

The trained VGG16 transfer learning model was evaluated on representative
chest X-ray images from three classes.

| Image | True Class | Prediction | Confidence | Grad-CAM | Score-CAM |
|---|---|---|---|---|---|
| Normal-1000.png | Normal | Normal | 100% | ✓ | ✓ |
| COVID-1000.png | COVID | COVID | 96.45% | ✓ | ✓ |
| Lung_Opacity-1000.png | Lung_Opacity | Lung_Opacity | 96.91% | ✓ | ✓ |

---

## Explainability Summary

Both Grad-CAM and Score-CAM were successfully applied to the classification
model.

The generated heatmaps provide visual explanations of the regions that
contributed to the model predictions.

---

## Model Information

**Architecture:** VGG16 Transfer Learning

**Classes:**

- Normal
- COVID-19
- Lung Opacity

**Input Resolution:**

224 × 224 pixels

**Model File:**
models/vgg16_3class-3000-V-1.keras