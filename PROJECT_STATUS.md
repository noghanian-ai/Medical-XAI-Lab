# Medical-XAI-Lab Status

Version:
2.0

Current model:
Custom VGG16-like CNN

Dataset:
COVID / Lung_Opacity / Normal

Completed:
- Prediction
- Grad-CAM
- Score-CAM

Next:
- Data loader
- Evaluation module
- ResNet50


Medical-XAI-Lab

Model loaded successfully

Prediction: Lung_Opacity

Confidence: 96.918%

Results saved:
outputs/GradCAM/...



Found 2410 files belonging to 3 classes.
Found 600 files belonging to 3 classes.

Classes:
['COVID', 'Lung_Opacity', 'Normal']



# Medical-XAI-Lab Status

Version:
2.1


## Current Model

Custom VGG16-like CNN


## Dataset

COVID / Lung_Opacity / Normal

Train:
2410 images

Validation:
600 images


## Completed

✅ Model loading

✅ Prediction pipeline

✅ Grad-CAM

✅ Score-CAM

✅ Data Loader


## Current Pipeline

Dataset
   |
Data Loader
   |
CNN Model
   |
Prediction
   |
XAI Visualization


## Next

1. Evaluation Module

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC


2. ResNet50 Transfer Learning

3. Model comparison

VGG16 vs ResNet50






# Medical-XAI-Lab Status

Version:
2.2


## Current Model

Custom VGG16-like CNN


## Dataset

COVID / Lung_Opacity / Normal


Training:
2410 images

Validation:
600 images


## Completed

✅ Prediction Pipeline

✅ Grad-CAM

✅ Score-CAM

✅ Data Loader

✅ Model Evaluation


## Evaluation Results

Accuracy:
93.00%

Precision:
93.52%

Recall:
93.00%

F1-score:
93.02%


## Outputs

- Classification Report
- Confusion Matrix


## Next

1. Save evaluation results as JSON

2. Improve visualization

3. ResNet50 Transfer Learning

4. Compare:
   VGG16 vs ResNet50

5. XAI comparison:
   Grad-CAM vs Score-CAM
   
   
   
   
   # Medical-XAI-Lab Status

Version:
2.3


## Dataset

3-Class Chest X-ray Dataset

Classes:
- COVID
- Lung_Opacity
- Normal


Training:
2400 images
(800 per class)


Validation:
600 images
(200 per class)



## Completed

✅ Model loading

✅ Prediction pipeline

✅ Grad-CAM

✅ Score-CAM

✅ Data Loader

✅ Evaluation Module


## Current Model

Custom VGG16-like CNN


## Evaluation Results

Validation Accuracy:
93.00%

Weighted Precision:
93.52%

Weighted Recall:
93.00%

Weighted F1-score:
93.02%


## Outputs

outputs/

├── GradCAM/
│
├── ScoreCAM/
│
└── Evaluation/
     └── confusion_matrix.png



## Next Steps

1. Save evaluation metrics
2. Save classification report
3. ResNet50 Transfer Learning
4. Model comparison
5. Grad-CAM vs Score-CAM analysis




# Medical-XAI-Lab Status

Version:
2.4


## Dataset

Chest X-ray 3-Class Dataset

Classes:
- COVID
- Lung_Opacity
- Normal


Training:
2400 images
(800 per class)


Validation:
600 images
(200 per class)



## Model

Custom VGG16-like CNN


## Completed Modules

✅ Data Loader

✅ Prediction Pipeline

✅ Grad-CAM

✅ Score-CAM

✅ Evaluation Module


## Evaluation Results

Accuracy:
93.00%

Weighted Precision:
93.52%

Weighted Recall:
93.00%

Weighted F1-score:
93.02%


## Saved Outputs

outputs/

├── GradCAM/
│
├── ScoreCAM/
│
└── Evaluation/
     ├── confusion_matrix.png
     ├── metrics.json
     └── classification_report.txt
     
     
     
     
     
     
     # Medical-XAI-Lab Status

Version:
2.5


## Dataset Validation

Automated dataset summary added.

Training:
2400 images

- COVID: 800
- Lung_Opacity: 800
- Normal: 800


Validation:
600 images

- COVID: 200
- Lung_Opacity: 200
- Normal: 200


## Completed Modules

✅ Dataset Summary

✅ Data Loader

✅ Prediction Pipeline

✅ Evaluation Module

✅ Grad-CAM

✅ Score-CAM


## Current Baseline Model

Custom VGG16-like CNN


## Baseline Performance

Accuracy:
93.00%

F1-score:
93.02%


## Outputs

outputs/

├── GradCAM/

├── ScoreCAM/

└── Evaluation/
    ├── confusion_matrix.png
    ├── metrics.json
    └── classification_report.txt
