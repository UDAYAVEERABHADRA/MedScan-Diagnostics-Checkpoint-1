Reviewed three relevant research papers/resources covering ResNet transfer learning for chest X-ray classification, three-class COVID-19/pneumonia/normal classification, and Grad-CAM explainability. Based on this research, I designed a PyTorch transfer-learning pipeline using an ImageNet-pretrained ResNet-18 with a three-class output head for NORMAL, PNEUMONIA and COVID-19.

The proposed pipeline includes image resizing to 224×224, RGB conversion, ImageNet normalization, training-only augmentation, stratified train/validation/test splitting and AdamW optimization. I defined macro and per-class ROC-AUC, accuracy, precision, recall/sensitivity, specificity, F1-score and confusion matrix as evaluation metrics. COVID-19 false-negative rate will be tracked separately because the project acceptance criterion requires it to remain below 15%.

I also designed the Grad-CAM component using the final convolutional block of ResNet-18 and prepared a local Gradio demo structure that will return the predicted class, class probabilities and heatmap. The repository contains the README, research notes, model-card template, requirements file, data pipeline skeleton and training/evaluation/Grad-CAM/Gradio code structure.

The two provided project data packs have also been included in the repository as supporting materials. The chest X-ray training images will be added to the designated image-data directory when available. Final AUC and COVID-19 false-negative-rate results will be reported after model training and testing; they are not being claimed as achieved at this checkpoint.

## Live URL / Repository Link

PASTE YOUR GITHUB REPOSITORY URL HERE
