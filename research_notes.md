# Research Notes — Checkpoint 1

## Paper 1
**Efficacy of Transfer Learning-based ResNet models in Chest X-ray image classification for detecting COVID-19 Pneumonia**

https://pmc.ncbi.nlm.nih.gov/articles/PMC8913041/

### Relevance
The paper evaluates ResNet-based transfer learning for chest X-ray classification and discusses augmentation, class imbalance and performance metrics.

### Project takeaway
ResNet transfer learning is a practical starting point for a relatively small image dataset. We therefore select ResNet-18 and plan staged fine-tuning.

---

## Paper 2
**Recognizing COVID-19 from chest X-ray images using deep transfer learning**

https://pmc.ncbi.nlm.nih.gov/articles/PMC8863907/

### Relevance
The work considers COVID-19, pneumonia and normal categories and uses deep transfer learning with visual explanation methods.

### Project takeaway
The three-class design matches the client requirement. COVID-19 sensitivity/false negatives should be monitored separately instead of relying only on overall accuracy.

---

## Paper 3
**Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization**

https://arxiv.org/abs/1610.02391

### Relevance
Grad-CAM provides class-discriminative localization maps from CNN gradients.

### Project takeaway
Use the final convolutional block of ResNet-18 for class-specific heatmaps. Heatmaps are interpretability aids and should not be treated as proof of pathology.

---

## GitHub implementation reference
https://github.com/cy730760-coder/chest-xray-pneumonia-classification

Used as an implementation reference for organizing a PyTorch chest X-ray project with ResNet, ROC-AUC and Grad-CAM.

## Design decisions
1. ResNet-18 because it is lightweight enough for a 3000-image internship project.
2. ImageNet initialization to benefit from pretrained visual features.
3. 224×224 input to match the standard ResNet-18 input pipeline.
4. Stratified splitting to preserve class proportions.
5. Macro-AUC plus per-class metrics to avoid hiding weak performance in a minority class.
6. Explicit COVID-19 FNR tracking because the acceptance criterion specifies <15%.
7. Grad-CAM on the final convolutional block for visual explanation.
8. Local Gradio deployment for the final demo.
