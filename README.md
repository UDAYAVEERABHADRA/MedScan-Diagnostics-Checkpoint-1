# MedScan Diagnostics — Chest X-Ray Classification

## Checkpoint 1: Research + Architecture Design

**Client:** MedScan Diagnostics  
**Industry:** HealthTech  
**Project:** AI/ML Intern — 3-month project  
**Task:** Chest X-ray classification using CNNs and transfer learning (ResNet-18)

### Objective
Develop a 3-class chest X-ray classifier for:
1. NORMAL
2. PNEUMONIA
3. COVID-19

The final system is planned to use ImageNet-pretrained ResNet-18, Grad-CAM explainability, and a local Gradio interface.

### Checkpoint 1 deliverables completed
- Research review of 3 relevant papers/resources
- Model architecture definition
- Evaluation metrics definition
- Repository structure
- Data pipeline skeleton
- Training/evaluation/Grad-CAM/Gradio code skeleton
- Model card with limitations
- Provided client data packs included in `data/`

> **Important:** The 3000 chest X-ray images are not included in this package because the uploaded data packs contain supporting healthcare test/symptom data, not the X-ray image dataset described in the project brief. Do not upload medical images to a public GitHub repository unless the client explicitly permits it.

### Architecture
Input X-ray → Resize 224×224 → RGB conversion → ImageNet normalization → ResNet-18 pretrained backbone → 3-class fully connected head → class probabilities.

### Training plan
- Transfer learning from ImageNet-pretrained ResNet-18
- Initial freezing of most pretrained layers
- Fine-tuning of the final residual block
- AdamW optimizer
- Cross-entropy loss, with class weighting if imbalance is confirmed
- Training-only augmentation
- Stratified train/validation/test split
- Early stopping/checkpointing based on validation performance

### Explainability
Grad-CAM will target the final convolutional block (`layer4[-1]`) of ResNet-18. The final application will return the predicted class, probabilities for all three classes, and a Grad-CAM heatmap.

### Evaluation metrics
Primary:
- Macro ROC-AUC
- Per-class ROC-AUC
- COVID-19 false-negative rate

Secondary:
- Accuracy
- Precision
- Recall/sensitivity
- Specificity
- F1-score
- Confusion matrix

Acceptance targets:
- Test AUC ≥ 0.87
- Grad-CAM renders for all 3 classes
- Gradio runs locally
- COVID-19 false-negative rate < 15%
- Model card documents limitations

These are **targets** and are not claimed as achieved until the model is trained and evaluated.

### Provided data packs
The uploaded `aiml_healthcare_symptoms.csv` contains **800 rows and 17 columns**.  
The uploaded `aiml_healthcare_test_cases.txt` contains **10 healthcare test cases** (TEST_1 through TEST_10). The test cases cover healthcare triage, screening, hypertension classification, anemia detection, healthy baseline, model validation, evaluation, explainability and data-split validation. fileciteturn0file0L4-L13

These packs are retained in the repository as provided project materials. They are not used as the chest X-ray training dataset.

### Repository structure
```text
medscan-chest-xray-classification/
├── data/
│   ├── aiml_healthcare_symptoms.csv
│   └── aiml_healthcare_test_cases.txt
├── models/
├── notebooks/
├── results/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── gradcam.py
│   └── app.py
├── model_card.md
├── research_notes.md
├── requirements.txt
├── CHECKPOINT_1_SUBMISSION.md
├── .gitignore
└── README.md
```

### Setup
```bash
pip install -r requirements.txt
```

Place the 3000 chest X-ray images in:
```text
data/xray_raw/NORMAL/
data/xray_raw/PNEUMONIA/
data/xray_raw/COVID-19/
```

Then run:
```bash
python src/train.py
python src/evaluate.py --checkpoint models/best_resnet18.pt
python src/app.py
```

### Medical safety
This is an educational/research prototype and is not a medical device. It must not be used as a substitute for qualified clinical assessment, radiology interpretation, laboratory testing, or established medical workflows.
