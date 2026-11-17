# Model Card — MedScan ResNet-18

## Model summary
ImageNet-pretrained ResNet-18 fine-tuned for three-class chest X-ray classification:
NORMAL, PNEUMONIA and COVID-19.

## Intended use
Educational and research demonstration of medical-image classification and explainability.

## Not intended for
- Clinical diagnosis
- Patient triage
- Treatment decisions
- Emergency decisions
- Autonomous medical use

## Data
The project brief specifies a provided 3000-image chest X-ray dataset. The two additional project data packs included in this submission are healthcare symptom/test-case files and are retained as supporting project materials.

## Evaluation
Final evaluation will report:
- Macro ROC-AUC
- Per-class ROC-AUC
- Accuracy
- Precision
- Recall/sensitivity
- Specificity
- F1-score
- Confusion matrix
- COVID-19 false-negative rate

Acceptance targets are AUC ≥0.87 and COVID-19 FNR <15%.

## Explainability
Grad-CAM will show regions contributing to the selected class prediction. The visualization is not proof that a highlighted region is clinically abnormal.

## Limitations
- Small dataset relative to clinical imaging requirements.
- Possible class imbalance.
- Potential acquisition and dataset bias.
- Possible patient-level leakage if patient identifiers are not controlled during splitting.
- Generalization to other hospitals, scanners, populations and image views is unknown.
- High AUC does not establish clinical utility.
- Probability scores are not automatically calibrated clinical probabilities.
- No prospective clinical validation or regulatory assessment is included.

## Safety
This model is not a medical device and must not replace qualified healthcare professionals or established diagnostic testing.
