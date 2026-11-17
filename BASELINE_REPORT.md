# MedScan Diagnostics — Week 3 Baseline Report

## 1. Objective
Build a reproducible image ingestion and preprocessing pipeline and establish a ResNet-18 baseline for three classes: NORMAL, PNEUMONIA and COVID-19.

## 2. Dataset availability and provenance
The Checkpoint 1 package contains `aiml_healthcare_symptoms.csv` and `aiml_healthcare_test_cases.txt`. It does **not** contain the 3,000 chest X-ray images specified in the project brief. These supporting files are not image data and are not used to train this image classifier.

**Current baseline status: BLOCKED — awaiting the approved chest X-ray image dataset.** No training metrics are reported because no image model could be trained or tested from the supplied package. Do not treat acceptance targets as measured results.

When the approved image dataset is received, record:
- Source/provider and permitted-use terms:
- Dataset version/date:
- Total readable images:
- NORMAL count:
- PNEUMONIA count:
- COVID-19 count:
- Patient-level grouping available?:
- Duplicate/leakage checks performed:

## 3. Ingestion and preprocessing implementation
The included pipeline:
- Reads supported image formats from one folder per class.
- Verifies images can be decoded and logs unreadable paths.
- Converts images to RGB and resizes to 224 × 224.
- Applies ImageNet channel normalization.
- Applies horizontal flip and small rotation to training images only.
- Uses stratified train/validation/test splitting (70% / 15% / 15%) with a fixed seed.
- Saves split manifests and split-count summaries.
- Uses ResNet-18 with an ImageNet-pretrained backbone and trainable final residual block/classification head.

## 4. Baseline model and training configuration
- Architecture: ResNet-18, three output classes.
- Loss: CrossEntropyLoss.
- Optimizer: AdamW.
- Learning rate: 0.0001.
- Weight decay: 0.0001.
- Default epochs: 10.
- Input size: 224 × 224.
- Random seed: 42.
- Model selection: best validation macro one-vs-rest ROC-AUC.

## 5. Evaluation protocol
Evaluate once on the held-out test split. Report accuracy, per-class precision/recall/F1, macro ROC-AUC (one-vs-rest), confusion matrix and COVID-19 false-negative rate.

Acceptance criteria from the project brief:
- Macro AUC ≥ 0.87.
- COVID-19 false-negative rate < 0.15.

These are targets, not achieved results.

## 6. Results
| Metric | Result |
|---|---|
| Number of X-ray images | Not available in supplied Checkpoint 1 ZIP |
| Training status | Not run — image dataset unavailable |
| Test accuracy | Not measured |
| Macro ROC-AUC | Not measured |
| COVID-19 false-negative rate | Not measured |

After training, the evaluation script writes the actual measured values to `results/baseline_metrics.json`. Update this section from that file; do not invent values.

## 7. Limitations and next steps
1. Obtain the approved, correctly labelled chest X-ray image dataset and confirm its permitted use.
2. Check patient identifiers and duplicates before splitting to reduce leakage risk; prefer patient-level splitting if identifiers are available.
3. Run the data validation, training and held-out evaluation commands in the README.
4. Review class-wise errors and confirm the COVID-19 false-negative rate against the target.
5. Preserve the split manifests, training history, metrics and model checkpoint as reproducibility artifacts.

## 8. Safety
This is an educational research prototype, not a medical device. It must not be used for diagnosis, triage or treatment decisions.
