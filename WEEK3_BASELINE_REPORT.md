# MedScan Diagnostics — Week 3 Baseline Model Report

## 1. Objective
Develop and evaluate baseline machine learning models using the available healthcare symptom dataset.

## 2. Dataset Overview
- Total records: 800
- Total columns: 17
- Input features: 15
- Target variable: Diagnosis
- Number of classes: 8
- Missing values: 0
- Duplicate records: 0

Each diagnosis class contains 100 records.

## 3. Data Preprocessing
- Patient_ID excluded from model inputs.
- Diagnosis used as the target variable.
- Numerical features processed using median imputation and standardization.
- Categorical features processed using most-frequent imputation and one-hot encoding.
- Stratified data splitting used to preserve class distribution.

## 4. Dataset Split

| Subset | Records | Percentage |
|---|---:|---:|
| Training | 560 | 70% |
| Validation | 120 | 15% |
| Testing | 120 | 15% |

## 5. Models Implemented
Two baseline classifiers were implemented:
1. Random Forest Classifier
2. Extra Trees Classifier

Random Forest was selected using validation macro F1-score.

## 6. Validation Results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Random Forest | 83.33% | 83.25% |
| Extra Trees | 81.67% | 81.54% |

## 7. Final Test Results

The selected Random Forest model was evaluated on the held-out test dataset.

| Metric | Result |
|---|---:|
| Accuracy | 81.67% |
| Macro Precision | 82.58% |
| Macro Recall | 81.67% |
| Macro F1-score | 81.59% |

## 8. Observations
- Pneumonia and Type2Diabetes achieved 100% recall on this test split.
- Healthy achieved 93% recall.
- Anaemia achieved 60% recall.
- Migraine achieved 73% recall.
- Further investigation is required for classes with lower recall.

## 9. Generated Artifacts
- `models/medscan_baseline.joblib`
- `results/baseline_metrics.json`
- `results/confusion_matrix.png`
- `results/test_predictions.csv`

The saved model was successfully reloaded and tested with a sample prediction.

## 10. Limitations
These results represent performance on a single held-out split of the available dataset. Dataset provenance and representativeness require verification. The model is an experimental academic prototype and is not intended for clinical diagnosis, treatment recommendations or medical decision-making.

## 11. Current Status
Dataset validation, preprocessing, baseline training, model comparison, held-out evaluation and model serialization have been implemented. Further testing, documentation review and milestone evidence verification remain necessary.
