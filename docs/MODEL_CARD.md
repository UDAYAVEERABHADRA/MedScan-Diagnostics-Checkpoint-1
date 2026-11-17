# MedScan Diagnostics — Model Card

## 1. Model Overview

- Model: Random Forest Classifier
- Selected Configuration: Configuration B
- Purpose: Healthcare diagnosis classification
- Serving Framework: FastAPI
- Endpoint: POST /predict
- Status: Academic research prototype

## 2. Dataset Information

- Dataset: aiml_healthcare_symptoms.csv
- Total Records: 800
- Total Columns: 17
- Target Variable: Diagnosis
- Number of Classes: 8

Diagnosis Classes:
1. Anaemia
2. Gastroenteritis
3. Healthy
4. Hypertension
5. Influenza
6. Migraine
7. Pneumonia
8. Type2Diabetes

Patient_ID was excluded from model training.

## 3. Data Splitting

- Training: 70% (560 records)
- Validation: 15% (120 records)
- Testing: 15% (120 records)

Stratified sampling was used.

## 4. Data Preprocessing

Numerical features:
- Median imputation
- StandardScaler

Categorical features:
- Most-frequent imputation
- OneHotEncoder

## 5. Selected Model Configuration

Algorithm: Random Forest

- n_estimators = 300
- max_depth = 12
- min_samples_split = 5
- min_samples_leaf = 2
- max_features = sqrt
- class_weight = balanced
- random_state = 42

## 6. Model Evaluation

### Validation Results

- Accuracy: 84.17%
- Macro Precision: 0.8467
- Macro Recall: 0.8417
- Macro F1: 0.8401

### Test Results

- Accuracy: 83.33%
- Macro Precision: 0.8397
- Macro Recall: 0.8333
- Macro F1: 0.8330

Configuration B was selected based on its better validation macro F1-score.

Both configurations achieved 83.33% test accuracy, so their test performance was nearly identical.

## 7. Limitations

- Dataset contains only 800 records.
- No external clinical validation has been performed.
- Dataset representativeness is not established.
- Predictions may be incorrect.
- Model performance may vary across populations.

## 8. Bias and Fairness Audit

A demographic subgroup fairness audit has not yet been completed because suitable subgroup labels and evaluation results are unavailable.

The model should not be considered unbiased or fair without further evaluation.

Future work should include subgroup performance analysis, false-negative rates, precision and recall comparisons.

## 9. Intended Use

This model is intended only for academic demonstration and research.

It must not be used for:
- Medical diagnosis
- Treatment recommendations
- Clinical decision-making
- Emergency triage

## 10. Safety Statement

MedScan Diagnostics is an academic research prototype and is not clinically validated. It must not replace qualified healthcare professionals.