# MedScan Diagnostics — Final Project Demonstration

**Duration: 5 Minutes**

## 1. Introduction (0:00–0:40)

"Good morning. Today I am presenting MedScan Diagnostics, a machine learning-based healthcare classification prototype.

The objective of this project is to demonstrate how structured healthcare data can be processed using machine learning and made accessible through an API.

The system includes data preprocessing, model training, hyperparameter experimentation, model evaluation and API integration."

## 2. Dataset and Model (0:40–1:30)

"The project uses a healthcare symptom dataset containing 800 records and 17 columns.

The target variable is Diagnosis, containing eight different classes.

Patient_ID was excluded from model training.

The dataset was divided into training, validation and testing sets using a stratified 70:15:15 split.

A Random Forest classifier was trained using two different hyperparameter configurations. MLflow was used to track the experiments."

## 3. Model Evaluation (1:30–2:15)

"Configuration B was selected based on its validation performance.

It achieved:
- Validation accuracy: 84.17%
- Validation macro F1-score: 0.8401
- Test accuracy: 83.33%
- Test macro F1-score: 0.8330

Both configurations achieved the same test accuracy, so their test performance was nearly identical."

## 4. Live API Demonstration (2:15–3:30)

"Next, I will demonstrate the model serving functionality.

The trained model is integrated with a FastAPI backend.

The API provides a POST /predict endpoint.

Through the interactive API documentation, users can submit feature values and receive a predicted class along with prediction probabilities."

**Demonstration:**

1. Open http://127.0.0.1:8000/docs
2. Expand POST /predict.
3. Click Try it out.
4. Enter valid example data using the displayed request schema.
5. Click Execute.
6. Explain the prediction response.

## 5. Integration and Performance Testing (3:30–4:20)

"The application was tested using Pytest.

All three integration tests passed:
- Health endpoint verification.
- Prediction response validation.
- Invalid feature rejection.

A load test was conducted with 50 concurrent requests and 250 total requests.

The system completed 249 requests successfully, with one failed request.

The measured throughput was 5.60 requests per second, with a mean latency of approximately 8.21 seconds."

## 6. Model Card and Limitations (4:20–5:00)

"The model card documents the dataset, model configuration, evaluation results, intended use and limitations.

A demographic subgroup fairness audit has not yet been completed.

The current system is an academic research prototype and has not undergone clinical validation.

Further work will focus on performance optimization, fairness evaluation, security and validation using more representative datasets.

Thank you."

---

## Demonstration Checklist

- [ ] Start FastAPI successfully.
- [ ] Open the interactive API documentation.
- [ ] Demonstrate one successful prediction.
- [ ] Show the three passing integration tests.
- [ ] Present actual load-testing results.
- [ ] Explain model limitations.
- [ ] Keep the recording within five minutes.