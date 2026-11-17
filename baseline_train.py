
from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Configuration
DATA_PATH = Path("data/aiml_healthcare_symptoms.csv")
RESULTS = Path("results")
MODELS = Path("models")
RANDOM_STATE = 42

RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

print("=" * 60)
print("MEDSCAN DIAGNOSTICS - BASELINE MODEL TRAINING")
print("=" * 60)

# 1. Load dataset
df = pd.read_csv(DATA_PATH)

if "Diagnosis" not in df.columns:
    raise ValueError("Target column 'Diagnosis' not found.")

# Exclude identifier and target from input features
X = df.drop(columns=["Patient_ID", "Diagnosis"])
y = df["Diagnosis"]

categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

print(f"\nDataset: {df.shape}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {y.nunique()}")

# 2. Stratified 70/15/15 split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30,
    random_state=RANDOM_STATE, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=RANDOM_STATE, stratify=y_temp
)

print("\nSplit sizes:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

# 3. Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical),
    ("categorical", categorical_pipeline, categorical)
])

# 4. Candidate baseline models
candidates = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )
}

validation_results = {}
trained = {}

# 5. Train and compare on validation data
for name, classifier in candidates.items():
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_val, predictions, average="macro", zero_division=0
    )

    validation_results[name] = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1)
    }

    trained[name] = pipeline

    print(f"\n{name}")
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation macro F1: {macro_f1:.4f}")

# 6. Select best model using validation macro F1
best_name = max(
    validation_results,
    key=lambda name: validation_results[name]["macro_f1"]
)

print(f"\nSelected model: {best_name}")

# 7. Refit selected model on training + validation data
X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])

best_model = trained[best_name]
best_model.fit(X_train_full, y_train_full)

# 8. Final evaluation on untouched test set
test_predictions = best_model.predict(X_test)

accuracy = accuracy_score(y_test, test_predictions)
precision, recall, macro_f1, _ = precision_recall_fscore_support(
    y_test, test_predictions,
    average="macro", zero_division=0
)

report = classification_report(
    y_test, test_predictions,
    output_dict=True, zero_division=0
)

print("\n" + "=" * 60)
print("FINAL HELD-OUT TEST RESULTS")
print("=" * 60)
print(f"Model: {best_name}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro Precision: {precision:.4f}")
print(f"Macro Recall: {recall:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print("\nClassification report:")
print(classification_report(
    y_test, test_predictions, zero_division=0
))

# 9. Save model
joblib.dump(best_model, MODELS / "medscan_baseline.joblib")

# 10. Save metrics
metrics = {
    "selected_model": best_name,
    "random_state": RANDOM_STATE,
    "split": {
        "training": len(X_train),
        "validation": len(X_val),
        "testing": len(X_test)
    },
    "validation_results": validation_results,
    "test_metrics": {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1)
    },
    "classification_report": report
}

with open(RESULTS / "baseline_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# 11. Save confusion matrix
labels = sorted(y.unique())
cm = confusion_matrix(y_test, test_predictions, labels=labels)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)
display.plot(xticks_rotation=45, cmap="Blues")
plt.title(f"MedScan - {best_name} Test Confusion Matrix")
plt.tight_layout()
plt.savefig(RESULTS / "confusion_matrix.png", dpi=200)
plt.close()

# 12. Save test predictions for audit
prediction_results = X_test.copy()
prediction_results["Actual_Diagnosis"] = y_test
prediction_results["Predicted_Diagnosis"] = test_predictions
prediction_results.to_csv(
    RESULTS / "test_predictions.csv", index=False
)

print("\nSaved files:")
print("- models/medscan_baseline.joblib")
print("- results/baseline_metrics.json")
print("- results/confusion_matrix.png")
print("- results/test_predictions.csv")
print("\nBASELINE TRAINING COMPLETED.")
