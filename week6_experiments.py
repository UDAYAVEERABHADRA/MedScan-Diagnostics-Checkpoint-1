from pathlib import Path
import json
import copy

import joblib
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# MedScan Week 6: two Random Forest hyperparameter experiments
DATA_PATH = Path("data/aiml_healthcare_symptoms.csv")
RESULTS = Path("results/week6")
MODELS = Path("models")
RANDOM_STATE = 42
TARGET = "Diagnosis"

# Same algorithm, two hyperparameter configurations.
CONFIGS = [
    {
        "run_name": "RandomForest_A_baseline",
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
    {
        "run_name": "RandomForest_B_regularized",
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
]

def make_pipeline(X, config):
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numerical),
        ("categorical", categorical_pipeline, categorical),
    ])

    classifier = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_split=config["min_samples_split"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"],
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier),
    ])

def calculate_metrics(y_true, y_pred):
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
    }

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")
    if "Patient_ID" in df.columns:
        X = df.drop(columns=["Patient_ID", TARGET])
    else:
        X = df.drop(columns=[TARGET])
    y = df[TARGET]

    print("=" * 62)
    print("MEDSCAN DIAGNOSTICS — WEEK 6 EXPERIMENTATION")
    print("=" * 62)
    print(f"Dataset: {DATA_PATH} | Records: {len(df)} | Features: {X.shape[1]}")
    print(f"Target: {TARGET} | Classes: {sorted(y.astype(str).unique())}")

    # Fixed stratified 70/15/15 split; test set remains untouched until final evaluation.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"Split sizes: train={len(X_train)}, validation={len(X_val)}, test={len(X_test)}")

    # SQLite tracking store avoids MLflow's deprecated local file-store backend.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("MedScan_Week6_Tabular_RandomForest")

    experiment_results = []
    best_config = None
    best_validation_f1 = -1.0

    for config in CONFIGS:
        print(f"\n--- {config['run_name']} ---")
        pipeline = make_pipeline(X_train, config)

        with mlflow.start_run(run_name=config["run_name"]):
            params = {
                "model": "RandomForestClassifier",
                "n_estimators": config["n_estimators"],
                "max_depth": str(config["max_depth"]),
                "min_samples_split": config["min_samples_split"],
                "min_samples_leaf": config["min_samples_leaf"],
                "max_features": config["max_features"],
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
                "split": "70/15/15 stratified",
                "dataset_rows": len(df),
                "feature_count": X.shape[1],
            }
            mlflow.log_params(params)

            pipeline.fit(X_train, y_train)
            val_predictions = pipeline.predict(X_val)
            val_metrics = calculate_metrics(y_val, val_predictions)
            mlflow.log_metrics(
                {f"validation_{key}": value for key, value in val_metrics.items()}
            )

            print(
                f"Validation accuracy={val_metrics['accuracy']:.4f} | "
                f"macro-F1={val_metrics['macro_f1']:.4f}"
            )

            # Refit on train + validation only after validation-based selection
            # is recorded. The held-out test set is not used for model selection.
            X_train_full = pd.concat([X_train, X_val], axis=0)
            y_train_full = pd.concat([y_train, y_val], axis=0)
            final_candidate = make_pipeline(X_train_full, config)
            final_candidate.fit(X_train_full, y_train_full)

            test_predictions = final_candidate.predict(X_test)
            test_metrics = calculate_metrics(y_test, test_predictions)
            mlflow.log_metrics(
                {f"test_{key}": value for key, value in test_metrics.items()}
            )

            report_text = classification_report(
                y_test, test_predictions, zero_division=0
            )
            report_path = RESULTS / f"{config['run_name']}_classification_report.txt"
            report_path.write_text(report_text, encoding="utf-8")
            mlflow.log_artifact(str(report_path))

            cm = confusion_matrix(y_test, test_predictions, labels=sorted(y.unique()))
            fig, ax = plt.subplots(figsize=(7, 6))
            ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=sorted(y.unique())
            ).plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
            ax.set_title(f"Week 6 — {config['run_name']}")
            fig.tight_layout()
            cm_path = RESULTS / f"{config['run_name']}_confusion_matrix.png"
            fig.savefig(cm_path, dpi=180)
            plt.close(fig)
            mlflow.log_artifact(str(cm_path))

            # Save the trusted locally trained model as an MLflow artifact.
            temp_model_path = RESULTS / f"{config['run_name']}_model.joblib"
            joblib.dump(final_candidate, temp_model_path)
            mlflow.log_artifact(str(temp_model_path), artifact_path="model")

            experiment_results.append({
                "configuration": config,
                "validation_metrics": val_metrics,
                "test_metrics": test_metrics,
            })

            if val_metrics["macro_f1"] > best_validation_f1:
                best_validation_f1 = val_metrics["macro_f1"]
                best_config = config
                best_model = final_candidate
                best_test_metrics = test_metrics
                best_test_predictions = test_predictions
                best_report = report_text

    # Save selected model and final evidence.
    best_model_path = MODELS / "medscan_week6_best.joblib"
    joblib.dump(best_model, best_model_path)

    comparison = pd.DataFrame([
        {
            "configuration": item["configuration"]["run_name"],
            "learning_setup": (
                f"trees={item['configuration']['n_estimators']}, "
                f"max_depth={item['configuration']['max_depth']}, "
                f"min_split={item['configuration']['min_samples_split']}, "
                f"min_leaf={item['configuration']['min_samples_leaf']}"
            ),
            **{f"validation_{k}": v for k, v in item["validation_metrics"].items()},
            **{f"test_{k}": v for k, v in item["test_metrics"].items()},
        }
        for item in experiment_results
    ])
    comparison.to_csv(RESULTS / "experiment_comparison.csv", index=False)
    (RESULTS / "best_classification_report.txt").write_text(
        best_report, encoding="utf-8"
    )

    summary = {
        "dataset": str(DATA_PATH),
        "records": len(df),
        "features": X.shape[1],
        "target": TARGET,
        "split_sizes": {
            "training": len(X_train),
            "validation": len(X_val),
            "testing": len(X_test),
        },
        "selection_metric": "validation_macro_f1",
        "best_configuration": best_config,
        "best_validation_macro_f1": best_validation_f1,
        "best_model_test_metrics": best_test_metrics,
        "experiments": experiment_results,
        "saved_model": str(best_model_path),
        "mlflow_tracking_uri": "sqlite:///mlflow.db",
        "note": "Experimental healthcare classification prototype; not for clinical diagnosis.",
    }
    (RESULTS / "week6_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 62)
    print("WEEK 6 EXPERIMENT COMPARISON")
    print("=" * 62)
    print(comparison.to_string(index=False))
    print("\nBest configuration (selected by validation macro-F1):")
    print(best_config)
    print("\nBest model held-out test metrics:")
    print(json.dumps(best_test_metrics, indent=2))
    print("\nSaved:")
    print(f"- {best_model_path}")
    print(f"- {RESULTS / 'experiment_comparison.csv'}")
    print(f"- {RESULTS / 'week6_summary.json'}")
    print(f"- {RESULTS / 'best_classification_report.txt'}")
    print("- MLflow database: mlflow.db")
    print("\nTo view experiments later: mlflow ui")

if __name__ == "__main__":
    main()
