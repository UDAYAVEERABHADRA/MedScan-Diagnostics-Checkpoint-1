from pathlib import Path
import os
import joblib
import pandas as pd

FEATURES = [
    "Age", "Gender", "Fever", "Cough", "Fatigue", "Breathlessness",
    "Chest_Pain", "Nausea", "Headache", "BMI", "BP_Systolic",
    "BP_Diastolic", "SpO2", "Glucose_mg_dL", "HbA1c",
]

class ModelNotReady(RuntimeError):
    pass

class HealthcarePredictor:
    def __init__(self, model_path=None):
        path = Path(model_path or os.getenv("MEDSCAN_MODEL_PATH", "models/medscan_week6_best.joblib"))
        if not path.is_file():
            raise ModelNotReady(f"Model artifact not found: {path}. Place medscan_week6_best.joblib in models/.")
        self.model = joblib.load(path)
        self.model_path = str(path)

    def predict(self, features: dict):
        row = pd.DataFrame([{name: features[name] for name in FEATURES}], columns=FEATURES)
        predicted = str(self.model.predict(row)[0])
        result = {"predicted_class": predicted}
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(row)[0]
            classes = getattr(self.model, "classes_", None)
            if classes is None and hasattr(self.model, "named_steps"):
                classes = getattr(list(self.model.named_steps.values())[-1], "classes_", [])
            result["probabilities"] = {str(label): float(prob) for label, prob in zip(classes, probabilities)}
        return result
