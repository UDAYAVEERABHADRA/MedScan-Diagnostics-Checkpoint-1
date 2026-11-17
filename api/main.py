from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from api.predictor import HealthcarePredictor, ModelNotReady

app = FastAPI(title="MedScan Diagnostics API", version="1.0.0", description="Research prototype for tabular healthcare classification.")
predictor = None

class PatientFeatures(BaseModel):
    Age: int = Field(ge=0, le=120)
    Gender: Literal["M", "F"]
    Fever: int = Field(ge=0, le=1)
    Cough: int = Field(ge=0, le=1)
    Fatigue: int = Field(ge=0, le=1)
    Breathlessness: int = Field(ge=0, le=1)
    Chest_Pain: int = Field(ge=0, le=1)
    Nausea: int = Field(ge=0, le=1)
    Headache: int = Field(ge=0, le=1)
    BMI: float = Field(gt=0, le=100)
    BP_Systolic: int = Field(gt=0, le=300)
    BP_Diastolic: int = Field(gt=0, le=200)
    SpO2: int = Field(ge=0, le=100)
    Glucose_mg_dL: int = Field(gt=0, le=1000)
    HbA1c: float = Field(ge=0, le=30)

@app.on_event("startup")
def load_model():
    global predictor
    try:
        predictor = HealthcarePredictor()
    except ModelNotReady:
        predictor = None

@app.get("/health")
def health():
    return {"status": "ok" if predictor else "model_not_loaded", "model_loaded": predictor is not None}

@app.post("/predict")
def predict(patient: PatientFeatures):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check MEDSCAN_MODEL_PATH and restart the API.")
    try:
        output = predictor.predict(patient.model_dump())
        return {**output, "warning": "Research prototype only; not a medical diagnosis. Consult a qualified healthcare professional."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(exc).__name__}")
