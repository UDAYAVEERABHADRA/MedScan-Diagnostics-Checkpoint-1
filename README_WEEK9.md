# MedScan Diagnostics — Week 9 Integration + Testing

This package serves the Week 6 tabular Random Forest pipeline through FastAPI, includes real-model integration tests, and a reproducible 50-concurrent-client benchmark.

## Copy into the project
Copy `api/`, `tests/`, and `load_testing/` into the project root. Keep the Week 6 model at `models/medscan_week6_best.joblib`.

## Install (inside the existing virtual environment)
```powershell
pip install -r requirements-week9.txt
```
If you already maintain a project requirements file, install the packages listed in `requirements-week9.txt` and merge them into it.

## Start the API
From the project root:
```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000
```
Open `http://127.0.0.1:8000/docs` for Swagger UI and `http://127.0.0.1:8000/health` for health status.

## Run integration tests
In a second PowerShell terminal, from the project root:
```powershell
python -m pytest tests/test_api.py -v
```

## Run load test (50 concurrent clients)
Keep the API running. In another terminal:
```powershell
python load_testing/benchmark.py --requests 250 --concurrency 50
```
The script prints successful/failed requests, throughput, mean/median/p95/min/max latency. Record the actual output in the milestone report; do not estimate or invent measurements.

## Sample request
```json
{
  "Age": 35, "Gender": "F", "Fever": 1, "Cough": 1, "Fatigue": 0,
  "Breathlessness": 0, "Chest_Pain": 0, "Nausea": 1, "Headache": 1,
  "BMI": 24.5, "BP_Systolic": 118, "BP_Diastolic": 78, "SpO2": 98,
  "Glucose_mg_dL": 95, "HbA1c": 5.4
}
```

This is a research prototype, not a clinically validated diagnostic system.
