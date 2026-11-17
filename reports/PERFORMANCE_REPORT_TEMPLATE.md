# Week 9 Performance Report — MedScan Diagnostics

## Integration
- Serving layer: FastAPI
- Model: Week 6 selected Random Forest pipeline (`models/medscan_week6_best.joblib`)
- Endpoints: `GET /health`, `POST /predict`
- Input validation: Pydantic schema for all 15 model features

## Integration tests
Command: `python -m pytest tests/test_api.py -v`

Observed result: [Paste actual pytest summary]

## Load test
Command: `python load_testing/benchmark.py --requests 250 --concurrency 50`

| Measure | Observed result |
|---|---|
| Total requests | 250 |
| Concurrent clients | 50 |
| Successful requests | [actual output] |
| Failed requests | [actual output] |
| Throughput (requests/second) | [actual output] |
| Mean latency (ms) | [actual output] |
| Median latency (ms) | [actual output] |
| p95 latency (ms) | [actual output] |
| Min / Max latency (ms) | [actual output] |

## Notes
Record machine specifications, whether the API was run with one Uvicorn worker, and any errors or bottlenecks. Measurements are environment-dependent and must be reported from the actual run.

The model is a research prototype and is not clinically validated.
