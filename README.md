# Credit Card Fraud Detection — End-to-End MLOps

A local-first MLOps portfolio project for imbalanced fraud classification. It extends the baseline model with MLflow experiment tracking, a validated FastAPI prediction service, feature-drift reports, a Streamlit dashboard, automated tests, CI, and Docker.

> Educational demonstration only. Drift does not prove model degradation, and a predicted score is not proof of fraud. Use human review and evaluate against trustworthy labelled outcomes before operational decisions.

## Architecture

    creditcard.csv -> validation/training -> model + metrics
                                 |              |
                                 v              v
                              MLflow        FastAPI /predict

    reference/current CSV -> PSI + KS drift report -> Streamlit dashboard
    Git push / pull request -> GitHub Actions -> unit tests

## Quickstart

Use Python 3.10 or 3.11.

    git clone https://github.com/BharathRoshan/Fraud-Detection.git
    cd Fraud-Detection
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python src/train.py --data creditcard.csv --output-dir artifacts

Windows PowerShell activation: .venv\Scripts\Activate.ps1

Training writes metrics.json, confusion_matrix.csv, and fraud_pipeline.joblib under artifacts/. Generated artifacts should not be committed.

## 1. Experiment tracking

    python src/train_tracked.py --data creditcard.csv --output-dir artifacts
    mlflow ui --backend-store-uri ./mlruns

Open the local MLflow UI URL (typically http://127.0.0.1:5000). Runs record parameters, metrics, and artifacts. The default tracking store is local.

## 2. Prediction API

Train the model, then start the API:

    uvicorn src.api:app --host 127.0.0.1 --port 8000

- Health endpoint: GET /health
- Interactive API docs: http://127.0.0.1:8000/docs
- Prediction endpoint: POST /predict

The JSON request has a features mapping. Every feature must match the training schema exactly. Generate a valid example body from a dataset row:

    import json
    import pandas as pd
    row = pd.read_csv("creditcard.csv").drop(columns=["Class"]).iloc[0]
    print(json.dumps({"features": {k: float(v) for k, v in row.items()}}))

Copy the printed JSON into Swagger UI. The API rejects missing, unexpected, and non-finite values. Configure the decision threshold using FRAUD_THRESHOLD (default 0.5); select a real threshold with validation data, not the final test set.

## 3. Feature drift monitoring

    python src/monitoring.py --reference data/reference.csv --current data/current.csv --output-dir monitoring_reports

The reference and current CSVs should share a feature schema. The report contains per-feature means, KS statistics/p-values, PSI scores, and drift flags. Target columns such as Class are excluded. Thresholds default to PSI >= 0.2 or KS p-value < 0.05; these are heuristics, not universal rules. Multiple tests can produce false alarms. Distribution drift is a signal to investigate, not proof that performance degraded.

## 4. Dashboard

    streamlit run app/dashboard.py

Select a generated monitoring_reports/drift_report.json or artifacts/metrics.json file. The dashboard reads local reports and does not upload them.

## 5. Tests

    pytest -q

The GitHub Actions workflow runs tests on pushes and pull requests.

## Docker (optional)

    docker build -t fraud-mlops .
    docker run --rm -p 8000:8000 -v "$(pwd)/artifacts:/app/artifacts" fraud-mlops

Train locally before starting the container. PowerShell users should use their shell's volume-mount syntax.

## Evaluation checklist

- [ ] Reproducible training with fixed random seed.
- [ ] MLflow logs parameters, metrics, and artifacts.
- [ ] API returns probability and thresholded class for valid input.
- [ ] API rejects invalid feature schemas and non-finite values.
- [ ] Drift tests flag a deliberately shifted feature but not identical data.
- [ ] CI passes from a clean environment.
- [ ] Record actual model metrics, limitations, and dashboard screenshots.

## Repository layout

- src/train.py — reproducible baseline
- src/train_tracked.py — MLflow experiment wrapper
- src/api.py — FastAPI inference
- src/monitoring.py — PSI and KS report
- app/dashboard.py — Streamlit report viewer
- tests/ — API and drift tests
- .github/workflows/ci.yml — automated tests

## Dataset and responsible use

This highly imbalanced benchmark contains transformed features. Preserve the original dataset attribution and terms. This project is for portfolio demonstration only; do not use it to automatically accuse or penalize people.
