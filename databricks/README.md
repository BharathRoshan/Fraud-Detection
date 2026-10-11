# Databricks notebooks

These notebooks extend the repository's fraud-classification baseline into a batch MLOps and monitoring workflow using Databricks Free Edition.

## Before you start

- Use a Unity Catalog catalog/schema where you have permissions to create tables and volumes.
- Create a volume such as `fraud_data` and upload `creditcard.csv` through the Databricks UI.
- Update `SOURCE_PATH` in Notebook 00 to your actual path:
  `/Volumes/<catalog>/<schema>/fraud_data/creditcard.csv`.
- Keep the same catalog and schema selected for all notebooks.
- Use Serverless notebook compute where available.
- Do not put access tokens, credentials, or workspace secrets into notebook source or Git.

## Run order

1. `00_data_ingestion_validation.ipynb`
2. `01_feature_prep_split.ipynb`
3. `02_model_training_mlflow.ipynb`
4. `03_batch_inference.ipynb`
5. `04_drift_performance_monitoring.ipynb`
6. `05_monitoring_dashboard.ipynb`
7. `06_pipeline_validation.ipynb`
8. `07_alerting_and_run_history.ipynb`

Run each notebook after its prerequisites complete. If rerunning from the beginning, note that the ingestion and split notebooks overwrite their corresponding tables. Notebook 04 stores a normal held-out test-batch comparison separately from its deliberately simulated drift batch.

## Tables produced

| Table (in your chosen catalog/schema) | Purpose |
|---|---|
| `fraud_transactions_raw` | Ingested source data |
| `fraud_transactions_train` | Training split |
| `fraud_transactions_test` | Held-out test split |
| `fraud_predictions_test` | Batch predictions, probabilities, labels, and MLflow run ID |
| `fraud_drift_normal` | Feature drift metrics for the held-out batch |
| `fraud_drift_simulated` | Synthetic drift used to test monitoring |
| `fraud_model_performance` | Labeled evaluation metrics |
| `fraud_model_run_registry` | MLflow run ID used by batch inference |
| `fraud_monitoring_events` | Threshold check results and monitoring history |

## Important interpretation notes

- PSI and KS flags identify distribution changes for investigation; they do not prove model quality has degraded.
- Performance metrics require known labels. In real systems, transaction labels may arrive later.
- Notebook 07 writes alert events to Delta. It does not by itself send email or mobile notifications.
- A saved notebook workflow is not proof that a scheduled job runs unattended. Confirm the job's run history and available Free Edition features separately.
