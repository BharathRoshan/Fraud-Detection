# Databricks MLOps architecture

This diagram describes the intended notebook flow in this repository. The simulated drift path is a monitoring test and must not be interpreted as real production drift.

```mermaid
flowchart TD
    A[creditcard.csv] --> B[00 · Ingestion and validation]
    B --> C[(Delta: raw transactions)]
    C --> D[01 · Feature preparation and stratified split]
    D --> E[(Delta: train and test)]
    E --> F[02 · Train class-weighted Logistic Regression]
    F --> G[MLflow run, metrics, model artifact]
    G --> H[03 · Batch inference]
    E --> H
    H --> I[(Delta: predictions)]
    E --> J[04 · Drift and performance monitoring]
    I --> J
    J --> K[(Delta: drift and performance metrics)]
    K --> L[05 · Monitoring dashboard]
    K --> M[07 · Threshold checks and run history]
    M --> N[(Delta: monitoring events)]
    N --> L
    M --> O[06 · Pipeline validation]
```

## Main components

- **Databricks notebooks:** execute ingestion, preparation, training, inference, monitoring, visualization, validation, and alert-history tasks.
- **Delta tables:** persist input data, split datasets, predictions, drift results, evaluation metrics, and monitoring events.
- **MLflow:** track training parameters and metrics and store the model artifact used for batch scoring.
- **Monitoring:** compare feature distributions with PSI and KS statistics, evaluate metrics when labels are available, and record threshold results.

## Operational boundaries

- The inference path is batch-based; this repository does not require a paid online model-serving endpoint.
- The alert notebook records threshold results in Delta. It does not send external email or push notifications.
- The simulated drift batch validates the detector and should remain separate from actual incoming data.
- Schedule automation only after a full job run has been verified in the target workspace.
