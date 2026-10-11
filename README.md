# Credit Card Fraud Detection

A reproducible baseline for binary fraud classification using the public credit-card transaction dataset in `creditcard.csv`. The repository includes the original exploratory notebook and a script-based training path that saves evaluation artifacts.

## Problem

Fraud detection is highly imbalanced: a model can achieve high accuracy while missing most fraudulent transactions. For that reason, evaluate the minority class using precision, recall, F1, average precision / PR-AUC, and the confusion matrix—not accuracy alone.

## Repository contents

- `Fraud Detection.ipynb` — original exploratory notebook.
- `src/train.py` — reproducible baseline training and evaluation.
- `creditcard.csv` — dataset already present in this repository.
- `requirements.txt` — dependencies for the baseline script.

## Run the baseline

Use Python 3.10 or 3.11.

```bash
git clone https://github.com/BharathRoshan/Fraud-Detection.git
cd Fraud-Detection
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/train.py --data creditcard.csv --output-dir artifacts
```

The script uses a stratified train/test split, scales the `Time` and `Amount` columns, trains a class-weighted logistic-regression baseline, and writes metrics, a confusion matrix, and a serialized pipeline to `artifacts/`.

Optional arguments:

```bash
python src/train.py --help
```

## Evaluation and interpretation

The script reports precision, recall, F1, ROC-AUC, average precision (PR-AUC), and the confusion matrix at the configured decision threshold. The default threshold is 0.5; it is a starting point, not a universally optimal fraud threshold.

For a stronger experiment:

1. Create train, validation, and test splits with stratification.
2. Tune the threshold using the validation set only.
3. Compare logistic regression with tree-based models and resampling approaches.
4. Report the precision-recall trade-off and estimated review workload.
5. Fix random seeds and keep the test set untouched until final evaluation.

## Important caveats

- This is a portfolio baseline, not a production fraud-control system.
- Dataset features `V1`–`V28` are transformed features; they are not directly interpretable business attributes.
- A random split may not represent future transactions. For deployment-like evaluation, consider a time-aware split where appropriate.
- Do not treat model scores as proof of fraud. Human review, privacy, monitoring, and a documented escalation process are necessary in real workflows.

## Dataset and citation

The CSV is the credit-card fraud dataset commonly distributed for the European cardholder fraud-detection benchmark. If you redistribute the dataset or publish results, retain the original dataset source and applicable terms with your project.
