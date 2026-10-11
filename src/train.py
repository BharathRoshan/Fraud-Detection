"""Train and evaluate a class-weighted logistic-regression fraud baseline."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("creditcard.csv"),
                        help="Path to the CSV containing a Class target column.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"),
                        help="Directory for model and evaluation outputs.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction held out for the final test split (default: 0.2).")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for the reported confusion matrix.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise SystemExit("--test-size must be between 0 and 1.")
    if not 0 <= args.threshold <= 1:
        raise SystemExit("--threshold must be between 0 and 1.")
    if not args.data.is_file():
        raise SystemExit(f"Dataset not found: {args.data}")

    data = pd.read_csv(args.data)
    if "Class" not in data.columns:
        raise SystemExit("Expected a binary target column named 'Class'.")
    if data.empty:
        raise SystemExit("The input CSV is empty.")
    if data["Class"].isna().any():
        raise SystemExit("The target column contains missing values.")
    if data["Class"].nunique() != 2:
        raise SystemExit("Expected exactly two target classes in 'Class'.")

    y = data["Class"].astype(int)
    X = data.drop(columns=["Class"])
    if not set(y.unique()).issubset({0, 1}):
        raise SystemExit("Expected Class labels 0 (legitimate) and 1 (fraud).")

    numeric_scale = [col for col in ["Time", "Amount"] if col in X.columns]
    preprocessor = ColumnTransformer(
        transformers=[("scale_time_amount", StandardScaler(), numeric_scale)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=args.random_state,
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= args.threshold).astype(int)
    metrics = {
        "model": "class_weighted_logistic_regression",
        "data_path": str(args.data),
        "random_state": args.random_state,
        "test_size": args.test_size,
        "decision_threshold": args.threshold,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "fraud_rate_test": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision_pr_auc": float(average_precision_score(y_test, probabilities)),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=[0, 1]),
        index=["actual_legitimate", "actual_fraud"],
        columns=["predicted_legitimate", "predicted_fraud"],
    ).to_csv(args.output_dir / "confusion_matrix.csv")
    joblib.dump(model, args.output_dir / "fraud_pipeline.joblib")

    summary = {key: value for key, value in metrics.items()
               if key != "classification_report"}
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
