"""Run the existing trainer and record metrics/artifacts in a local MLflow experiment."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import mlflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("creditcard.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--experiment", default="fraud-detection")
    args = parser.parse_args()
    if not args.data.is_file():
        raise SystemExit(f"Dataset not found: {args.data}")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)
    command = [
        sys.executable, "src/train.py",
        "--data", str(args.data),
        "--output-dir", str(args.output_dir),
        "--threshold", str(args.threshold),
        "--random-state", str(args.random_state),
    ]
    with mlflow.start_run() as run:
        mlflow.log_params({
            "model": "class_weighted_logistic_regression",
            "random_state": args.random_state,
            "threshold": args.threshold,
            "data_path": str(args.data),
        })
        subprocess.run(command, check=True)
        metrics_path = args.output_dir / "metrics.json"
        if not metrics_path.is_file():
            raise SystemExit(f"Trainer did not create {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for key in ("precision", "recall", "f1", "roc_auc", "average_precision_pr_auc", "accuracy"):
            mlflow.log_metric(key, float(metrics[key]))
        mlflow.log_artifacts(str(args.output_dir), artifact_path="training_artifacts")
        print(f"MLflow run ID: {run.info.run_id}")
        print(f"Tracking URI: {tracking_uri}")


if __name__ == "__main__":
    main()
