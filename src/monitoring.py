"""Compare reference and current feature distributions and write a drift report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Compute PSI using reference quantile bins with small-count smoothing."""
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)
    ref_pct = np.clip(ref_counts / len(ref), 1e-6, None)
    cur_pct = np.clip(cur_counts / len(cur), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compare_frames(reference: pd.DataFrame, current: pd.DataFrame,
                   psi_threshold: float = 0.2, p_value_threshold: float = 0.05) -> dict:
    """Return per-feature drift statistics. Target columns are excluded."""
    ignored = {"Class", "class", "target", "label"}
    common = [c for c in reference.columns if c in current.columns and c not in ignored]
    results = []
    for column in common:
        ref = pd.to_numeric(reference[column], errors="coerce").dropna()
        cur = pd.to_numeric(current[column], errors="coerce").dropna()
        if ref.empty or cur.empty:
            continue
        ks = ks_2samp(ref, cur)
        psi = population_stability_index(ref, cur)
        results.append({
            "feature": column,
            "reference_rows": int(len(ref)),
            "current_rows": int(len(cur)),
            "reference_mean": float(ref.mean()),
            "current_mean": float(cur.mean()),
            "ks_statistic": float(ks.statistic),
            "ks_p_value": float(ks.pvalue),
            "psi": psi,
            "drift_detected": bool(psi >= psi_threshold or ks.pvalue < p_value_threshold),
        })
    drifted = [item["feature"] for item in results if item["drift_detected"]]
    return {
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "features_compared": len(results),
        "drifted_features": drifted,
        "drift_detected": bool(drifted),
        "thresholds": {"psi": psi_threshold, "ks_p_value": p_value_threshold},
        "features": results,
        "interpretation": (
            "Drift is a change in feature distribution, not proof that model performance degraded. "
            "Evaluate labelled outcomes when they become available."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Reference/baseline CSV.")
    parser.add_argument("--current", type=Path, required=True, help="Recent/current CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("monitoring_reports"))
    parser.add_argument("--psi-threshold", type=float, default=0.2)
    parser.add_argument("--p-value-threshold", type=float, default=0.05)
    args = parser.parse_args()
    for path in (args.reference, args.current):
        if not path.is_file():
            raise SystemExit(f"CSV not found: {path}")
    report = compare_frames(
        pd.read_csv(args.reference),
        pd.read_csv(args.current),
        psi_threshold=args.psi_threshold,
        p_value_threshold=args.p_value_threshold,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "drift_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame(report["features"]).to_csv(args.output_dir / "feature_drift.csv", index=False)
    print(json.dumps({k: v for k, v in report.items() if k != "features"}, indent=2))


if __name__ == "__main__":
    main()
