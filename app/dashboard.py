"""Local viewer for training metrics and feature-drift reports."""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Fraud MLOps Monitor", page_icon="📈", layout="wide")
st.title("Fraud Detection — MLOps Monitor")
st.caption("Distribution drift is a signal to investigate, not proof of degraded performance.")

report_type = st.radio("Report type", ["Feature drift", "Training metrics"], horizontal=True)
default_path = "monitoring_reports/drift_report.json" if report_type == "Feature drift" else "artifacts/metrics.json"
path = Path(st.text_input("Local report JSON path", default_path))

if not path.is_file():
    st.info("No report found at " + str(path) + ". Generate a report using the README commands, then refresh.")
    st.stop()

try:
    report = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    st.error("Could not read report: " + str(exc))
    st.stop()

if report_type == "Feature drift":
    c1, c2, c3 = st.columns(3)
    c1.metric("Features compared", report.get("features_compared", 0))
    c2.metric("Features flagged", len(report.get("drifted_features", [])))
    c3.metric("Reference / current rows", f"{report.get('reference_rows', 0):,} / {report.get('current_rows', 0):,}")
    if report.get("drift_detected"):
        st.warning("Distribution drift detected. Investigate data sources and evaluate labelled outcomes when available.")
    else:
        st.success("No feature crossed the configured drift thresholds in this report.")
    features = pd.DataFrame(report.get("features", []))
    if not features.empty:
        st.subheader("Feature-level statistics")
        st.dataframe(features, use_container_width=True)
        chart = px.bar(features.sort_values("psi", ascending=False), x="feature", y="psi", title="Population Stability Index by feature")
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.info("No numeric features were available for comparison.")
    st.caption(report.get("interpretation", ""))
else:
    keys = ["precision", "recall", "f1", "roc_auc", "average_precision_pr_auc", "accuracy"]
    metrics = {key: report[key] for key in keys if key in report}
    cols = st.columns(max(1, len(metrics)))
    for col, (key, value) in zip(cols, metrics.items()):
        col.metric(key.replace("_", " ").title(), f"{float(value):.4f}")
    st.json({key: value for key, value in report.items() if key != "classification_report"})
