import pandas as pd

from src.monitoring import compare_frames, population_stability_index


def test_psi_is_zero_for_identical_distribution():
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert population_stability_index(series, series) == 0.0


def test_shifted_feature_is_flagged():
    reference = pd.DataFrame({"Amount": [1, 2, 3, 4, 5] * 20, "Class": [0] * 100})
    current = pd.DataFrame({"Amount": [100, 110, 120, 130, 140] * 20, "Class": [0] * 100})
    report = compare_frames(reference, current)
    assert report["drift_detected"] is True
    assert "Amount" in report["drifted_features"]
    assert all(item["feature"] != "Class" for item in report["features"])


def test_identical_data_does_not_flag_drift():
    frame = pd.DataFrame({"V1": list(range(100)), "Amount": list(range(100))})
    report = compare_frames(frame, frame)
    assert report["drift_detected"] is False
