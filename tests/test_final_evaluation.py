from __future__ import annotations

import numpy as np
import pandas as pd

from wc_forecast.models.final_evaluation import build_prediction_report


def test_build_prediction_report_contains_probability_columns() -> None:
    matches = pd.DataFrame(
        {
            "date": ["2022-01-01", "2022-01-02"],
            "home_team": ["Team A", "Team B"],
            "away_team": ["Team C", "Team D"],
            "home_score": [2, 1],
            "away_score": [0, 1],
            "match_outcome": ["HOME_WIN", "DRAW"],
        }
    )

    uncalibrated_proba = np.array(
        [
            [0.10, 0.20, 0.70],
            [0.20, 0.50, 0.30],
        ]
    )

    calibrated_proba = np.array(
        [
            [0.10, 0.25, 0.65],
            [0.22, 0.46, 0.32],
        ]
    )

    report = build_prediction_report(
        matches=matches,
        uncalibrated_proba=uncalibrated_proba,
        calibrated_proba=calibrated_proba,
    )

    expected_columns = {
        "uncalibrated_p_home_win",
        "uncalibrated_p_draw",
        "uncalibrated_p_away_win",
        "calibrated_p_home_win",
        "calibrated_p_draw",
        "calibrated_p_away_win",
        "uncalibrated_prediction",
        "calibrated_prediction",
        "uncalibrated_confidence",
        "calibrated_confidence",
    }

    assert expected_columns.issubset(set(report.columns))
    assert len(report) == 2
    assert report["uncalibrated_prediction"].tolist() == ["HOME_WIN", "DRAW"]
    assert report["calibrated_prediction"].tolist() == ["HOME_WIN", "DRAW"]
