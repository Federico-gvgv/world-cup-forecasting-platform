from __future__ import annotations

import numpy as np
import pandas as pd

from wc_forecast.models.evaluate import OUTCOME_CLASSES


def build_prediction_report(
    matches: pd.DataFrame,
    uncalibrated_proba: np.ndarray,
    calibrated_proba: np.ndarray,
) -> pd.DataFrame:
    """
    Build a row-level prediction report for the final test set.
    """
    report_columns = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "match_outcome",
    ]

    prediction_report = matches[report_columns].copy()

    for class_index, class_name in enumerate(OUTCOME_CLASSES):
        clean_class_name = class_name.lower()

        prediction_report[f"uncalibrated_p_{clean_class_name}"] = (
            uncalibrated_proba[:, class_index]
        )
        prediction_report[f"calibrated_p_{clean_class_name}"] = calibrated_proba[
            :, class_index
        ]

    uncalibrated_pred_indices = np.argmax(uncalibrated_proba, axis=1)
    calibrated_pred_indices = np.argmax(calibrated_proba, axis=1)

    prediction_report["uncalibrated_prediction"] = [
        OUTCOME_CLASSES[index] for index in uncalibrated_pred_indices
    ]
    prediction_report["calibrated_prediction"] = [
        OUTCOME_CLASSES[index] for index in calibrated_pred_indices
    ]

    prediction_report["uncalibrated_confidence"] = np.max(
        uncalibrated_proba,
        axis=1,
    )
    prediction_report["calibrated_confidence"] = np.max(
        calibrated_proba,
        axis=1,
    )

    return prediction_report
