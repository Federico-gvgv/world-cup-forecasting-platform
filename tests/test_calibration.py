from __future__ import annotations

import numpy as np
import pandas as pd

from wc_forecast.models.calibration import (
    ProbabilityCalibrator,
    expected_calibration_error,
)


def test_expected_calibration_error_perfect_predictions_is_zero() -> None:
    y_true = pd.Series(["HOME_WIN", "DRAW", "AWAY_WIN"])

    y_proba = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    ece = expected_calibration_error(
        y_true=y_true,
        y_proba=y_proba,
        n_bins=10,
    )

    assert ece == 0.0


def test_probability_calibrator_outputs_valid_probabilities() -> None:
    y_true = pd.Series(
        [
            "HOME_WIN",
            "HOME_WIN",
            "DRAW",
            "DRAW",
            "AWAY_WIN",
            "AWAY_WIN",
        ]
    )

    base_proba = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.60, 0.25, 0.15],
            [0.25, 0.55, 0.20],
            [0.20, 0.60, 0.20],
            [0.10, 0.25, 0.65],
            [0.15, 0.20, 0.65],
        ]
    )

    calibrator = ProbabilityCalibrator()
    calibrator.fit(
        y_proba=base_proba,
        y_true=y_true,
    )

    calibrated_proba = calibrator.predict_proba(base_proba)

    assert calibrated_proba.shape == (len(base_proba), 3)

    np.testing.assert_allclose(
        calibrated_proba.sum(axis=1),
        np.ones(len(base_proba)),
    )


def test_probability_calibrator_changes_probabilities() -> None:
    y_true = pd.Series(
        [
            "HOME_WIN",
            "HOME_WIN",
            "DRAW",
            "DRAW",
            "AWAY_WIN",
            "AWAY_WIN",
        ]
    )

    base_proba = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.60, 0.25, 0.15],
            [0.25, 0.55, 0.20],
            [0.20, 0.60, 0.20],
            [0.10, 0.25, 0.65],
            [0.15, 0.20, 0.65],
        ]
    )

    calibrator = ProbabilityCalibrator()
    calibrator.fit(
        y_proba=base_proba,
        y_true=y_true,
    )

    calibrated_proba = calibrator.predict_proba(base_proba)

    assert not np.allclose(base_proba, calibrated_proba)
