from __future__ import annotations

import numpy as np
import pandas as pd

from wc_forecast.models.baselines import (
    EloLogisticRegressionBaseline,
    EmpiricalOutcomeBaseline,
)
from wc_forecast.models.evaluate import (
    OUTCOME_CLASSES,
    evaluate_probability_predictions,
    multiclass_brier_score,
)


def test_empirical_baseline_predicts_training_distribution() -> None:
    y = pd.Series(
        [
            "HOME_WIN",
            "HOME_WIN",
            "DRAW",
            "AWAY_WIN",
        ]
    )

    model = EmpiricalOutcomeBaseline()
    model.fit(y)

    probabilities = model.predict_proba(n_rows=2)

    expected = np.array(
        [
            [0.25, 0.25, 0.5],
            [0.25, 0.25, 0.5],
        ]
    )

    np.testing.assert_allclose(probabilities, expected)


def test_multiclass_brier_score_perfect_prediction_is_zero() -> None:
    y_true = pd.Series(["HOME_WIN", "DRAW", "AWAY_WIN"])

    y_proba = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    score = multiclass_brier_score(
        y_true=y_true,
        y_proba=y_proba,
        classes=OUTCOME_CLASSES,
    )

    assert score == 0.0


def test_evaluate_probability_predictions_returns_expected_metrics() -> None:
    y_true = pd.Series(["HOME_WIN", "DRAW", "AWAY_WIN"])

    y_proba = np.array(
        [
            [0.1, 0.1, 0.8],
            [0.2, 0.6, 0.2],
            [0.7, 0.2, 0.1],
        ]
    )

    metrics = evaluate_probability_predictions(
        y_true=y_true,
        y_proba=y_proba,
    )

    assert set(metrics.keys()) == {
        "accuracy",
        "log_loss",
        "brier_score",
    }

    assert metrics["accuracy"] == 1.0
    assert metrics["log_loss"] > 0.0
    assert metrics["brier_score"] > 0.0


def test_elo_logistic_regression_baseline_outputs_probabilities() -> None:
    matches = pd.DataFrame(
        {
            "elo_diff": [-100, -50, 0, 50, 100, 150],
            "home_expected_score": [0.35, 0.43, 0.50, 0.57, 0.65, 0.70],
            "neutral": [True, True, False, False, True, False],
            "match_outcome": [
                "AWAY_WIN",
                "AWAY_WIN",
                "DRAW",
                "DRAW",
                "HOME_WIN",
                "HOME_WIN",
            ],
        }
    )

    model = EloLogisticRegressionBaseline()
    model.fit(matches)

    probabilities = model.predict_proba(matches)

    assert probabilities.shape == (len(matches), 3)

    row_sums = probabilities.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(len(matches)))
