from __future__ import annotations

import numpy as np
import pandas as pd

from wc_forecast.models.sklearn_models import (
    DEFAULT_FEATURE_COLUMNS,
    build_gradient_boosting_model,
    build_logistic_regression_model,
)


def _example_training_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "elo_diff": [-150, -100, -50, 0, 50, 100, 150, 200, -200],
            "home_expected_score": [0.30, 0.35, 0.43, 0.50, 0.57, 0.65, 0.70, 0.75, 0.25],
            "neutral": [True, True, False, False, True, False, True, False, True],
            "recent_points_diff": [-2, -1, -1, 0, 1, 1, 2, 2, -2],
            "recent_goals_for_diff": [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, -2.0],
            "recent_goals_against_diff": [1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, 2.0],
            "recent_goal_diff_diff": [-3, -2, -1, 0, 1, 2, 3, 4, -4],
            "match_outcome": [
                "AWAY_WIN",
                "AWAY_WIN",
                "DRAW",
                "DRAW",
                "HOME_WIN",
                "HOME_WIN",
                "HOME_WIN",
                "HOME_WIN",
                "AWAY_WIN",
            ],
        }
    )


def test_logistic_regression_outputs_valid_probabilities() -> None:
    data = _example_training_data()

    model = build_logistic_regression_model(
        feature_columns=DEFAULT_FEATURE_COLUMNS,
    )
    model.fit(data)

    probabilities = model.predict_proba(data)

    assert probabilities.shape == (len(data), 3)
    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(len(data)),
    )


def test_gradient_boosting_outputs_valid_probabilities() -> None:
    data = _example_training_data()

    model = build_gradient_boosting_model(
        feature_columns=DEFAULT_FEATURE_COLUMNS,
    )
    model.fit(data)

    probabilities = model.predict_proba(data)

    assert probabilities.shape == (len(data), 3)
    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(len(data)),
    )