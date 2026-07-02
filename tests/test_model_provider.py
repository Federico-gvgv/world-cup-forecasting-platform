from __future__ import annotations

import numpy as np
import pandas as pd

from wc_forecast.models.sklearn_models import build_logistic_regression_model
from wc_forecast.simulation.model_provider import (
    ModelBackedProbabilityProvider,
    TeamFeatureStore,
)
from wc_forecast.simulation.match import MatchProbabilities


def _example_matches_with_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "home_team": ["Team A", "Team B"],
            "away_team": ["Team B", "Team C"],
            "home_score": [2, 1],
            "away_score": [0, 1],
            "home_elo_post": [1510.0, 1498.0],
            "away_elo_post": [1490.0, 1502.0],
        }
    )


def _example_training_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "elo_diff": [-150, -100, -50, 0, 50, 100, 150, 200, -200],
            "home_expected_score": [
                0.30,
                0.35,
                0.43,
                0.50,
                0.57,
                0.65,
                0.70,
                0.75,
                0.25,
            ],
            "neutral": [True, True, False, False, True, False, True, False, True],
            "recent_points_diff": [-2, -1, -1, 0, 1, 1, 2, 2, -2],
            "recent_goals_for_diff": [
                -1.5,
                -1.0,
                -0.5,
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                -2.0,
            ],
            "recent_goals_against_diff": [
                1.5,
                1.0,
                0.5,
                0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                2.0,
            ],
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


def test_team_feature_store_creates_future_match_features() -> None:
    matches = _example_matches_with_features()

    feature_store = TeamFeatureStore.from_matches(matches)

    features = feature_store.make_match_features(
        team_a="Team A",
        team_b="Team B",
        neutral=True,
    )

    expected_columns = {
        "elo_diff",
        "home_expected_score",
        "neutral",
        "recent_points_diff",
        "recent_goals_for_diff",
        "recent_goals_against_diff",
        "recent_goal_diff_diff",
    }

    assert expected_columns.issubset(set(features.columns))
    assert len(features) == 1


def test_model_backed_probability_provider_returns_probabilities() -> None:
    train = _example_training_data()

    model = build_logistic_regression_model()
    model.fit(train)

    feature_store = TeamFeatureStore.from_matches(
        _example_matches_with_features()
    )

    provider = ModelBackedProbabilityProvider(
        model=model,
        feature_store=feature_store,
        neutral=True,
    )

    probabilities = provider(
        team_a="Team A",
        team_b="Team B",
    )

    assert isinstance(probabilities, MatchProbabilities)

    probability_sum = (
        probabilities.home_win
        + probabilities.draw
        + probabilities.away_win
    )

    np.testing.assert_allclose(probability_sum, 1.0)