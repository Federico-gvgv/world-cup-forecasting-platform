from __future__ import annotations

import pandas as pd

from wc_forecast.data.features import (
    RollingFeatureConfig,
    add_rolling_team_features,
)

def test_first_match_uses_default_rolling_features() -> None:
    matches = pd.DataFrame(
        {
            "date": ["2020-01-01"],
            "home_team": ["Team A"],
            "away_team": ["Team B"],
            "home_score": [2],
            "away_score": [0],
        }
    )

    result = add_rolling_team_features(matches)

    first_match = result.iloc[0]

    assert first_match["home_recent_points_avg"] == 1.0
    assert first_match["away_recent_points_avg"] == 1.0
    assert first_match["home_recent_goal_diff_avg"] == 0.0
    assert first_match["away_recent_goal_diff_avg"] == 0.0


def test_second_match_uses_only_previous_matches() -> None:
    matches = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "home_team": ["Team A", "Team A"],
            "away_team": ["Team B", "Team C"],
            "home_score": [2, 1],
            "away_score": [0, 1],
        }
    )

    result = add_rolling_team_features(matches)

    first_match = result.iloc[0]
    second_match = result.iloc[1]

    assert first_match["home_recent_points_avg"] == 1.0
    assert first_match["away_recent_points_avg"] == 1.0

    assert second_match["home_recent_points_avg"] == 3.0
    assert second_match["home_recent_goals_for_avg"] == 2.0
    assert second_match["home_recent_goals_against_avg"] == 0.0
    assert second_match["home_recent_goal_diff_avg"] == 2.0

    assert second_match["away_recent_points_avg"] == 1.0


def test_rolling_window_uses_recent_matches_only() -> None:
    matches = pd.DataFrame(
        {
            "date": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
            ],
            "home_team": ["Team A", "Team A", "Team A", "Team A"],
            "away_team": ["Team B", "Team C", "Team D", "Team E"],
            "home_score": [1, 2, 3, 0],
            "away_score": [0, 0, 0, 0],
        }
    )

    result = add_rolling_team_features(
        matches,
        config=RollingFeatureConfig(window_size=2),
    )

    fourth_match = result.iloc[3]

    assert fourth_match["home_recent_goals_for_avg"] == 2.5
    assert fourth_match["home_recent_points_avg"] == 3.0