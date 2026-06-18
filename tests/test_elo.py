from __future__ import annotations

import pandas as pd

from wc_forecast.ratings.elo import (
    EloConfig,
    EloRatingSystem,
    actual_home_score,
    add_elo_features,
    expected_score,
)


def test_expected_score_equal_ratings_is_half() -> None:
    assert expected_score(1500.0, 1500.0) == 0.5


def test_actual_home_score() -> None:
    assert actual_home_score(home_score=2, away_score=1) == 1.0
    assert actual_home_score(home_score=1, away_score=1) == 0.5
    assert actual_home_score(home_score=0, away_score=3) == 0.0


def test_elo_win_increases_winner_rating() -> None:
    rating_system = EloRatingSystem(
        config=EloConfig(
            initial_rating=1500.0,
            k_factor=20.0,
            home_advantage=0.0,
        )
    )

    result = rating_system.update_match(
        home_team="Team A",
        away_team="Team B",
        home_score=2,
        away_score=0,
        neutral=True,
    )

    assert result["home_elo_post"] > result["home_elo_pre"]
    assert result["away_elo_post"] < result["away_elo_pre"]


def test_elo_draw_between_equal_teams_does_not_change_ratings() -> None:
    rating_system = EloRatingSystem(
        config=EloConfig(
            initial_rating=1500.0,
            k_factor=20.0,
            home_advantage=0.0,
        )
    )

    result = rating_system.update_match(
        home_team="Team A",
        away_team="Team B",
        home_score=1,
        away_score=1,
        neutral=True,
    )

    assert result["home_elo_post"] == 1500.0
    assert result["away_elo_post"] == 1500.0


def test_add_elo_features_uses_pre_match_ratings() -> None:
    matches = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "home_team": ["Team A", "Team A"],
            "away_team": ["Team B", "Team C"],
            "home_score": [2, 1],
            "away_score": [0, 1],
            "neutral": [True, True],
        }
    )

    matches_with_elo = add_elo_features(
        matches,
        config=EloConfig(
            initial_rating=1500.0,
            k_factor=20.0,
            home_advantage=0.0,
        ),
    )

    first_match = matches_with_elo.iloc[0]
    second_match = matches_with_elo.iloc[1]

    assert first_match["home_elo_pre"] == 1500.0
    assert first_match["away_elo_pre"] == 1500.0

    # Team A won the first match, so its pre-match rating in the second
    # match should already be higher than the initial rating.
    assert second_match["home_elo_pre"] > 1500.0

    # Team C has not played before, so it should start at the initial rating.
    assert second_match["away_elo_pre"] == 1500.0