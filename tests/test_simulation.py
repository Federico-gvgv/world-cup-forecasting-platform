from __future__ import annotations

import numpy as np
import pytest

from wc_forecast.simulation.group_stage import (
    GroupFixture,
    get_group_qualifiers,
    simulate_group_stage,
)
from wc_forecast.simulation.knockout import simulate_knockout_match
from wc_forecast.simulation.match import MatchProbabilities, simulate_match


def test_match_probabilities_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        MatchProbabilities(
            home_win=0.5,
            draw=0.3,
            away_win=0.3,
        )


def test_simulate_match_with_certain_home_win() -> None:
    rng = np.random.default_rng(42)

    result = simulate_match(
        home_team="Team A",
        away_team="Team B",
        probabilities=MatchProbabilities(
            home_win=1.0,
            draw=0.0,
            away_win=0.0,
        ),
        rng=rng,
    )

    assert result.outcome == "HOME_WIN"


def test_group_stage_points_are_correct() -> None:
    rng = np.random.default_rng(42)

    fixtures = [
        GroupFixture(
            home_team="Team A",
            away_team="Team B",
            probabilities=MatchProbabilities(1.0, 0.0, 0.0),
        ),
        GroupFixture(
            home_team="Team C",
            away_team="Team D",
            probabilities=MatchProbabilities(0.0, 1.0, 0.0),
        ),
    ]

    standings = simulate_group_stage(
        fixtures=fixtures,
        rng=rng,
    )

    points_by_team = dict(
        zip(
            standings["team"],
            standings["points"],
            strict=True,
        )
    )

    assert points_by_team["Team A"] == 3
    assert points_by_team["Team B"] == 0
    assert points_by_team["Team C"] == 1
    assert points_by_team["Team D"] == 1


def test_group_qualifiers_returns_top_two() -> None:
    rng = np.random.default_rng(42)

    fixtures = [
        GroupFixture(
            home_team="Team A",
            away_team="Team B",
            probabilities=MatchProbabilities(1.0, 0.0, 0.0),
        ),
        GroupFixture(
            home_team="Team C",
            away_team="Team D",
            probabilities=MatchProbabilities(1.0, 0.0, 0.0),
        ),
        GroupFixture(
            home_team="Team A",
            away_team="Team C",
            probabilities=MatchProbabilities(1.0, 0.0, 0.0),
        ),
        GroupFixture(
            home_team="Team B",
            away_team="Team D",
            probabilities=MatchProbabilities(1.0, 0.0, 0.0),
        ),
    ]

    standings = simulate_group_stage(fixtures=fixtures, rng=rng)
    qualifiers = get_group_qualifiers(standings, n_qualifiers=2)

    assert len(qualifiers) == 2
    assert qualifiers[0] == "Team A"


def test_knockout_match_returns_winner() -> None:
    rng = np.random.default_rng(42)

    winner = simulate_knockout_match(
        team_a="Team A",
        team_b="Team B",
        probabilities=MatchProbabilities(
            home_win=1.0,
            draw=0.0,
            away_win=0.0,
        ),
        rng=rng,
    )

    assert winner == "Team A"