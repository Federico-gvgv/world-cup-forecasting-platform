from __future__ import annotations

import numpy as np
import pytest

from wc_forecast.simulation.group_stage import (
    GroupFixture,
    GroupMatchup,
    get_group_qualifiers,
    simulate_group_stage,
)
from wc_forecast.simulation.knockout import simulate_knockout_match
from wc_forecast.simulation.match import MatchProbabilities, simulate_match
from wc_forecast.simulation.tournament import (
    KnockoutPairing,
    KnockoutSlot,
    TournamentConfig,
    simulate_tournament,
)


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


def _fixed_home_win_provider(
    team_a: str,
    team_b: str,
) -> MatchProbabilities:
    return MatchProbabilities(
        home_win=1.0,
        draw=0.0,
        away_win=0.0,
    )


def test_tournament_simulation_outputs_probability_table() -> None:
    group_matchups = {
        "A": [
            GroupMatchup(
                home_team="Team A",
                away_team="Team B",
            )
        ],
        "B": [
            GroupMatchup(
                home_team="Team C",
                away_team="Team D",
            )
        ],
    }

    knockout_pairings = [
        KnockoutPairing(
            team_a=KnockoutSlot(group_name="A", position=1),
            team_b=KnockoutSlot(group_name="B", position=2),
        ),
        KnockoutPairing(
            team_a=KnockoutSlot(group_name="B", position=1),
            team_b=KnockoutSlot(group_name="A", position=2),
        ),
    ]

    config = TournamentConfig(
        group_matchups=group_matchups,
        knockout_pairings=knockout_pairings,
        n_simulations=10,
        random_seed=42,
    )

    results = simulate_tournament(
        config=config,
        probability_provider=_fixed_home_win_provider,
    )

    expected_columns = {
        "team",
        "group_qualified_probability",
        "semi_final_probability",
        "final_probability",
        "tournament_win_probability",
    }

    assert expected_columns.issubset(set(results.columns))
    assert len(results) == 4


def test_deterministic_tournament_has_expected_winner() -> None:
    group_matchups = {
        "A": [
            GroupMatchup(
                home_team="Team A",
                away_team="Team B",
            )
        ],
        "B": [
            GroupMatchup(
                home_team="Team C",
                away_team="Team D",
            )
        ],
    }

    knockout_pairings = [
        KnockoutPairing(
            team_a=KnockoutSlot(group_name="A", position=1),
            team_b=KnockoutSlot(group_name="B", position=2),
        ),
        KnockoutPairing(
            team_a=KnockoutSlot(group_name="B", position=1),
            team_b=KnockoutSlot(group_name="A", position=2),
        ),
    ]

    config = TournamentConfig(
        group_matchups=group_matchups,
        knockout_pairings=knockout_pairings,
        n_simulations=10,
        random_seed=42,
    )

    results = simulate_tournament(
        config=config,
        probability_provider=_fixed_home_win_provider,
    )

    team_a = results[results["team"] == "Team A"].iloc[0]

    assert team_a["tournament_win_probability"] == 1.0