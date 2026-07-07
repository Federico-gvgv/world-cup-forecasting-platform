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
    rank_third_place_teams,
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


def test_rank_third_place_teams_uses_points_wins_and_tie_break() -> None:
    third_place_rows = [
        {"team": "Team A", "points": 4, "wins": 1, "tie_break": 0.10},
        {"team": "Team B", "points": 5, "wins": 1, "tie_break": 0.20},
        {"team": "Team C", "points": 4, "wins": 2, "tie_break": 0.05},
        {"team": "Team D", "points": 4, "wins": 1, "tie_break": 0.90},
    ]

    ranked_teams = rank_third_place_teams(third_place_rows)

    assert ranked_teams == ["Team B", "Team C", "Team D", "Team A"]


def test_world_cup_2026_format_sends_32_teams_to_knockout() -> None:
    group_matchups = {}

    for group_index in range(12):
        group_name = chr(ord("A") + group_index)
        teams = [
            f"Team {group_name}{team_index}"
            for team_index in range(1, 5)
        ]
        group_matchups[group_name] = [
            GroupMatchup(home_team=home_team, away_team=away_team)
            for home_index, home_team in enumerate(teams)
            for away_team in teams[home_index + 1:]
        ]

    config = TournamentConfig(
        group_matchups=group_matchups,
        knockout_pairings=[],
        tournament_format="world_cup_2026",
        n_simulations=1,
        random_seed=42,
    )

    results = simulate_tournament(
        config=config,
        probability_provider=_fixed_home_win_provider,
    )

    assert len(results) == 48
    assert results["group_qualified_probability"].sum() == 32.0
    assert results["round_of_32_probability"].sum() == 32.0
    assert results["round_of_16_probability"].sum() == 16.0
    assert results["quarter_final_probability"].sum() == 8.0
    assert results["semi_final_probability"].sum() == 4.0
    assert results["final_probability"].sum() == 2.0
    assert results["tournament_win_probability"].sum() == 1.0
