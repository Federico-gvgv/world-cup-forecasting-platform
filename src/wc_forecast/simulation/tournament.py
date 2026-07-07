from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from wc_forecast.simulation.group_stage import (
    GroupMatchup,
    simulate_group_stage_with_provider,
)
from wc_forecast.simulation.knockout import simulate_knockout_match
from wc_forecast.simulation.match import MatchProbabilities


ProbabilityProvider = Callable[[str, str], MatchProbabilities]


ADVANCEMENT_COLUMNS = [
    "group_qualified",
    "round_of_32",
    "round_of_16",
    "quarter_final",
    "semi_final",
    "final",
    "tournament_win",
]


@dataclass(frozen=True)
class KnockoutSlot:
    group_name: str
    position: int


@dataclass(frozen=True)
class KnockoutPairing:
    team_a: KnockoutSlot
    team_b: KnockoutSlot


@dataclass(frozen=True)
class KnockoutSeed:
    group_name: str
    position: int
    team: str


@dataclass(frozen=True)
class TournamentConfig:
    group_matchups: dict[str, list[GroupMatchup]]
    knockout_pairings: list[KnockoutPairing]
    tournament_format: str = "explicit"
    n_simulations: int = 1_000
    random_seed: int = 42


def _all_teams(group_matchups: dict[str, list[GroupMatchup]]) -> list[str]:
    teams = {
        team
        for matchups in group_matchups.values()
        for matchup in matchups
        for team in [matchup.home_team, matchup.away_team]
    }

    return sorted(teams)


def _initialise_achievements(teams: list[str]) -> dict[str, dict[str, float]]:
    return {
        team: {column: 0.0 for column in ADVANCEMENT_COLUMNS}
        for team in teams
    }


def _mark_teams(
    achievements: dict[str, dict[str, float]],
    teams: list[str],
    column: str,
) -> None:
    for team in teams:
        achievements[team][column] = 1.0


def _knockout_stage_column(n_teams_remaining: int) -> str | None:
    """
    Map number of teams alive at the start of a knockout round to
    the stage they have reached.
    """
    mapping = {
        32: "round_of_32",
        16: "round_of_16",
        8: "quarter_final",
        4: "semi_final",
        2: "final",
    }

    return mapping.get(n_teams_remaining)


def rank_third_place_teams(third_place_rows: list[dict[str, object]]) -> list[str]:
    """
    Rank third-place teams across groups using the V1 group-stage tie-breaks.

    The simplified 2026-style format uses points, then wins, then the random
    group-stage tie-break value. Goal difference is intentionally not included
    until scoreline simulation is added.
    """
    ranked_rows = sorted(
        third_place_rows,
        key=lambda row: (
            int(row["points"]),
            int(row["wins"]),
            float(row.get("tie_break", 0.0)),
        ),
        reverse=True,
    )

    return [str(row["team"]) for row in ranked_rows]


def _generate_simplified_round_of_32_pairs(
    seeds: list[KnockoutSeed],
) -> list[tuple[str, str]]:
    """
    Build a deterministic simplified 32-team knockout bracket.

    Seeds are expected in strength order: group winners first, runners-up
    second, best third-place teams last. The bracket pairs early seeds against
    late seeds while making a simple deterministic effort to avoid immediate
    same-group rematches.
    """
    if len(seeds) != 32:
        raise ValueError(
            "world_cup_2026 format requires exactly 32 knockout qualifiers."
        )

    top_half = seeds[:16]
    bottom_half = list(reversed(seeds[16:]))
    pairs: list[tuple[str, str]] = []

    for top_seed in top_half:
        opponent_index = next(
            (
                index
                for index, candidate in enumerate(bottom_half)
                if candidate.group_name != top_seed.group_name
            ),
            0,
        )
        opponent = bottom_half.pop(opponent_index)
        pairs.append((top_seed.team, opponent.team))

    return pairs


def _build_world_cup_2026_knockout_pairs(
    group_standings: dict[str, pd.DataFrame],
) -> tuple[list[tuple[str, str]], list[str]]:
    winner_seeds: list[KnockoutSeed] = []
    runner_up_seeds: list[KnockoutSeed] = []
    third_place_rows: list[dict[str, object]] = []

    for group_name, standings in group_standings.items():
        for row in standings.itertuples(index=False):
            if int(row.position) == 1:
                winner_seeds.append(
                    KnockoutSeed(
                        group_name=group_name,
                        position=1,
                        team=str(row.team),
                    )
                )
            elif int(row.position) == 2:
                runner_up_seeds.append(
                    KnockoutSeed(
                        group_name=group_name,
                        position=2,
                        team=str(row.team),
                    )
                )
            elif int(row.position) == 3:
                third_place_rows.append(
                    {
                        "group_name": group_name,
                        "position": 3,
                        "team": str(row.team),
                        "points": int(row.points),
                        "wins": int(row.wins),
                        "tie_break": float(row.tie_break),
                    }
                )

    ranked_third_place_teams = rank_third_place_teams(third_place_rows)
    third_place_lookup = {
        str(row["team"]): row
        for row in third_place_rows
    }

    third_place_seeds = [
        KnockoutSeed(
            group_name=str(third_place_lookup[team]["group_name"]),
            position=3,
            team=team,
        )
        for team in ranked_third_place_teams[:8]
    ]

    seeds = [
        *winner_seeds,
        *runner_up_seeds,
        *third_place_seeds,
    ]
    knockout_pairs = _generate_simplified_round_of_32_pairs(seeds)
    knockout_teams = [seed.team for seed in seeds]

    return knockout_pairs, knockout_teams


def _build_explicit_knockout_pairs(
    config: TournamentConfig,
    slot_to_team: dict[tuple[str, int], str],
) -> tuple[list[tuple[str, str]], list[str]]:
    current_pairs: list[tuple[str, str]] = []
    initial_knockout_teams: list[str] = []

    for pairing in config.knockout_pairings:
        team_a = slot_to_team[
            (pairing.team_a.group_name, pairing.team_a.position)
        ]
        team_b = slot_to_team[
            (pairing.team_b.group_name, pairing.team_b.position)
        ]

        current_pairs.append((team_a, team_b))
        initial_knockout_teams.extend([team_a, team_b])

    return current_pairs, initial_knockout_teams


def simulate_single_tournament(
    config: TournamentConfig,
    probability_provider: ProbabilityProvider,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    """
    Simulate one full tournament.

    This function is model-agnostic. It only requires a probability provider
    that can return W/D/L probabilities for any two teams.
    """
    teams = _all_teams(config.group_matchups)
    achievements = _initialise_achievements(teams)

    slot_to_team: dict[tuple[str, int], str] = {}
    group_standings: dict[str, pd.DataFrame] = {}

    for group_name, matchups in config.group_matchups.items():
        standings = simulate_group_stage_with_provider(
            matchups=matchups,
            probability_provider=probability_provider,
            rng=rng,
        )
        group_standings[group_name] = standings

        for row in standings.itertuples(index=False):
            slot_to_team[(group_name, int(row.position))] = str(row.team)

    if config.tournament_format == "world_cup_2026":
        current_pairs, initial_knockout_teams = (
            _build_world_cup_2026_knockout_pairs(group_standings)
        )
    elif config.tournament_format == "explicit":
        current_pairs, initial_knockout_teams = _build_explicit_knockout_pairs(
            config=config,
            slot_to_team=slot_to_team,
        )
    else:
        raise ValueError(f"Unsupported tournament format: {config.tournament_format}")

    _mark_teams(
        achievements=achievements,
        teams=initial_knockout_teams,
        column="group_qualified",
    )

    initial_stage = _knockout_stage_column(len(initial_knockout_teams))
    if initial_stage is not None:
        _mark_teams(
            achievements=achievements,
            teams=initial_knockout_teams,
            column=initial_stage,
        )

    while current_pairs:
        winners: list[str] = []

        for team_a, team_b in current_pairs:
            probabilities = probability_provider(team_a, team_b)

            winner = simulate_knockout_match(
                team_a=team_a,
                team_b=team_b,
                probabilities=probabilities,
                rng=rng,
            )

            winners.append(winner)

        if len(winners) == 1:
            achievements[winners[0]]["tournament_win"] = 1.0
            break

        reached_stage = _knockout_stage_column(len(winners))
        if reached_stage is not None:
            _mark_teams(
                achievements=achievements,
                teams=winners,
                column=reached_stage,
            )

        if len(winners) % 2 != 0:
            raise ValueError(
                "Knockout simulation produced an odd number of winners. "
                "Cannot construct the next round."
            )

        current_pairs = list(
            zip(
                winners[::2],
                winners[1::2],
                strict=True,
            )
        )

    return achievements


def simulate_tournament(
    config: TournamentConfig,
    probability_provider: ProbabilityProvider,
) -> pd.DataFrame:
    """
    Run many tournament simulations and estimate advancement probabilities.
    """
    if config.n_simulations <= 0:
        raise ValueError("n_simulations must be positive.")

    teams = _all_teams(config.group_matchups)
    cumulative_counts = _initialise_achievements(teams)

    rng = np.random.default_rng(config.random_seed)

    for _ in range(config.n_simulations):
        achievements = simulate_single_tournament(
            config=config,
            probability_provider=probability_provider,
            rng=rng,
        )

        for team in teams:
            for column in ADVANCEMENT_COLUMNS:
                cumulative_counts[team][column] += achievements[team][column]

    rows: list[dict[str, float | str]] = []

    for team in teams:
        row: dict[str, float | str] = {"team": team}

        for column in ADVANCEMENT_COLUMNS:
            row[f"{column}_probability"] = (
                cumulative_counts[team][column] / config.n_simulations
            )

        rows.append(row)

    results = pd.DataFrame(rows)

    return results.sort_values(
        by="tournament_win_probability",
        ascending=False,
    ).reset_index(drop=True)
