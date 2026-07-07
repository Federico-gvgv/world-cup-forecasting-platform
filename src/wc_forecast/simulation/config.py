from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import yaml

from wc_forecast.simulation.group_stage import GroupMatchup
from wc_forecast.simulation.tournament import (
    KnockoutPairing,
    KnockoutSlot,
    TournamentConfig,
)


def make_round_robin_matchups(teams: list[str]) -> list[GroupMatchup]:
    """
    Create all pairwise group-stage matchups for a group.
    """
    return [
        GroupMatchup(
            home_team=team_a,
            away_team=team_b,
        )
        for team_a, team_b in combinations(teams, 2)
    ]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find config file: {path}")

    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)

    if not isinstance(loaded, dict):
        raise ValueError("Tournament config must be a YAML mapping.")

    return loaded


def load_tournament_config(path: Path) -> TournamentConfig:
    """
    Load a tournament simulation config from YAML.

    Expected structure:

    format: world_cup_2026  # optional; defaults to explicit

    simulation:
      n_simulations: 10000
      random_seed: 42

    groups:
      A:
        - Team 1
        - Team 2

    knockout_pairings:  # required for explicit mode only
      - team_a:
          group: A
          position: 1
        team_b:
          group: B
          position: 2
    """
    config = _load_yaml(path)

    tournament_format = str(config.get("format", "explicit"))
    simulation_config = config.get("simulation", {})
    groups = config.get("groups")
    knockout_pairings_config = config.get("knockout_pairings", [])

    if not isinstance(groups, dict):
        raise ValueError("Config must contain a 'groups' mapping.")

    if tournament_format not in {"explicit", "world_cup_2026"}:
        raise ValueError(f"Unsupported tournament format: {tournament_format}")

    if tournament_format == "explicit" and "knockout_pairings" not in config:
        raise ValueError("Explicit config must contain a 'knockout_pairings' list.")

    if not isinstance(knockout_pairings_config, list):
        raise ValueError("'knockout_pairings' must be a list.")

    if tournament_format == "world_cup_2026" and len(groups) != 12:
        raise ValueError("world_cup_2026 format requires exactly 12 groups.")

    group_matchups: dict[str, list[GroupMatchup]] = {}

    for group_name, teams in groups.items():
        if not isinstance(teams, list):
            raise ValueError(f"Group {group_name} must be a list of teams.")

        if len(teams) < 2:
            raise ValueError(f"Group {group_name} must contain at least two teams.")

        if tournament_format == "world_cup_2026" and len(teams) != 4:
            raise ValueError(
                f"Group {group_name} must contain exactly four teams "
                "for world_cup_2026 format."
            )

        group_matchups[str(group_name)] = make_round_robin_matchups(
            teams=[str(team) for team in teams]
        )

    knockout_pairings: list[KnockoutPairing] = []

    for pairing in knockout_pairings_config:
        team_a = pairing["team_a"]
        team_b = pairing["team_b"]

        knockout_pairings.append(
            KnockoutPairing(
                team_a=KnockoutSlot(
                    group_name=str(team_a["group"]),
                    position=int(team_a["position"]),
                ),
                team_b=KnockoutSlot(
                    group_name=str(team_b["group"]),
                    position=int(team_b["position"]),
                ),
            )
        )

    return TournamentConfig(
        group_matchups=group_matchups,
        knockout_pairings=knockout_pairings,
        tournament_format=tournament_format,
        n_simulations=int(simulation_config.get("n_simulations", 1_000)),
        random_seed=int(simulation_config.get("random_seed", 42)),
    )


def load_simulation_options(path: Path) -> dict[str, Any]:
    """
    Load options that are not part of TournamentConfig itself, such as
    calibration and neutral-site settings.
    """
    config = _load_yaml(path)
    simulation_config = config.get("simulation", {})

    if not isinstance(simulation_config, dict):
        raise ValueError("'simulation' must be a mapping.")

    return {
        "use_calibration": bool(simulation_config.get("use_calibration", True)),
        "neutral": bool(simulation_config.get("neutral", True)),
    }
