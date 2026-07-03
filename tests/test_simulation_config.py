from __future__ import annotations

from pathlib import Path

import yaml

from wc_forecast.simulation.config import (
    load_simulation_options,
    load_tournament_config,
    make_round_robin_matchups,
)


def test_make_round_robin_matchups_for_four_teams() -> None:
    teams = ["A", "B", "C", "D"]

    matchups = make_round_robin_matchups(teams)

    assert len(matchups) == 6


def test_load_tournament_config(tmp_path: Path) -> None:
    config = {
        "simulation": {
            "n_simulations": 100,
            "random_seed": 123,
            "use_calibration": True,
            "neutral": True,
        },
        "groups": {
            "A": ["Team A", "Team B"],
            "B": ["Team C", "Team D"],
        },
        "knockout_pairings": [
            {
                "team_a": {
                    "group": "A",
                    "position": 1,
                },
                "team_b": {
                    "group": "B",
                    "position": 2,
                },
            },
            {
                "team_a": {
                    "group": "B",
                    "position": 1,
                },
                "team_b": {
                    "group": "A",
                    "position": 2,
                },
            },
        ],
    }

    config_path = tmp_path / "tournament.yaml"

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)

    tournament_config = load_tournament_config(config_path)

    assert tournament_config.n_simulations == 100
    assert tournament_config.random_seed == 123
    assert set(tournament_config.group_matchups.keys()) == {"A", "B"}
    assert len(tournament_config.knockout_pairings) == 2


def test_load_simulation_options(tmp_path: Path) -> None:
    config = {
        "simulation": {
            "n_simulations": 100,
            "random_seed": 123,
            "use_calibration": False,
            "neutral": True,
        },
        "groups": {
            "A": ["Team A", "Team B"],
            "B": ["Team C", "Team D"],
        },
        "knockout_pairings": [],
    }

    config_path = tmp_path / "tournament.yaml"

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)

    options = load_simulation_options(config_path)

    assert options["use_calibration"] is False
    assert options["neutral"] is True