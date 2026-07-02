from __future__ import annotations

from itertools import combinations
from pathlib import Path

from wc_forecast.simulation.group_stage import GroupMatchup
from wc_forecast.simulation.match import MatchProbabilities
from wc_forecast.simulation.tournament import (
    KnockoutPairing,
    KnockoutSlot,
    TournamentConfig,
    simulate_tournament,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
SIMULATIONS_DIR = REPORTS_DIR / "simulations"


def make_round_robin_matchups(
    teams: list[str],
) -> list[GroupMatchup]:
    matchups = []

    for home_team, away_team in combinations(teams, 2):
        matchups.append(
            GroupMatchup(
                home_team=home_team,
                away_team=away_team,
            )
        )

    return matchups


def fixed_probability_provider(
    team_a: str,
    team_b: str,
) -> MatchProbabilities:
    """
    Temporary probability provider for simulator testing.

    Later, this will call the trained forecasting model.
    """
    return MatchProbabilities(
        home_win=0.45,
        draw=0.25,
        away_win=0.30,
    )


def main() -> None:
    group_matchups = {
        "A": make_round_robin_matchups(
            teams=[
                "Argentina",
                "Mexico",
                "Poland",
                "Saudi Arabia",
            ],
        ),
        "B": make_round_robin_matchups(
            teams=[
                "France",
                "Denmark",
                "Tunisia",
                "Australia",
            ],
        ),
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
        n_simulations=10_000,
        random_seed=42,
    )

    results = simulate_tournament(
        config=config,
        probability_provider=fixed_probability_provider,
    )

    print("Tournament simulation results:")
    print(results.to_string(index=False))

    SIMULATIONS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = SIMULATIONS_DIR / "example_tournament_simulation.csv"
    results.to_csv(output_path, index=False)

    print()
    print(f"Saved simulation results to {output_path}")


if __name__ == "__main__":
    main()