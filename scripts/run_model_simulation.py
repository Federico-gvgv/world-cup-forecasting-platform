from __future__ import annotations

from itertools import combinations
from pathlib import Path

from wc_forecast.simulation.group_stage import GroupMatchup
from wc_forecast.simulation.model_provider import (
    build_model_backed_probability_provider,
)
from wc_forecast.simulation.tournament import (
    KnockoutPairing,
    KnockoutSlot,
    TournamentConfig,
    simulate_tournament,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
SIMULATIONS_DIR = REPORTS_DIR / "simulations"

TRAIN_PATH = PROCESSED_DATA_DIR / "train_features.csv"
VALIDATION_PATH = PROCESSED_DATA_DIR / "validation_features.csv"
MATCHES_FEATURES_PATH = PROCESSED_DATA_DIR / "matches_with_features.csv"


def make_round_robin_matchups(
    teams: list[str],
) -> list[GroupMatchup]:
    matchups = []

    for team_a, team_b in combinations(teams, 2):
        matchups.append(
            GroupMatchup(
                home_team=team_a,
                away_team=team_b,
            )
        )

    return matchups


def main() -> None:
    for path in [TRAIN_PATH, VALIDATION_PATH, MATCHES_FEATURES_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find {path}. Run the full feature pipeline first."
            )

    probability_provider = build_model_backed_probability_provider(
        train_path=TRAIN_PATH,
        validation_path=VALIDATION_PATH,
        matches_with_features_path=MATCHES_FEATURES_PATH,
        use_calibration=True,
        neutral=True,
    )

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
        probability_provider=probability_provider,
    )

    print("Model-backed tournament simulation results:")
    print(results.to_string(index=False))

    SIMULATIONS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = SIMULATIONS_DIR / "model_backed_example_simulation.csv"
    results.to_csv(output_path, index=False)

    print()
    print(f"Saved simulation results to {output_path}")


if __name__ == "__main__":
    main()
