from __future__ import annotations

from itertools import combinations
from pathlib import Path

from wc_forecast.simulation.group_stage import GroupFixture
from wc_forecast.simulation.match import MatchProbabilities
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


def make_round_robin_fixtures(
    teams: list[str],
) -> list[GroupFixture]:
    """
    Create group fixtures.

    The probabilities here are placeholders because the model-backed
    provider is used for knockout matches in the current tournament engine.

    In the next refactor, group fixtures will also use a probability provider
    dynamically.
    """
    fixtures = []

    placeholder_probabilities = MatchProbabilities(
        home_win=0.45,
        draw=0.25,
        away_win=0.30,
    )

    for team_a, team_b in combinations(teams, 2):
        fixtures.append(
            GroupFixture(
                home_team=team_a,
                away_team=team_b,
                probabilities=placeholder_probabilities,
            )
        )

    return fixtures


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

    group_fixtures = {
        "A": make_round_robin_fixtures(
            teams=[
                "Argentina",
                "Mexico",
                "Poland",
                "Saudi Arabia",
            ],
        ),
        "B": make_round_robin_fixtures(
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
        group_fixtures=group_fixtures,
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
