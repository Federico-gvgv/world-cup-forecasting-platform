from __future__ import annotations

import argparse
from pathlib import Path

from wc_forecast.simulation.config import (
    load_simulation_options,
    load_tournament_config,
)
from wc_forecast.simulation.model_provider import (
    build_model_backed_probability_provider,
)
from wc_forecast.simulation.tournament import simulate_tournament


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
SIMULATIONS_DIR = REPORTS_DIR / "simulations"

TRAIN_PATH = PROCESSED_DATA_DIR / "train_features.csv"
VALIDATION_PATH = PROCESSED_DATA_DIR / "validation_features.csv"
MATCHES_FEATURES_PATH = PROCESSED_DATA_DIR / "matches_with_features.csv"

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "world_cup_2026.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a model-backed tournament simulation."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to tournament YAML config.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=SIMULATIONS_DIR / "model_backed_simulation.csv",
        help="Path to save simulation results.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for path in [TRAIN_PATH, VALIDATION_PATH, MATCHES_FEATURES_PATH, args.config]:
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find {path}. Run the full pipeline first."
            )

    tournament_config = load_tournament_config(args.config)
    simulation_options = load_simulation_options(args.config)

    probability_provider = build_model_backed_probability_provider(
        train_path=TRAIN_PATH,
        validation_path=VALIDATION_PATH,
        matches_with_features_path=MATCHES_FEATURES_PATH,
        use_calibration=simulation_options["use_calibration"],
        neutral=simulation_options["neutral"],
    )

    results = simulate_tournament(
        config=tournament_config,
        probability_provider=probability_provider,
    )

    print("Model-backed tournament simulation results:")
    print(results.to_string(index=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    print()
    print(f"Saved simulation results to {args.output}")


if __name__ == "__main__":
    main()
