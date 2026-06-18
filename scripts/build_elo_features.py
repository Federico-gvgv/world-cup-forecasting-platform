from __future__ import annotations

from pathlib import Path

import pandas as pd

from wc_forecast.data.split import chronological_split
from wc_forecast.ratings.elo import EloConfig, add_elo_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "matches_clean.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

def main() -> None:
    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Clean data file not found: {CLEAN_DATA_PATH}. "
            "RUN scripts/prepare_data.py first to generate the clean data."
        )
    
    print(f"Loading clean data from {CLEAN_DATA_PATH}")
    matches = pd.read_csv(CLEAN_DATA_PATH)

    print("Adding Elo features...")
    matches_with_elo = add_elo_features(
        matches,
        config=EloConfig(
            initial_rating=1500,
            k_factor=20,
            scale=400,
            home_advantage=75,
        ),
    )

    print("Creating chronological splits with Elo features...")
    train, validation, test = chronological_split(matches_with_elo)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    matches_with_elo.to_csv(PROCESSED_DATA_DIR / "matches_with_elo.csv", index=False)
    train.to_csv(PROCESSED_DATA_DIR / "train_with_elo.csv", index=False)
    validation.to_csv(PROCESSED_DATA_DIR / "validation_with_elo.csv", index=False)
    test.to_csv(PROCESSED_DATA_DIR / "test_with_elo.csv", index=False)

    print("Saved Elo feature files:")
    print(f"  - {PROCESSED_DATA_DIR / 'matches_with_elo.csv'}")
    print(f"  - {PROCESSED_DATA_DIR / 'train_with_elo.csv'}")
    print(f"  - {PROCESSED_DATA_DIR / 'validation_with_elo.csv'}")
    print(f"  - {PROCESSED_DATA_DIR / 'test_with_elo.csv'}")

    print()
    print("Preview:")
    columns = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_elo_pre",
        "away_elo_pre",
        "elo_diff",
        "home_expected_score",
    ]
    print(matches_with_elo[columns].tail(10))


if __name__ == "__main__":
    main()