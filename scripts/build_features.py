from __future__ import annotations

from pathlib import Path

import pandas as pd

from wc_forecast.data.features import RollingFeatureConfig, add_rolling_team_features
from wc_forecast.data.split import chronological_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MATCHES_WITH_ELO_PATH = PROCESSED_DATA_DIR / "matches_with_elo.csv"


def main() -> None:
    if not MATCHES_WITH_ELO_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {MATCHES_WITH_ELO_PATH}."
            "Run build_elo_features.py first."
        )
    
    print(f"Loading matches with Elo from {MATCHES_WITH_ELO_PATH}")
    matches = pd.read_csv(MATCHES_WITH_ELO_PATH)

    print("Adding causal rolling team-form features...")
    matches_with_features = add_rolling_team_features(
        matches,
        config=RollingFeatureConfig(window_size=5)
    )

    print("Creating chronological split...")
    train, validation, test = chronological_split(matches_with_features)

    matches_with_features.to_csv(
        PROCESSED_DATA_DIR / "matches_with_features.csv",
        index=False,
    )
    train.to_csv(PROCESSED_DATA_DIR / "train_features.csv", index=False)
    validation.to_csv(PROCESSED_DATA_DIR / "validation_features.csv", index=False)
    test.to_csv(PROCESSED_DATA_DIR / "test_features.csv", index=False)

    print("Saved feature files:")
    print(f"- {PROCESSED_DATA_DIR / 'matches_with_features.csv'}")
    print(f"- {PROCESSED_DATA_DIR / 'train_features.csv'}")
    print(f"- {PROCESSED_DATA_DIR / 'validation_features.csv'}")
    print(f"- {PROCESSED_DATA_DIR / 'test_features.csv'}")

    preview_columns = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "elo_diff",
        "recent_points_diff",
        "recent_goal_diff_diff",
    ]

    print()
    print("Preview:")
    print(matches_with_features[preview_columns].tail(10))


if __name__ == "__main__":
    main()
