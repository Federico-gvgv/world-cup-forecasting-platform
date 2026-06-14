from __future__ import annotations

from pathlib import Path

import pandas as pd

from wc_forecast.data.clean import clean_results
from wc_forecast.data.split import chronological_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "results.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

def main() -> None:
    """
    Prepare raw match results for modelling.

    Input:
        data/raw/results.csv
    
    Outputs:
        data/processed/train.csv
        data/processed/validation.csv
        data/processed/test.csv
        data/processed/matches_clean.csv
    """
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data file not found: {RAW_DATA_PATH}")
    
    print(f"Loading raw data from {RAW_DATA_PATH}")
    raw_matches = pd.read_csv(RAW_DATA_PATH)

    print("Cleaning matches...")
    clean_matches = clean_results(raw_matches)

    print("Creating chronological splits...")
    train, validation, test = chronological_split(clean_matches)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    clean_matches.to_csv(PROCESSED_DATA_DIR / "matches_clean.csv", index=False)
    train.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    validation.to_csv(PROCESSED_DATA_DIR / "validation.csv", index=False)
    test.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

    print("Saved processed files:")
    print(f"  - {PROCESSED_DATA_DIR / 'matches_clean.csv'}")
    print(f"  - {PROCESSED_DATA_DIR / 'train.csv'}")
    print(f"  - {PROCESSED_DATA_DIR / 'validation.csv'}")
    print(f"  - {PROCESSED_DATA_DIR / 'test.csv'}")

    print()
    print("Split sizes:")
    print(f"Train:      {len(train):,}")
    print(f"Validation: {len(validation):,}")
    print(f"Test:       {len(test):,}")

if __name__ == "__main__":
    main()