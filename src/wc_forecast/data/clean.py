from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
}

def create_match_outcome(row: pd.Series) -> str:
    """
    Create the 3-class target variable for the match outcome.

    Returns:
        HOME_WIN if the home team scored more goals.
        AWAY_WIN if the away team scored more goals.
        DRAW if both teams scored the same number of goals.
    """
    if row["home_score"] > row["away_score"]:
        return "HOME_WIN"
    elif row["home_score"] < row["away_score"]:
        return "AWAY_WIN"
    else:
        return "DRAW"
    
def clean_results(raw_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw international football results.

    This function:
    - standardises column names
    - parses dates
    - ensures scores are numeric
    - remove rows with missing required values
    - standardises team names
    - creates a match_outcome target column
    - sorts matches chronologically    
    """
    matches = raw_matches.copy()

    # Standardise column names, e.g. "Home Team" -> "home_team"
    matches.columns = (
        matches.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    missing_columns = REQUIRED_COLUMNS - set(matches.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Parse date column
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")

    # Convert scores to numbers
    matches["home_score"] = pd.to_numeric(matches["home_score"], errors="coerce")
    matches["away_score"] = pd.to_numeric(matches["away_score"], errors="coerce")

    # Drop rows where key information is missing
    matches = matches.dropna(
        subset=["date", "home_team", "away_team", "home_score", "away_score"]
    )

    # Scores should be integers
    matches["home_score"] = matches["home_score"].astype(int)
    matches["away_score"] = matches["away_score"].astype(int)

    # Clean team names
    matches["home_team"] = matches["home_team"].str.strip()
    matches["away_team"] = matches["away_team"].str.strip()

    # If neutral column exists, make sure it is boolean
    if "neutral" in matches.columns:
        matches["neutral"] = (
            matches["neutral"]
            .replace(
                {
                    "TRUE": True,
                    "True": True,
                    "true": True,
                    "FALSE": False,
                    "False": False,
                    "false": False,
                    1: True,
                    0: False,
                }
            )
            .fillna(False)
            .astype(bool)
        )
    
    # Create target column
    matches["match_outcome"] = matches.apply(create_match_outcome, axis=1)

    # Sort matches chronologically
    matches = matches.sort_values("date").reset_index(drop=True)

    return matches