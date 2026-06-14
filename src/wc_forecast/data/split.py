from __future__ import annotations

import pandas as pd

def chronological_split(
        matches: pd.DataFrame,
        train_end_date: str = "2018-01-01",
        validation_end_date: str = "2022-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split matches chronologically.
    
    Default split:
    - train: matches before 2018
    - validation: matches from 2018 to 2021
    - test: matches from 2022 onwards

    This avoids leakage from future matches into past predictions.
    """
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])

    train_end = pd.Timestamp(train_end_date)
    validation_end = pd.Timestamp(validation_end_date)

    train = matches[matches["date"] < train_end].copy()

    validation = matches[
        (matches["date"] >= train_end) & (matches["date"] < validation_end)
    ].copy()

    test = matches[matches["date"] >= validation_end].copy()

    if train.empty:
        raise ValueError("Train split is empty.")
    
    if validation.empty:
        raise ValueError("Validation split is empty.")
    
    if test.empty:
        raise ValueError("Test split is empty.")
    
    return train, validation, test