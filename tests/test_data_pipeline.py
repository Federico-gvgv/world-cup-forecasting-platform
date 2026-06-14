from __future__ import annotations

import pandas as pd

from wc_forecast.data.clean import clean_results
from wc_forecast.data.split import chronological_split

def test_clean_results_creates_match_outcome() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "home_team": ["Team A", "Team B", "Team C"],
            "away_team": ["Team D", "Team E", "Team F"],
            "home_score": [2, 1, 0],
            "away_score": [1, 1, 3],
            "tournament": ["Friendly", "Friendly", "Friendly"],
            "neutral": [False, False, True],
        }
    )

    clean = clean_results(raw)

    assert list(clean["match_outcome"]) == ["HOME_WIN", "DRAW", "AWAY_WIN"]

def test_clean_results_sorts_by_date() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2020-01-03", "2020-01-01", "2020-01-02"],
            "home_team": ["Team C", "Team A", "Team B"],
            "away_team": ["Team F", "Team D", "Team E"],
            "home_score": [0, 2, 1],
            "away_score": [3, 1, 1],
        }
    )

    clean = clean_results(raw)

    assert clean["date"].is_monotonic_increasing

def test_chronological_split_has_no_overlap() -> None:
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2017-01-01",
                    "2017-06-01",
                    "2018-01-01",
                    "2020-01-01",
                    "2022-01-01",
                    "2023-01-01",
                ]
            ),
            "home_team": ["A", "B", "C", "D", "E", "F"],
            "away_team": ["G", "H", "I", "J", "K", "L"],
            "home_score": [1, 2, 0, 3, 1, 2],
            "away_score": [0, 2, 1, 1, 1, 3],
            "match_outcome": [
                "HOME_WIN",
                "DRAW",
                "AWAY_WIN",
                "HOME_WIN",
                "DRAW",
                "AWAY_WIN",
            ],
        }
    )

    train, validation, test = chronological_split(matches)

    assert train["date"].max() < validation["date"].min()
    assert validation["date"].max() < test["date"].min()

def test_chronological_split_dates_are_correct() -> None:
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2017-12-31",
                    "2018-01-01",
                    "2021-12-31",
                    "2022-01-01",
                ]
            ),
            "home_team": ["A", "B", "C", "D"],
            "away_team": ["E", "F", "G", "H"],
            "home_score": [1, 1, 1, 1],
            "away_score": [0, 0, 0, 0],
            "match_outcome": [
                "HOME_WIN",
                "HOME_WIN",
                "HOME_WIN",
                "HOME_WIN",
            ],
        }
    )

    train, validation, test = chronological_split(matches)

    assert train["date"].max() < pd.Timestamp("2018-01-01")
    assert validation["date"].max() >= pd.Timestamp("2018-01-01")
    assert validation["date"].max() < pd.Timestamp("2022-01-01")
    assert test["date"].min() >= pd.Timestamp("2022-01-01")