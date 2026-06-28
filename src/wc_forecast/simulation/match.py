from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wc_forecast.models.evaluate import OUTCOME_CLASSES

VALID_OUTCOMES = set(OUTCOME_CLASSES)


@dataclass
class MatchProbabilities:
    home_win: float
    draw: float
    away_win: float

    def __post_init__(self) -> None:
        probabilities = self.as_array()

        if np.any(probabilities < 0.0):
            raise ValueError("Probabilities must be non-negative.")

        total_probability = probabilities.sum()

        if not np.isclose(total_probability, 1.0):
            raise ValueError(
                f"Probabilities must sum to 1. Got {total_probability:.4f}."
            )

    def as_array(self) -> np.ndarray:
        return np.array([self.away_win, self.draw, self.home_win])


@dataclass(frozen=True)
class SimulatedMatch:
    home_team: str
    away_team: str
    outcome: str

    def __post_init__(self) -> None:
        if self.outcome not in VALID_OUTCOMES:
            raise ValueError(f"Invalid outcome: {self.outcome}.")


def simulate_match(
    home_team: str,
    away_team: str,
    probabilities: MatchProbabilities,
    rng: np.random.Generator,
) -> SimulatedMatch:
    """
    Simulate a match outcome from win/draw/loss probabilities.
    """
    outcome = rng.choice(
        OUTCOME_CLASSES,
        p=probabilities.as_array(),
    )

    return SimulatedMatch(
        home_team=home_team,
        away_team=away_team,
        outcome=str(outcome),
    )
