from __future__ import annotations

import numpy as np

from wc_forecast.simulation.match import MatchProbabilities, simulate_match


def simulate_knockout_match(
    team_a: str,
    team_b: str,
    probabilities: MatchProbabilities,
    rng: np.random.Generator,
) -> str:
    """
    Simulate a knockout match.

    If the W/D/L model produces a draw, we resolve it with a simple random penalty shootout.

    Later, this can be replaced with a scoreline model or penalty model.
    """
    simulated_match = simulate_match(
        home_team=team_a,
        away_team=team_b,
        probabilities=probabilities,
        rng=rng,
    )

    if simulated_match.outcome == "HOME_WIN":
        return team_a

    if simulated_match.outcome == "AWAY_WIN":
        return team_b

    return str(rng.choice([team_a, team_b]))
