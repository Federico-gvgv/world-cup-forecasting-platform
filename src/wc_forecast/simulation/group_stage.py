from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wc_forecast.simulation.match import MatchProbabilities, simulate_match


@dataclass(frozen=True)
class GroupFixture:
    home_team: str
    away_team: str
    probabilities: MatchProbabilities

@dataclass(frozen=True)
class GroupMatchup:
    home_team: str
    away_team: str


def _empty_standing(team: str) -> dict[str, int | str]:
    return {
        "team": team,
        "played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "points": 0,
    }


def simulate_group_stage(
    fixtures: list[GroupFixture],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Simulate group-stage fixtures and return standings.

    V1 uses W/D/L only:
    - win = 3 points
    - draw = 1 point
    - loss = 0 points

    Tie-breaks:
    - points
    - wins
    - random tie-break value

    Later, when we add scoreline simulation, we can add goal difference
    and goals scored.
    """
    teams = sorted(
        {
            team
            for fixture in fixtures
            for team in [fixture.home_team, fixture.away_team]
        }
    )

    standings = {
        team: _empty_standing(team)
        for team in teams
    }

    for fixture in fixtures:
        simulated_match = simulate_match(
            home_team=fixture.home_team,
            away_team=fixture.away_team,
            probabilities=fixture.probabilities,
            rng=rng,
        )

        home = standings[fixture.home_team]
        away = standings[fixture.away_team]

        home["played"] += 1
        away["played"] += 1

        if simulated_match.outcome == "HOME_WIN":
            home["wins"] += 1
            home["points"] += 3
            away["losses"] += 1

        elif simulated_match.outcome == "AWAY_WIN":
            away["wins"] += 1
            away["points"] += 3
            home["losses"] += 1

        else:
            home["draws"] += 1
            away["draws"] += 1
            home["points"] += 1
            away["points"] += 1

    standings_df = pd.DataFrame(standings.values())

    # Random tie-break for V1.
    standings_df["tie_break"] = rng.random(len(standings_df))

    standings_df = standings_df.sort_values(
        by=["points", "wins", "tie_break"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    standings_df["position"] = standings_df.index + 1

    return standings_df[
        [
            "position",
            "team",
            "played",
            "wins",
            "draws",
            "losses",
            "points",
            "tie_break",
        ]
    ]


def get_group_qualifiers(
    standings: pd.DataFrame,
    n_qualifiers: int = 2,
) -> list[str]:
    """
    Return the top N teams from a simulated group table.
    """
    return standings.head(n_qualifiers)["team"].tolist()


def simulate_group_stage_with_provider(
    matchups: list[GroupMatchup],
    probability_provider,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Simulate group-stage matchups using a dynamic probability provider.

    The provider is called for every fixture, so group-stage probabilities
    come from the forecasting model instead of fixed placeholders.
    """
    fixtures = [
        GroupFixture(
            home_team=matchup.home_team,
            away_team=matchup.away_team,
            probabilities=probability_provider(
                matchup.home_team,
                matchup.away_team,
            ),
        )
        for matchup in matchups
    ]

    return simulate_group_stage(
        fixtures=fixtures,
        rng=rng,
    )
