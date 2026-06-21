from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

@dataclass(frozen=True)
class RollingFeatureConfig:
    window_size: int = 5
    default_points: float = 1.0
    default_goals_for: float = 1.0
    default_goals_against: float = 1.0
    default_goal_diff: float = 0.0

def _average_recent(
        history: list[dict[str, float]],
        key: str,
        window_size: int,
        default: float,
) -> float:
    if not history:
        return default
    
    recent_matches = history[-window_size:]
    return sum(match[key] for match in recent_matches) / len(recent_matches)

def _team_recent_features(
        history: list[dict[str, float]],
        prefix: str,
        config: RollingFeatureConfig,
) -> dict[str, float]:
    return{
        f"{prefix}_recent_points_avg": _average_recent(
            history,
            key="points",
            window_size=config.window_size,
            default=config.default_points,
        ),
        f"{prefix}_recent_goals_for_avg": _average_recent(
            history,
            key="goals_for",
            window_size=config.window_size,
            default=config.default_goals_for,
        ),
        f"{prefix}_recent_goals_against_avg": _average_recent(
            history,
            key="goals_against",
            window_size=config.window_size,
            default=config.default_goals_against,
        ),
        f"{prefix}_recent_goal_diff_avg": _average_recent(
            history,
            key="goal_diff",
            window_size=config.window_size,
            default=config.default_goal_diff,
        ),
    }

def _points_for_result(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    
    if goals_for == goals_against:
        return 1
    
    return 0

def add_rolling_team_features(
        matches: pd.DataFrame,
        config: RollingFeatureConfig | None = None,
) -> pd.DataFrame:
    """
    Add causal rolling team-form features.

    For each match, features are computed using only previous matches.
    The current match result is added to history only after its features have been recorded.
    """
    config = config or RollingFeatureConfig()

    required_columns = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    }

    missing_columns = required_columns - set(matches.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values("date").reset_index(drop=True)

    team_histories: dict[str, list[dict[str, float]]] = defaultdict(list)
    feature_rows: list[dict[str, float]] = []

    for row in matches.itertuples(index=False):
        home_team = row.home_team
        away_team = row.away_team
        home_score = int(row.home_score)
        away_score = int(row.away_score)

        home_history = team_histories[home_team]
        away_history = team_histories[away_team]

        home_features = _team_recent_features(
            home_history,
            prefix="home",
            config=config,
        )
        away_features = _team_recent_features(
            away_history,
            prefix="away",
            config=config,
        )

        features = {
            **home_features,
            **away_features,
        }

        features["recent_points_diff"] = (
            features["home_recent_points_avg"]
            - features["away_recent_points_avg"]
        )
        features["recent_goals_for_diff"] = (
            features["home_recent_goals_for_avg"]
            - features["away_recent_goals_for_avg"]
        )
        features["recent_goals_against_diff"] = (
            features["home_recent_goals_against_avg"]
            - features["away_recent_goals_against_avg"]
        )
        features["recent_goal_diff_diff"] = (
            features["home_recent_goal_diff_avg"]
            - features["away_recent_goal_diff_avg"]
        )

        feature_rows.append(features)

        home_points = _points_for_result(home_score, away_score)
        away_points = _points_for_result(away_score, home_score)

        team_histories[home_team].append(
            {
                "points": float(home_points),
                "goals_for": float(home_score),
                "goals_against": float(away_score),
                "goal_diff": float(home_score - away_score),
            }
        )

        team_histories[away_team].append(
            {
                "points": float(away_points),
                "goals_for": float(away_score),
                "goals_against": float(home_score),
                "goal_diff": float(away_score - home_score),
            }
        )

    feature_df = pd.DataFrame(feature_rows)

    return pd.concat([matches, feature_df], axis=1)

