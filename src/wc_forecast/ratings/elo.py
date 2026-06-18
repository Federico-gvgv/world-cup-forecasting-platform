from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

@dataclass(frozen=True)
class EloConfig:
    initial_rating: float = 1500.0
    k_factor: float = 20.0
    scale: float = 400.0
    home_advantage: float = 75.0

def expected_score(
        team_rating: float,
        opponent_rating: float,
        scale: float = 400.0,
) -> float:
    """
    Calculate the expected score for one team against another.
    
    Returns a value between 0 and 1:
    - close to 1 means team is expected to win
    - close to 0.5 means teams are evenly matched
    - close to 0 means team is expected to lose
    """
    return 1.0 / (1.0 + 10 ** ((opponent_rating - team_rating) / scale))

def actual_home_score(home_score: int, away_score: int) -> float:
    """
    Convert match result into Elo score from the home team's perspective.
    
    Home win -> 1.0
    Draw -> 0.5
    Away win -> 0.0
    """
    if home_score > away_score:
        return 1.0
    elif home_score == away_score:
        return 0.5
    else:
        return 0.0
    
@dataclass
class EloRatingSystem:
    config: EloConfig = field(default_factory=EloConfig)
    ratings: dict[str, float] = field(default_factory=dict)

    def get_rating(self, team: str) -> float:
        """
        Get the current Elo rating for a team.
        If the team has no rating yet, return the initial rating.
        """
        return self.ratings.get(team, self.config.initial_rating)
    
    def update_match(
            self,
            home_team: str,
            away_team: str,
            home_score: int,
            away_score: int,
            neutral: bool = False,
    ) -> dict[str, float]:
        """
        Record pre-match ratings, update ratings after match, and return useful Elo features.

        This is causal:
        - pre-match ratings are read before the update
        - post-match ratings are written after the result
        """
        home_elo_pre = self.get_rating(home_team)
        away_elo_pre = self.get_rating(away_team)

        home_rating_for_prediction = home_elo_pre
        if not neutral:
            home_rating_for_prediction += self.config.home_advantage
        
        away_rating_for_prediction = away_elo_pre

        home_expected = expected_score(
            home_rating_for_prediction,
            away_rating_for_prediction,
            scale=self.config.scale,
        )
        away_expected = 1.0 - home_expected

        home_actual = actual_home_score(home_score, away_score)
        away_actual = 1.0 - home_actual

        home_elo_post = home_elo_pre + self.config.k_factor * (home_actual - home_expected)
        away_elo_post = away_elo_pre + self.config.k_factor * (away_actual - away_expected)

        self.ratings[home_team] = home_elo_post
        self.ratings[away_team] = away_elo_post

        return {
            "home_elo_pre": home_elo_pre,
            "away_elo_pre": away_elo_pre,
            "elo_diff": home_elo_pre - away_elo_pre,
            "home_expected_score": home_expected,
            "away_expected_score": away_expected,
            "home_elo_post": home_elo_post,
            "away_elo_post": away_elo_post,
        }
    
def add_elo_features(
        matches: pd.DataFrame,
        config: EloConfig | None = None,
) -> pd.DataFrame:
    """
    Add sequential Elo features to a match dataframe.
    
    The dataframe must contain:
    - date
    - home_team
    - away_team
    - home_score
    - away_score

    If a neutral column exists, it is used. Otherwise matches are trated as non_neutral.
    """
    required_columns = {"date", "home_team", "away_team", "home_score", "away_score"}

    missing_columns = required_columns - set(matches.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values("date").reset_index(drop=True) # Sort matches chronologically

    rating_system = EloRatingSystem(config=config or EloConfig())

    elo_rows: list[dict[str, float]] = []

    for row in matches.itertuples(index=False):
        neutral = bool(getattr(row, "neutral", False))
        elo_features = rating_system.update_match(
            home_team=row.home_team,
            away_team=row.away_team,
            home_score=int(row.home_score),
            away_score=int(row.away_score),
            neutral=neutral,
        )
        elo_rows.append(elo_features)

    elo_df = pd.DataFrame(elo_rows)

    return pd.concat([matches, elo_df], axis=1)