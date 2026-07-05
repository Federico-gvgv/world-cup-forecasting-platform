from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from wc_forecast.data.features import RollingFeatureConfig
from wc_forecast.models.calibration import ProbabilityCalibrator
from wc_forecast.models.sklearn_models import (
    DEFAULT_FEATURE_COLUMNS,
    SklearnProbabilityModel,
    build_logistic_regression_model,
)
from wc_forecast.ratings.elo import EloConfig, expected_score
from wc_forecast.simulation.match import MatchProbabilities


@dataclass
class TeamState:
    elo_rating: float
    recent_points: deque[float] = field(default_factory=deque)
    recent_goals_for: deque[float] = field(default_factory=deque)
    recent_goals_against: deque[float] = field(default_factory=deque)
    recent_goal_diff: deque[float] = field(default_factory=deque)


def _points_for_result(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3

    if goals_for == goals_against:
        return 1

    return 0


def _average_or_default(values: deque[float], default: float) -> float:
    if not values:
        return default

    return float(sum(values) / len(values))


class TeamFeatureStore:
    """
    Stores latest team state needed for future match prediction.

    It is built from historical matches in chronological order.
    For a future match, it returns the features known before that match:
    - latest Elo rating
    - recent points
    - recent goals for/conceded
    - recent goal difference
    """

    def __init__(
        self,
        states: dict[str, TeamState],
        elo_config: EloConfig | None = None,
        rolling_config: RollingFeatureConfig | None = None,
    ) -> None:
        self.states = states
        self.elo_config = elo_config or EloConfig()
        self.rolling_config = rolling_config or RollingFeatureConfig()

    @classmethod
    def from_matches(
        cls,
        matches: pd.DataFrame,
        elo_config: EloConfig | None = None,
        rolling_config: RollingFeatureConfig | None = None,
    ) -> "TeamFeatureStore":
        elo_config = elo_config or EloConfig()
        rolling_config = rolling_config or RollingFeatureConfig()

        required_columns = {
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "home_elo_post",
            "away_elo_post",
        }

        missing_columns = required_columns - set(matches.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        matches = matches.copy()
        matches["date"] = pd.to_datetime(matches["date"])
        matches = matches.sort_values("date").reset_index(drop=True)

        states: dict[str, TeamState] = defaultdict(
            lambda: TeamState(
                elo_rating=elo_config.initial_rating,
                recent_points=deque(maxlen=rolling_config.window_size),
                recent_goals_for=deque(maxlen=rolling_config.window_size),
                recent_goals_against=deque(maxlen=rolling_config.window_size),
                recent_goal_diff=deque(maxlen=rolling_config.window_size),
            )
        )

        for row in matches.itertuples(index=False):
            home_team = str(row.home_team)
            away_team = str(row.away_team)

            home_score = int(row.home_score)
            away_score = int(row.away_score)

            home_state = states[home_team]
            away_state = states[away_team]

            home_state.elo_rating = float(row.home_elo_post)
            away_state.elo_rating = float(row.away_elo_post)

            home_points = _points_for_result(home_score, away_score)
            away_points = _points_for_result(away_score, home_score)

            home_state.recent_points.append(float(home_points))
            home_state.recent_goals_for.append(float(home_score))
            home_state.recent_goals_against.append(float(away_score))
            home_state.recent_goal_diff.append(float(home_score - away_score))

            away_state.recent_points.append(float(away_points))
            away_state.recent_goals_for.append(float(away_score))
            away_state.recent_goals_against.append(float(home_score))
            away_state.recent_goal_diff.append(float(away_score - home_score))

        return cls(
            states=dict(states),
            elo_config=elo_config,
            rolling_config=rolling_config,
        )

    def get_state(self, team: str) -> TeamState:
        if team in self.states:
            return self.states[team]

        return TeamState(
            elo_rating=self.elo_config.initial_rating,
            recent_points=deque(maxlen=self.rolling_config.window_size),
            recent_goals_for=deque(maxlen=self.rolling_config.window_size),
            recent_goals_against=deque(maxlen=self.rolling_config.window_size),
            recent_goal_diff=deque(maxlen=self.rolling_config.window_size),
        )

    def make_match_features(
        self,
        team_a: str,
        team_b: str,
        neutral: bool = True,
    ) -> pd.DataFrame:
        """
        Build one feature row for a future match.

        team_a is treated as the home/team A side in the probability output.
        For World Cup simulations, neutral=True should usually be used.
        """
        team_a_state = self.get_state(team_a)
        team_b_state = self.get_state(team_b)

        team_a_rating_for_prediction = team_a_state.elo_rating

        if not neutral:
            team_a_rating_for_prediction += self.elo_config.home_advantage

        home_expected = expected_score(
            team_rating=team_a_rating_for_prediction,
            opponent_rating=team_b_state.elo_rating,
            scale=self.elo_config.scale,
        )

        home_recent_points = _average_or_default(
            team_a_state.recent_points,
            self.rolling_config.default_points,
        )
        away_recent_points = _average_or_default(
            team_b_state.recent_points,
            self.rolling_config.default_points,
        )

        home_recent_goals_for = _average_or_default(
            team_a_state.recent_goals_for,
            self.rolling_config.default_goals_for,
        )
        away_recent_goals_for = _average_or_default(
            team_b_state.recent_goals_for,
            self.rolling_config.default_goals_for,
        )

        home_recent_goals_against = _average_or_default(
            team_a_state.recent_goals_against,
            self.rolling_config.default_goals_against,
        )
        away_recent_goals_against = _average_or_default(
            team_b_state.recent_goals_against,
            self.rolling_config.default_goals_against,
        )

        home_recent_goal_diff = _average_or_default(
            team_a_state.recent_goal_diff,
            self.rolling_config.default_goal_diff,
        )
        away_recent_goal_diff = _average_or_default(
            team_b_state.recent_goal_diff,
            self.rolling_config.default_goal_diff,
        )

        features = {
            "elo_diff": team_a_state.elo_rating - team_b_state.elo_rating,
            "home_expected_score": home_expected,
            "neutral": neutral,
            "recent_points_diff": home_recent_points - away_recent_points,
            "recent_goals_for_diff": (
                home_recent_goals_for - away_recent_goals_for
            ),
            "recent_goals_against_diff": (
                home_recent_goals_against - away_recent_goals_against
            ),
            "recent_goal_diff_diff": (
                home_recent_goal_diff - away_recent_goal_diff
            ),
        }

        return pd.DataFrame([features], columns=DEFAULT_FEATURE_COLUMNS)


class ModelBackedProbabilityProvider:
    """
    Probability provider for tournament simulation.

    Given two teams, it:
    - builds current match features
    - calls the trained model
    - optionally applies calibration
    - returns MatchProbabilities

    For neutral matches, it symmetrises predictions by evaluating both
    team orderings and averaging them. This avoids artefacts where the
    arbitrary Team A / Team B ordering changes neutral-site probabilities.
    """

    def __init__(
        self,
        model: SklearnProbabilityModel,
        feature_store: TeamFeatureStore,
        calibrator: ProbabilityCalibrator | None = None,
        neutral: bool = True,
    ) -> None:
        self.model = model
        self.feature_store = feature_store
        self.calibrator = calibrator
        self.neutral = neutral

    def __call__(
        self,
        team_a: str,
        team_b: str,
    ) -> MatchProbabilities:
        if self.neutral:
            probabilities = self._predict_neutral_symmetric(
                team_a=team_a,
                team_b=team_b,
            )
        else:
            probabilities = self._predict_one_direction(
                team_a=team_a,
                team_b=team_b,
                neutral=False,
            )

        probabilities = np.clip(probabilities, 0.0, 1.0)
        probabilities = probabilities / probabilities.sum()

        return MatchProbabilities(
            home_win=float(probabilities[2]),
            draw=float(probabilities[1]),
            away_win=float(probabilities[0]),
        )

    def _predict_one_direction(
        self,
        team_a: str,
        team_b: str,
        neutral: bool,
    ) -> np.ndarray:
        features = self.feature_store.make_match_features(
            team_a=team_a,
            team_b=team_b,
            neutral=neutral,
        )

        probabilities = self.model.predict_proba(features)

        if self.calibrator is not None:
            probabilities = self.calibrator.predict_proba(probabilities)

        return probabilities[0]

    def _predict_neutral_symmetric(
        self,
        team_a: str,
        team_b: str,
    ) -> np.ndarray:
        """
        Predict a neutral-site match in an order-invariant way.

        Model probabilities are ordered as [AWAY_WIN, DRAW, HOME_WIN].

        Forward prediction:
            team_a vs team_b
            [P(team_b win), P(draw), P(team_a win)]

        Reverse prediction:
            team_b vs team_a
            [P(team_a win), P(draw), P(team_b win)]

        Convert reverse prediction back into team_a/team_b perspective,
        then average.
        """
        forward = self._predict_one_direction(
            team_a=team_a,
            team_b=team_b,
            neutral=True,
        )

        reverse = self._predict_one_direction(
            team_a=team_b,
            team_b=team_a,
            neutral=True,
        )

        team_b_win_probability = 0.5 * (
            forward[0] + reverse[2]
        )
        draw_probability = 0.5 * (
            forward[1] + reverse[1]
        )
        team_a_win_probability = 0.5 * (
            forward[2] + reverse[0]
        )

        return np.array(
            [
                team_b_win_probability,
                draw_probability,
                team_a_win_probability,
            ]
        )


def build_model_backed_probability_provider(
    train_path: Path,
    validation_path: Path,
    matches_with_features_path: Path,
    use_calibration: bool = True,
    neutral: bool = True,
) -> ModelBackedProbabilityProvider:
    """
    Build a trained model-backed probability provider.

    Training logic:
    - fit base model on train period
    - optionally fit calibrator on validation period
    - build current team states from all available processed matches
    """
    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    matches = pd.read_csv(matches_with_features_path)

    target_column = "match_outcome"

    model = build_logistic_regression_model(
        feature_columns=DEFAULT_FEATURE_COLUMNS,
    )
    model.fit(train)

    calibrator = None

    if use_calibration:
        validation_proba = model.predict_proba(validation)

        calibrator = ProbabilityCalibrator()
        calibrator.fit(
            y_proba=validation_proba,
            y_true=validation[target_column],
        )

    feature_store = TeamFeatureStore.from_matches(matches)

    return ModelBackedProbabilityProvider(
        model=model,
        feature_store=feature_store,
        calibrator=calibrator,
        neutral=neutral,
    )
