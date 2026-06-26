from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wc_forecast.models.evaluate import OUTCOME_CLASSES

DEFAULT_FEATURE_COLUMNS = [
    "elo_diff",
    "home_expected_score",
    "neutral",
    "recent_points_diff",
    "recent_goals_for_diff",
    "recent_goals_against_diff",
    "recent_goal_diff_diff",
]


class SklearnProbabilityModel:
    """
    Thin wrapper around sklearn classifiers.

    It ensures that predict_proba always returns probabilities in this order:

    AWAY_WIN, DRAW, HOME_WIN
    """

    def __init__(
        self,
        model: ClassifierMixin,
        feature_columns: list[str] | None = None,
        classes: list[str] | None = None,
    ) -> None:
        self.model = model
        self.feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
        self.classes = classes or OUTCOME_CLASSES

    def fit(
        self,
        matches: pd.DataFrame,
        target_column: str = "match_outcome",
    ) -> "SklearnProbabilityModel":
        self._validate_columns(matches, include_target=True, target_column=target_column)

        X = matches[self.feature_columns].copy()
        y = matches[target_column]

        self.model.fit(X, y)

        return self
    
    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        self._validate_columns(matches, include_target=False)

        X = matches[self.feature_columns].copy()
        raw_proba = self.model.predict_proba(X)

        if hasattr(self.model, "classes_"):
            model_classes = list(self.model.classes_)
        else:
            model_classes = list(self.model[-1].classes_)
            
        aligned_proba = np.zeros((len(matches), len(self.classes)))

        for output_index, class_name in enumerate(self.classes):
            if class_name in model_classes:
                model_class_index = model_classes.index(class_name)
                aligned_proba[:, output_index] = raw_proba[:, model_class_index]

        return aligned_proba
    
    def _validate_columns(
        self,
        matches: pd.DataFrame,
        include_target: bool,
        target_column: str = "match_outcome",
    ) -> None:
        required_columns = set(self.feature_columns)
        
        if include_target:
            required_columns.add(target_column)

        missing_columns = required_columns - set(matches.columns)

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")


def build_logistic_regression_model(
    feature_columns: list[str] | None = None,
) -> SklearnProbabilityModel:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    # Pipeline exposes classes_ after fit, but type checkers do not know that.
    return SklearnProbabilityModel(
        model=model,
        feature_columns=feature_columns,
    )


def build_gradient_boosting_model(
    feature_columns: list[str] | None = None,
) -> SklearnProbabilityModel:
    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        random_state=42,
    )

    return SklearnProbabilityModel(
        model=model,
        feature_columns=feature_columns,
    )
