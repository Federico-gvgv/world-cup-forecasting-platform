from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wc_forecast.models.evaluate import OUTCOME_CLASSES

class EmpiricalOutcomeBaseline:
    """
    Baseline that always predicts the training-set outcome distribution.

    Example:
    If the training data contains:
    - 48% home wins
    - 30% draws
    - 22% away wins

    Then every future match gets those same probabilities, regardless of the teams playing.
    """

    def __init__(self, classes: list[str] | None = None) -> None:
        self.classes = classes or OUTCOME_CLASSES
        self.class_probabilities_: np.ndarray | None = None

    def fit(self, y: pd.Series) -> "EmpiricalOutcomeBaseline":
        class_counts = y.value_counts(normalize=True)

        self.class_probabilities_ = np.array(
            [class_counts.get(cls, 0.0) for cls in self.classes]
        )
        return self
    
    def predict_proba(self, n_rows: int) -> np.ndarray:
        if self.class_probabilities_ is None:
            raise ValueError("Model must be fitted before predicting.")

        return np.tile(self.class_probabilities_, (n_rows, 1))
    

class EloLogisticRegressionBaseline:
    """
    3-class logistic regression model using only Elo-derived features.
    """

    def __init__(
            self,
            feature_columns: list[str] | None = None,
            classes: list[str] | None = None,
    ) -> None:
        self.feature_columns = feature_columns or [
            "elo_diff",
            "home_expected_score",
            "neutral",
        ]
        self.classes = classes or OUTCOME_CLASSES

        self.model = Pipeline(
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

        def fit(
                self,
                matches: pd.DataFrame,
                target_column: str = "match_outcome",
        ) -> "EloLogisticRegressionBaseline":
            self._validate_columns(matches, include_target=True, target_column=target_column)

            X = matches[self.feature_columns].copy()
            y = matches[target_column]

            self.model.fit(X, y)

            return self
        
        def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
            self._validate_columns(matches, include_target=False)

            X = matches[self.feature_columns].copy()

            raw_proba = self.model.predict_proba(X)
            model_classes = list(self.model.named_steps["logistic_regression"].classes_)

            aligned_proba = np.zeros((len(matches), len(self.classes)))

            for output_idx, class_name in enumerate(self.classes):
                if class_name in model_classes:
                    model_class_idx = model_classes.index(class_name)
                    aligned_proba[:, output_idx] = raw_proba[:, model_class_idx]

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