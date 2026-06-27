from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wc_forecast.models.evaluate import OUTCOME_CLASSES


class ProbabilityCalibrator:
    """
    Multiclass probability calibrator.
    
    It learns a mapping from base model probabilities to calibrated probabilities using a held-out calibration set.
    
    This is a simple multinomial Platt-style calibration layer.
    """

    def __init__(
        self,
        classes: list[str] | None = None,
        eps: float = 1e-12,
    ) -> None:
        self.classes = classes or OUTCOME_CLASSES
        self.eps = eps

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
        y_proba: np.ndarray,
        y_true: pd.Series,
    ) -> "ProbabilityCalibrator":
        probability_features = self._probabilities_to_features(y_proba)
        self.model.fit(probability_features, y_true)
        return self

    def predict_proba(self, y_proba: np.ndarray) -> np.ndarray:
        probability_features = self._probabilities_to_features(y_proba)
        raw_proba = self.model.predict_proba(probability_features)

        model_classes = list(self.model[-1].classes_)
        aligned_proba = np.zeros((len(y_proba), len(self.classes)))

        for output_index, class_name in enumerate(self.classes):
            if class_name in model_classes:
                model_class_index = model_classes.index(class_name)
                aligned_proba[:, output_index] = raw_proba[:, model_class_index]

        return aligned_proba

    def _probabilities_to_features(self, y_proba: np.ndarray) -> np.ndarray:
        clipped_proba = np.clip(y_proba, self.eps, 1.0 - self.eps)
        return np.log(clipped_proba)

def expected_calibration_error(
        y_true: pd.Series,
        y_proba: np.ndarray,
        classes: list[str] | None = None,
        n_bins: int = 10,
) -> float:
    """
    Compute confidence-based expected calibration error.

    Lower is better.

    This checks whether predictions made with confidence around 70% are correct around 70% of the time,
    predictions made with confidence around 80% are correct around 80% of the time, etc.
    """
    classes = classes or OUTCOME_CLASSES

    y_true_array = y_true.to_numpy()
    predicted_indices = np.argmax(y_proba, axis=1)
    predicted_labels = np.array([classes[index] for index in predicted_indices])

    confidences = np.max(y_proba, axis=1)
    correctness = predicted_labels == y_true_array

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0

    for bin_index in range(n_bins):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]

        if bin_index == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)

        if not np.any(in_bin):
            continue

        bin_accuracy = float(np.mean(correctness[in_bin]))
        bin_confidence = float(np.mean(confidences[in_bin]))
        bin_weight = float(np.mean(in_bin))

        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)
