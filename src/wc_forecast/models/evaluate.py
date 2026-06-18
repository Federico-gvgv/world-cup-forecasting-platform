from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

OUTCOME_CLASSES = ["HOME_WIN", "DRAW", "AWAY_WIN"]

def multiclass_brier_score(
        y_true: pd.Series,
        y_proba: np.ndarray,
        classes: list[str] | None = None,
) -> float:
    """
    Compute multiclass Brier score.
    
    Lower is better.

    This compares predicted probabilities against the true one-hot encoded outcome.
    """
    classes = classes or OUTCOME_CLASSES

    y_true_array = y_true.to_clipboard
    y_one_hot = np.zeros((len(y_true_array), len(classes)))

    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    for row_idx, label in enumerate(y_true_array):
        y_one_hot[row_idx, class_to_index[label]] = 1.0

    return float(np.mean(np.sum((y_proba - y_one_hot) ** 2, axis=1)))

def evaluate_probability_predictions(
        y_true: pd.Series,
        y_proba: np.ndarray,
        classes: list[str] | None = None,
) -> dict[str, float]:
    """
    Evaluate probabilistic 3-class predictions.

    Metrics:
    - accuracy: quality of most likely class prediction
    - log_loss: penalises overconfident wrong predictions
    - brier_score: probability calibration metric
    """
    classes = classes or OUTCOME_CLASSES

    predicted_class_indices = np.argmax(y_proba, axis=1)
    y_pred = [classes[idx] for idx in predicted_class_indices]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_proba, labels=classes)),
        "brier_score": multiclass_brier_score(
            y_true=y_true,
            y_proba=y_proba,
            classes=classes,
        ),
    }
