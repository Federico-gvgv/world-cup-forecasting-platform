from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wc_forecast.models.evaluate import OUTCOME_CLASSES


def plot_confidence_reliability_diagram(
    y_true: pd.Series,
    y_proba: np.ndarray,
    output_path: Path,
    title: str,
    classes: list[str] | None = None,
    n_bins: int = 10,
) -> None:
    """
    Save a confidence-based reliability diagram.

    The x-axis is average predicted confidence.
    The y-axis is empirical accuracy.

    A perfectly calibrated model should lie close to the diagonal.
    """
    classes = classes or OUTCOME_CLASSES

    y_true_array = y_true.to_numpy()
    predicted_indices = np.argmax(y_proba, axis=1)
    predicted_labels = np.array([classes[index] for index in predicted_indices])

    confidences = np.max(y_proba, axis=1)
    correctness = predicted_labels == y_true_array

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_confidences: list[float] = []
    bin_accuracies: list[float] = []

    for bin_index in range(n_bins):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]

        if bin_index == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)

        if not np.any(in_bin):
            continue

        bin_confidences.append(float(np.mean(confidences[in_bin])))
        bin_accuracies.append(float(np.mean(correctness[in_bin])))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 7))
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", label="Perfect calibration")
    plt.plot(bin_confidences, bin_accuracies, marker="o", label="Model")
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.title(title)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()