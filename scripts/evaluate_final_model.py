from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wc_forecast.models.calibration import (
    ProbabilityCalibrator,
    expected_calibration_error,
)
from wc_forecast.models.evaluate import evaluate_probability_predictions
from wc_forecast.models.final_evaluation import build_prediction_report
from wc_forecast.models.plots import plot_confidence_reliability_diagram
from wc_forecast.models.sklearn_models import (
    DEFAULT_FEATURE_COLUMNS,
    build_logistic_regression_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"
PREDICTIONS_DIR = REPORTS_DIR / "predictions"

TRAIN_PATH = PROCESSED_DATA_DIR / "train_features.csv"
VALIDATION_PATH = PROCESSED_DATA_DIR / "validation_features.csv"
TEST_PATH = PROCESSED_DATA_DIR / "test_features.csv"


def evaluate_with_calibration_error(
    y_true: pd.Series,
    y_proba: np.ndarray,
) -> dict[str, float]:
    metrics = evaluate_probability_predictions(
        y_true=y_true,
        y_proba=y_proba,
    )

    metrics["ece"] = expected_calibration_error(
        y_true=y_true,
        y_proba=y_proba,
        n_bins=10,
    )

    return metrics


def main() -> None:
    for path in [TRAIN_PATH, VALIDATION_PATH, TEST_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find {path}. "
                "Run the full feature pipeline first."
            )

    print(f"Loading train data from {TRAIN_PATH}")
    train = pd.read_csv(TRAIN_PATH)

    print(f"Loading validation data from {VALIDATION_PATH}")
    validation = pd.read_csv(VALIDATION_PATH)

    print(f"Loading test data from {TEST_PATH}")
    test = pd.read_csv(TEST_PATH)

    target_column = "match_outcome"

    print()
    print("Dataset sizes:")
    print(f"Train:      {len(train):,}")
    print(f"Validation: {len(validation):,}")
    print(f"Test:       {len(test):,}")

    print()
    print("Fitting selected base model on train period...")
    base_model = build_logistic_regression_model(
        feature_columns=DEFAULT_FEATURE_COLUMNS,
    )
    base_model.fit(train)

    print("Generating validation probabilities for calibration...")
    validation_proba = base_model.predict_proba(validation)

    print("Fitting calibrator on validation period...")
    calibrator = ProbabilityCalibrator()
    calibrator.fit(
        y_proba=validation_proba,
        y_true=validation[target_column],
    )

    print("Generating final test probabilities...")
    test_uncalibrated_proba = base_model.predict_proba(test)
    test_calibrated_proba = calibrator.predict_proba(test_uncalibrated_proba)

    print("Evaluating final test metrics...")
    uncalibrated_metrics = evaluate_with_calibration_error(
        y_true=test[target_column],
        y_proba=test_uncalibrated_proba,
    )

    calibrated_metrics = evaluate_with_calibration_error(
        y_true=test[target_column],
        y_proba=test_calibrated_proba,
    )

    metrics = pd.DataFrame(
        [
            {
                "model": "Logistic regression, uncalibrated",
                **uncalibrated_metrics,
            },
            {
                "model": "Logistic regression, calibrated",
                **calibrated_metrics,
            },
        ]
    )

    print()
    print("Final test metrics:")
    print(metrics.to_string(index=False))

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = METRICS_DIR / "final_test_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    print()
    print(f"Saved final test metrics to {metrics_path}")

    print("Saving final test reliability diagrams...")

    plot_confidence_reliability_diagram(
        y_true=test[target_column],
        y_proba=test_uncalibrated_proba,
        output_path=FIGURES_DIR / "final_test_reliability_uncalibrated.png",
        title="Final Test Reliability: Uncalibrated Logistic Regression",
    )

    plot_confidence_reliability_diagram(
        y_true=test[target_column],
        y_proba=test_calibrated_proba,
        output_path=FIGURES_DIR / "final_test_reliability_calibrated.png",
        title="Final Test Reliability: Calibrated Logistic Regression",
    )

    print(f"- {FIGURES_DIR / 'final_test_reliability_uncalibrated.png'}")
    print(f"- {FIGURES_DIR / 'final_test_reliability_calibrated.png'}")

    print("Saving row-level final test predictions...")

    prediction_report = build_prediction_report(
        matches=test,
        uncalibrated_proba=test_uncalibrated_proba,
        calibrated_proba=test_calibrated_proba,
    )

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = PREDICTIONS_DIR / "final_test_predictions.csv"
    prediction_report.to_csv(predictions_path, index=False)

    print(f"Saved final test predictions to {predictions_path}")


if __name__ == "__main__":
    main()
