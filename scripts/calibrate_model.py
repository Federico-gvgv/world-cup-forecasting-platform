from __future__ import annotations

from pathlib import Path

import pandas as pd

from wc_forecast.models.calibration import (
    ProbabilityCalibrator,
    expected_calibration_error,
)
from wc_forecast.models.evaluate import evaluate_probability_predictions
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

TRAIN_PATH = PROCESSED_DATA_DIR / "train_features.csv"
VALIDATION_PATH = PROCESSED_DATA_DIR / "validation_features.csv"


def split_validation_for_calibration(
        validation: pd.DataFrame,
        calibration_end_date: str = "2020-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the validation period into:
    - calibration set
    - calibration evaluation set
    
    This avoids fitting and evaluating calibration on the same rows.
    """
    validation = validation.copy()
    validation["date"] = pd.to_datetime(validation["date"])

    calibration_end = pd.Timestamp(calibration_end_date)

    calibration = validation[validation["date"] < calibration_end].copy()
    calibration_eval = validation[validation["date"] >= calibration_end].copy()

    if calibration.empty:
        raise ValueError("Calibration split is empty.")

    if calibration_eval.empty:
        raise ValueError("Calibration evaluation split is empty.")

    return calibration, calibration_eval


def evaluate_with_calibration_error(
        y_true: pd.Series,
        y_proba,
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
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {TRAIN_PATH}. "
            "Run `python scripts/build_features.py` first."
        )

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {VALIDATION_PATH}. "
            "Run `python scripts/build_features.py` first."
        )

    print(f"Loading train data from {TRAIN_PATH}")
    train = pd.read_csv(TRAIN_PATH)

    print(f"Loading validation data from {VALIDATION_PATH}")
    validation = pd.read_csv(VALIDATION_PATH)

    calibration, calibration_eval = split_validation_for_calibration(validation)

    print()
    print("Split sizes:")
    print(f"Train:                  {len(train):,}")
    print(f"Calibration:            {len(calibration):,}")
    print(f"Calibration evaluation: {len(calibration_eval):,}")

    target_column = "match_outcome"

    print()
    print("Fitting base logistic regression model on train period...")
    base_model = build_logistic_regression_model(
        feature_columns=DEFAULT_FEATURE_COLUMNS,
    )
    base_model.fit(train)

    print("Generating base probabilities...")
    calibration_proba = base_model.predict_proba(calibration)
    calibration_eval_uncalibrated_proba = base_model.predict_proba(calibration_eval)

    print("Evaluating uncalibrated model...")
    uncalibrated_metrics = evaluate_with_calibration_error(
        y_true=calibration_eval[target_column],
        y_proba=calibration_eval_uncalibrated_proba,
    )

    print("Fitting probability calibrator on calibration period...")
    calibrator = ProbabilityCalibrator()
    calibrator.fit(
        y_proba=calibration_proba,
        y_true=calibration[target_column],
    )

    print("Evaluating calibrated model...")
    calibration_eval_calibrated_proba = calibrator.predict_proba(
        calibration_eval_uncalibrated_proba
    )

    calibrated_metrics = evaluate_with_calibration_error(
        y_true=calibration_eval[target_column],
        y_proba=calibration_eval_calibrated_proba,
    )

    results = pd.DataFrame(
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
    print("Calibration evaluation metrics:")
    print(results.to_string(index=False))

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = METRICS_DIR / "calibration_validation_metrics.csv"
    results.to_csv(metrics_path, index=False)

    print()
    print(f"Saved calibration metrics to {metrics_path}")

    print("Saving reliability diagrams...")

    plot_confidence_reliability_diagram(
        y_true=calibration_eval[target_column],
        y_proba=calibration_eval_uncalibrated_proba,
        output_path=FIGURES_DIR / "reliability_uncalibrated.png",
        title="Reliability Diagram: Uncalibrated Logistic Regression",
    )

    plot_confidence_reliability_diagram(
        y_true=calibration_eval[target_column],
        y_proba=calibration_eval_calibrated_proba,
        output_path=FIGURES_DIR / "reliability_calibrated.png",
        title="Reliability Diagram: Calibrated Logistic Regression",
    )

    print(f"- {FIGURES_DIR / 'reliability_uncalibrated.png'}")
    print(f"- {FIGURES_DIR / 'reliability_calibrated.png'}")


if __name__ == "__main__":
    main()