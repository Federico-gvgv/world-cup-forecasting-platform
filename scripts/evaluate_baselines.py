from __future__ import annotations

from pathlib import Path

import pandas as pd

from wc_forecast.models.baselines import (
    EloLogisticRegressionBaseline,
    EmpiricalOutcomeBaseline,
)
from wc_forecast.models.evaluate import evaluate_probability_predictions

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"

TRAIN_PATH = PROCESSED_DATA_DIR / "train_with_elo.csv"
VALIDATION_PATH = PROCESSED_DATA_DIR / "validation_with_elo.csv"

def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {TRAIN_PATH}. "
            "Run build_elo_features.py first."
        )
    
    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {VALIDATION_PATH}. "
            "Run build_elo_features.py first."
        )
    
    print(f"Loading train data from {TRAIN_PATH}")
    train = pd.read_csv(TRAIN_PATH)

    print(f"Loading validation data from {VALIDATION_PATH}")
    validation = pd.read_csv(VALIDATION_PATH)

    target_column = "match_outcome"

    results: list[dict[str, float | str]] = []

    print("Evaluating empirical outcome baseline...")
    empirical_baseline = EmpiricalOutcomeBaseline()
    empirical_baseline.fit(train[target_column])

    empirical_proba = empirical_baseline.predict_proba(n_rows=len(validation))

    empirical_metrics = evaluate_probability_predictions(
        y_true=validation[target_column],
        y_proba=empirical_proba,
    )

    results.append(
        {
            "model": "Empirical outcome baseline",
            **empirical_metrics,
        }
    )

    print("Evaluating Elo-only logistic regression baseline...")
    elo_baseline = EloLogisticRegressionBaseline()
    elo_baseline.fit(train)

    elo_proba = elo_baseline.predict_proba(validation)

    elo_metrics = evaluate_probability_predictions(
        y_true=validation[target_column],
        y_proba=elo_proba,
    )

    results.append(
        {
            "model": "Elo-only logistic regression",
            **elo_metrics,
        }
    )

    metrics = pd.DataFrame(results)

    print()
    print("Validation metrics:")
    print(metrics.to_string(index=False))

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = METRICS_DIR / "baseline_validation_metrics.csv"
    metrics.to_csv(output_path, index=False)

    print()
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()