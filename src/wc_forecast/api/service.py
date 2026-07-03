from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from wc_forecast.simulation.config import (
    load_simulation_options,
    load_tournament_config,
)
from wc_forecast.simulation.match import MatchProbabilities
from wc_forecast.simulation.model_provider import (
    ModelBackedProbabilityProvider,
    build_model_backed_probability_provider,
)
from wc_forecast.simulation.tournament import simulate_tournament


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"

TRAIN_PATH = PROCESSED_DATA_DIR / "train_features.csv"
VALIDATION_PATH = PROCESSED_DATA_DIR / "validation_features.csv"
MATCHES_FEATURES_PATH = PROCESSED_DATA_DIR / "matches_with_features.csv"


class ForecastService:
    """
    Service layer for API endpoints.

    Keeps FastAPI route handlers thin.
    """

    def __init__(self) -> None:
        self._base_provider = build_model_backed_probability_provider(
            train_path=TRAIN_PATH,
            validation_path=VALIDATION_PATH,
            matches_with_features_path=MATCHES_FEATURES_PATH,
            use_calibration=True,
            neutral=True,
        )

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
    ) -> MatchProbabilities:
        provider = ModelBackedProbabilityProvider(
            model=self._base_provider.model,
            feature_store=self._base_provider.feature_store,
            calibrator=self._base_provider.calibrator,
            neutral=neutral,
        )

        return provider(
            team_a=home_team,
            team_b=away_team,
        )

    def load_model_metrics(self) -> list[dict[str, object]]:
        metric_files = [
            METRICS_DIR / "baseline_validation_metrics.csv",
            METRICS_DIR / "calibration_validation_metrics.csv",
            METRICS_DIR / "final_test_metrics.csv",
        ]

        rows: list[dict[str, object]] = []

        for metric_file in metric_files:
            if not metric_file.exists():
                continue

            metrics = pd.read_csv(metric_file)

            for row in metrics.to_dict(orient="records"):
                row["source_file"] = metric_file.name
                rows.append(row)

        return rows

    def simulate_tournament_from_config(
        self,
        config_path: str,
        n_simulations: int | None = None,
        use_calibration: bool | None = None,
        neutral: bool | None = None,
    ) -> tuple[int, list[dict[str, object]]]:
        config_file = self._resolve_config_path(config_path)

        tournament_config = load_tournament_config(config_file)
        simulation_options = load_simulation_options(config_file)

        if n_simulations is not None:
            tournament_config = replace(
                tournament_config,
                n_simulations=n_simulations,
            )

        effective_use_calibration = (
            simulation_options["use_calibration"]
            if use_calibration is None
            else use_calibration
        )
        effective_neutral = (
            simulation_options["neutral"]
            if neutral is None
            else neutral
        )

        provider = ModelBackedProbabilityProvider(
            model=self._base_provider.model,
            feature_store=self._base_provider.feature_store,
            calibrator=(
                self._base_provider.calibrator
                if effective_use_calibration
                else None
            ),
            neutral=effective_neutral,
        )

        results = simulate_tournament(
            config=tournament_config,
            probability_provider=provider,
        )

        return tournament_config.n_simulations, results.to_dict(orient="records")

    def _resolve_config_path(self, config_path: str) -> Path:
        path = Path(config_path)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        path = path.resolve()

        if not path.is_relative_to(PROJECT_ROOT):
            raise ValueError("Config path must be inside the project directory.")

        if not path.exists():
            raise FileNotFoundError(f"Could not find config file: {path}")

        return path
