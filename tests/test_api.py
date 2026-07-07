from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from wc_forecast.api import main as api_main
from wc_forecast.simulation.match import MatchProbabilities


class FakeForecastService:
    def predict_match(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
    ) -> MatchProbabilities:
        return MatchProbabilities(
            home_win=0.5,
            draw=0.25,
            away_win=0.25,
        )

    def load_model_metrics(self) -> list[dict[str, object]]:
        return [
            {
                "model": "fake-model",
                "accuracy": 0.5,
                "log_loss": 1.0,
                "brier_score": 0.6,
                "ece": 0.1,
                "source_file": "fake_metrics.csv",
            }
        ]

    def simulate_tournament_from_config(
        self,
        config_path: str,
        n_simulations: int | None = None,
        use_calibration: bool | None = None,
        neutral: bool | None = None,
    ) -> tuple[int, list[dict[str, object]]]:
        return (
            n_simulations or 1000,
            [
                {
                    "team": "Team A",
                    "group_qualified_probability": 1.0,
                    "round_of_32_probability": 1.0,
                    "round_of_16_probability": 1.0,
                    "quarter_final_probability": 1.0,
                    "semi_final_probability": 1.0,
                    "final_probability": 1.0,
                    "tournament_win_probability": 1.0,
                }
            ],
        )


@pytest.fixture(autouse=True)
def use_fake_forecast_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_main, "ForecastService", FakeForecastService)


def test_health_endpoint() -> None:
    with TestClient(api_main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "ok"
    assert payload["service"] == "world-cup-forecasting-api"


def test_predict_match_rejects_same_team() -> None:
    with TestClient(api_main.app) as client:
        response = client.post(
            "/predict_match",
            json={
                "home_team": "Argentina",
                "away_team": "Argentina",
                "neutral": True,
            },
        )

    assert response.status_code == 400


def test_predict_match_returns_probabilities() -> None:
    with TestClient(api_main.app) as client:
        response = client.post(
            "/predict_match",
            json={
                "home_team": "Argentina",
                "away_team": "France",
                "neutral": True,
            },
        )

    assert response.status_code == 200

    payload = response.json()

    probability_sum = (
        payload["home_win_probability"]
        + payload["draw_probability"]
        + payload["away_win_probability"]
    )

    assert abs(probability_sum - 1.0) < 1e-6
    assert payload["predicted_outcome"] in {
        "HOME_WIN",
        "DRAW",
        "AWAY_WIN",
    }


def test_model_metrics_endpoint() -> None:
    with TestClient(api_main.app) as client:
        response = client.get("/model_metrics")

    assert response.status_code == 200

    payload = response.json()

    assert "metrics" in payload
    assert isinstance(payload["metrics"], list)
    assert payload["metrics"][0]["model"] == "fake-model"


def test_simulate_tournament_endpoint() -> None:
    with TestClient(api_main.app) as client:
        response = client.post(
            "/simulate_tournament",
            json={
                "config_path": "configs/example_tournament.yaml",
                "n_simulations": 25,
                "use_calibration": True,
                "neutral": True,
            },
        )

    assert response.status_code == 200

    payload = response.json()

    assert payload["n_simulations"] == 25
    assert "results" in payload
    assert len(payload["results"]) > 0

    first_row = payload["results"][0]

    assert "team" in first_row
    assert "tournament_win_probability" in first_row
