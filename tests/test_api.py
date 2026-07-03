from __future__ import annotations

from fastapi.testclient import TestClient

from wc_forecast.api.main import app


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "ok"
    assert payload["service"] == "world-cup-forecasting-api"


def test_predict_match_rejects_same_team() -> None:
    with TestClient(app) as client:
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
    with TestClient(app) as client:
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
    with TestClient(app) as client:
        response = client.get("/model_metrics")

    assert response.status_code == 200

    payload = response.json()

    assert "metrics" in payload
    assert isinstance(payload["metrics"], list)


def test_simulate_tournament_endpoint() -> None:
    with TestClient(app) as client:
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