from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from wc_forecast.api.schemas import (
    HealthResponse,
    MatchProbabilityResponse,
    ModelMetricsResponse,
    PredictMatchRequest,
    SimulateTournamentRequest,
    TournamentSimulationResponse,
)
from wc_forecast.api.service import ForecastService


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.forecast_service = ForecastService()
    yield


app = FastAPI(
    title="World Cup Forecasting API",
    description=(
        "Probabilistic international football forecasting API with "
        "Elo, rolling features, calibration, and Monte Carlo simulation."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="world-cup-forecasting-api",
    )


@app.post("/predict_match", response_model=MatchProbabilityResponse)
def predict_match(request: PredictMatchRequest) -> MatchProbabilityResponse:
    if request.home_team == request.away_team:
        raise HTTPException(
            status_code=400,
            detail="home_team and away_team must be different.",
        )

    service: ForecastService = app.state.forecast_service

    probabilities = service.predict_match(
        home_team=request.home_team,
        away_team=request.away_team,
        neutral=request.neutral,
    )

    probability_map = {
        "HOME_WIN": probabilities.home_win,
        "DRAW": probabilities.draw,
        "AWAY_WIN": probabilities.away_win,
    }

    predicted_outcome = max(
        probability_map,
        key=probability_map.get,
    )

    return MatchProbabilityResponse(
        home_team=request.home_team,
        away_team=request.away_team,
        neutral=request.neutral,
        home_win_probability=probabilities.home_win,
        draw_probability=probabilities.draw,
        away_win_probability=probabilities.away_win,
        predicted_outcome=predicted_outcome,
    )


@app.get("/model_metrics", response_model=ModelMetricsResponse)
def model_metrics() -> ModelMetricsResponse:
    service: ForecastService = app.state.forecast_service

    return ModelMetricsResponse(
        metrics=service.load_model_metrics(),
    )


@app.post("/simulate_tournament", response_model=TournamentSimulationResponse)
def simulate_tournament_endpoint(
    request: SimulateTournamentRequest,
) -> TournamentSimulationResponse:
    service: ForecastService = app.state.forecast_service

    try:
        n_simulations, results = service.simulate_tournament_from_config(
            config_path=request.config_path,
            n_simulations=request.n_simulations,
            use_calibration=request.use_calibration,
            neutral=request.neutral,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return TournamentSimulationResponse(
        config_path=request.config_path,
        n_simulations=n_simulations,
        results=results,
    )