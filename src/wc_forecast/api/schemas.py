from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str


class PredictMatchRequest(BaseModel):
    home_team: str = Field(..., examples=["Argentina"])
    away_team: str = Field(..., examples=["France"])
    neutral: bool = True


class MatchProbabilityResponse(BaseModel):
    home_team: str
    away_team: str
    neutral: bool
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    predicted_outcome: str


class MetricRow(BaseModel):
    model: str
    accuracy: float | None = None
    log_loss: float | None = None
    brier_score: float | None = None
    ece: float | None = None
    source_file: str | None = None


class ModelMetricsResponse(BaseModel):
    metrics: list[MetricRow]


class SimulateTournamentRequest(BaseModel):
    config_path: str = "configs/world_cup_2026.yaml"
    n_simulations: int | None = Field(default=None, ge=1, le=20_000)
    use_calibration: bool | None = None
    neutral: bool | None = None


class TournamentSimulationRow(BaseModel):
    team: str
    group_qualified_probability: float | None = None
    round_of_32_probability: float | None = None
    round_of_16_probability: float | None = None
    quarter_final_probability: float | None = None
    semi_final_probability: float | None = None
    final_probability: float | None = None
    tournament_win_probability: float | None = None


class TournamentSimulationResponse(BaseModel):
    config_path: str
    n_simulations: int
    results: list[TournamentSimulationRow]
