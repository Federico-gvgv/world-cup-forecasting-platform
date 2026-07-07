# Final Report: World Cup Forecasting and Simulation Platform

## 1. Summary

This project builds an end-to-end probabilistic forecasting and simulation platform for international football tournaments. The goal is not only to predict the most likely match outcome, but to produce calibrated probabilities that can be evaluated with proper scoring rules and used inside a Monte Carlo tournament simulator.

The system includes a historical data pipeline, sequential Elo ratings, rolling team-form features, model comparison, calibration analysis, final test evaluation, tournament simulation, a FastAPI service, a Streamlit dashboard, and Docker Compose support. The final model is a calibrated logistic regression using Elo and rolling-form features.

## 2. Problem Framing

International football forecasting is naturally probabilistic. Even strong teams lose, draws are common, and tournament paths depend on many sequential random outcomes. A useful model should therefore produce probabilities such as `HOME_WIN`, `DRAW`, and `AWAY_WIN`, rather than only a hard class prediction.

The project emphasizes:

- Chronological validation rather than random splits
- Leakage avoidance in feature engineering
- Proper scoring rules such as log loss and Brier score
- Calibration and reliability analysis
- Monte Carlo simulation for tournament-level probabilities

## 3. Data and Feature Pipeline

The pipeline starts from historical international match results and creates chronological train, validation, and test splits. This is important because football forecasting is a time-dependent problem: the model should be evaluated on future matches, not randomly mixed historical rows.

The main features are:

- Sequential Elo rating difference
- Elo expected score
- Neutral venue indicator
- Rolling recent points
- Rolling goals scored
- Rolling goals conceded
- Rolling goal difference

Elo and rolling-form features are computed causally. For each match, only information available before that match is used. The current match is added to team history only after its features have been generated.

## 4. Model Comparison

Several models were compared on the validation period:

| Model | Accuracy | Log loss | Brier score |
|---|---:|---:|---:|
| Empirical outcome baseline | 0.478 | 1.051 | 0.634 |
| Elo-only logistic regression | 0.594 | 0.874 | 0.515 |
| Logistic regression, Elo + rolling features | 0.599 | 0.867 | 0.510 |
| Gradient boosting, Elo + rolling features | 0.595 | 0.869 | 0.511 |

The empirical baseline is useful as a reference point, but it cannot adapt to team strength. Elo-based models improve substantially. Adding rolling-form features provides the best validation log loss and Brier score among the compared models. Gradient boosting is competitive but does not clearly outperform logistic regression, so the simpler logistic model was selected.

## 5. Calibration and Final Evaluation

The selected logistic regression model was calibrated using validation-period probabilities. Calibration was evaluated with log loss, Brier score, and expected calibration error.

Calibration-period evaluation:

| Model | Accuracy | Log loss | Brier score | ECE |
|---|---:|---:|---:|---:|
| Logistic regression, uncalibrated | 0.614 | 0.847 | 0.497 | 0.0262 |
| Logistic regression, calibrated | 0.612 | 0.844 | 0.495 | 0.0287 |

Final test evaluation:

| Model | Accuracy | Log loss | Brier score | ECE |
|---|---:|---:|---:|---:|
| Logistic regression, uncalibrated | 0.596 | 0.880 | 0.517 | 0.0208 |
| Logistic regression, calibrated | 0.603 | 0.877 | 0.515 | 0.0144 |

On the final test period, calibration improves accuracy, log loss, Brier score, and expected calibration error. The calibrated logistic regression is therefore used as the model-backed probability provider for serving and tournament simulation.

## 6. Tournament Simulation

The tournament simulator takes a YAML config defining groups, knockout pairings, simulation count, calibration settings, and neutral-site behavior. For each simulated tournament, it:

1. Builds group matchups.
2. Gets match probabilities from the model-backed provider.
3. Simulates group-stage results and standings.
4. Maps group positions into knockout fixtures.
5. Simulates knockout rounds.
6. Records whether each team reached each stage.

Repeating this process thousands of times produces estimated probabilities for group qualification, knockout stages, final appearances, and tournament wins.

One important implementation detail is class-order handling. The sklearn model outputs probabilities in `AWAY_WIN`, `DRAW`, `HOME_WIN` order, while simulation objects expose `home_win`, `draw`, and `away_win`. Tests now explicitly verify that this mapping is correct. Neutral-site predictions are also symmetrized by evaluating both team orderings and averaging them back into one perspective, reducing artificial ordering bias.

## 7. API and Dashboard

The FastAPI app exposes:

- `GET /health`
- `POST /predict_match`
- `GET /model_metrics`
- `POST /simulate_tournament`

The API service layer loads the model-backed provider, reads metrics, validates config paths, and runs simulations. Pydantic schemas define request and response contracts.

The Streamlit dashboard provides a simple interactive interface for:

- Match prediction
- Tournament simulation
- Model metrics display

Docker Compose can run the API and dashboard together, mounting the generated `data/`, `reports/`, and `configs/` directories into the containers.

## 8. Engineering Practices

The project is organized as a Python package under `src/`, with scripts for reproducible pipeline execution and tests covering the main behaviors. The test suite includes data cleaning, Elo, feature generation, model output probabilities, calibration, final evaluation reports, simulation logic, config loading, API endpoints, and model-provider probability mapping.

The current full test suite passes with 41 tests.

## 9. Limitations

The model is intentionally lightweight and uses match-level historical data only. It does not include player injuries, squad selection, tactical systems, bookmaker odds, travel burden, rest days, competition importance, or player-level quality. Group-stage tie-breaks are simplified, and knockout draws are resolved with a simple random tie-break rather than scoreline or penalty-specific modeling.

The project now includes a 48-team, 12-group `world_cup_2026` config. It should still be interpreted as a World Cup-style simulator rather than an exact reproduction of FIFA's official 2026 bracket, because the Round of 32 assignment uses a simplified deterministic seeding rule instead of the official third-place allocation table. The smaller example config remains useful for quick tests and demonstrations.

## 10. Future Work

Future improvements could include FIFA's official 2026 third-place allocation table, scoreline simulation, Poisson-style goal modeling, more realistic group tie-breaks, squad-strength features, confidence intervals around tournament probabilities, and cloud deployment for the API and dashboard.

Overall, the project demonstrates an end-to-end probabilistic forecasting workflow: careful data preparation, leakage-aware features, model comparison, probability calibration, simulation, testing, and interactive serving.
