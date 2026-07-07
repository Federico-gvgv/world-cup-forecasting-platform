# Model Card: World Cup Forecasting Model

## Overview
This model predicts three-class international football match outcomes:
HOME_WIN, DRAW, and AWAY_WIN.

## Intended Use
The model is designed for probabilistic forecasting and tournament simulation, not betting or guaranteed match prediction.

## Data
The model uses historical international football results with chronological train, validation, and test splits.

## Features
- Sequential Elo rating difference
- Elo expected score
- Neutral venue indicator
- Rolling recent points
- Rolling goals scored
- Rolling goals conceded
- Rolling goal difference

## Models Compared
- Empirical outcome baseline
- Elo-only logistic regression
- Logistic regression with Elo and rolling features
- Gradient boosting with Elo and rolling features

## Evaluation
Models were evaluated using:
- Accuracy
- Log loss
- Multiclass Brier score
- Expected Calibration Error
- Reliability diagrams

## Validation Strategy
The project uses chronological splitting to avoid leakage. Features such as Elo and rolling form are computed sequentially using only matches available before each fixture.

## Calibration
A calibration layer was fitted on validation data and evaluated using proper scoring rules and reliability diagrams.

## Limitations
- The model uses match-level historical data only.
- It does not currently use player injuries, squad strength, tactical changes, or bookmaker odds.
- Group-stage tie-breaks are simplified.
- Knockout draws are resolved with a simple random shootout rule.
- The model should not be interpreted as a betting system.

## Future Work
- Add FIFA's official 2026 third-place allocation/bracket rules
- Add scoreline/Poisson simulation
- Improve tie-break rules
- Add FIFA rankings or squad-strength features
- Add bootstrapped confidence intervals
- Deploy API and dashboard to Cloud Run
