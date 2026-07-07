from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


DEFAULT_API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8000",
)


st.set_page_config(
    page_title="World Cup Forecasting Platform",
    page_icon="⚽",
    layout="wide",
)


def api_get(endpoint: str, api_base_url: str) -> dict[str, Any]:
    response = requests.get(
        f"{api_base_url}{endpoint}",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def api_post(
    endpoint: str,
    payload: dict[str, Any],
    api_base_url: str,
    timeout: int = 60,
) -> dict[str, Any]:
    response = requests.post(
        f"{api_base_url}{endpoint}",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def render_probability_cards(prediction: dict[str, Any]) -> None:
    col1, col2, col3 = st.columns(3)

    col1.metric(
        label=f"{prediction['home_team']} win",
        value=f"{prediction['home_win_probability']:.1%}",
    )
    col2.metric(
        label="Draw",
        value=f"{prediction['draw_probability']:.1%}",
    )
    col3.metric(
        label=f"{prediction['away_team']} win",
        value=f"{prediction['away_win_probability']:.1%}",
    )

    st.info(f"Predicted outcome: **{prediction['predicted_outcome']}**")


def render_match_prediction(api_base_url: str) -> None:
    st.header("Match prediction")

    with st.form("match_prediction_form"):
        col1, col2, col3 = st.columns([2, 2, 1])

        home_team = col1.text_input(
            "Team A",
            value="Argentina",
        )
        away_team = col2.text_input(
            "Team B",
            value="France",
        )
        neutral = col3.checkbox(
            "Neutral venue",
            value=True,
        )

        submitted = st.form_submit_button("Predict match")

    if not submitted:
        return

    if home_team.strip() == away_team.strip():
        st.error("Team A and Team B must be different.")
        return

    try:
        prediction = api_post(
            endpoint="/predict_match",
            payload={
                "home_team": home_team.strip(),
                "away_team": away_team.strip(),
                "neutral": neutral,
            },
            api_base_url=api_base_url,
        )
    except requests.RequestException as error:
        st.error(f"Prediction request failed: {error}")
        return

    render_probability_cards(prediction)

    probability_df = pd.DataFrame(
        {
            "Outcome": [
                f"{prediction['home_team']} win",
                "Draw",
                f"{prediction['away_team']} win",
            ],
            "Probability": [
                prediction["home_win_probability"],
                prediction["draw_probability"],
                prediction["away_win_probability"],
            ],
        }
    )

    st.bar_chart(
        probability_df.set_index("Outcome"),
        y="Probability",
    )


def render_model_metrics(api_base_url: str) -> None:
    st.header("Model metrics")

    try:
        payload = api_get(
            endpoint="/model_metrics",
            api_base_url=api_base_url,
        )
    except requests.RequestException as error:
        st.error(f"Could not load model metrics: {error}")
        return

    metrics = payload.get("metrics", [])

    if not metrics:
        st.warning("No metrics found. Run the evaluation scripts first.")
        return

    metrics_df = pd.DataFrame(metrics)

    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True,
    )


def render_tournament_simulation(api_base_url: str) -> None:
    st.header("Tournament simulation")

    with st.form("tournament_simulation_form"):
        config_path = st.text_input(
            "Tournament config path",
            value="configs/world_cup_2026.yaml",
        )

        col1, col2, col3 = st.columns(3)

        n_simulations = col1.number_input(
            "Number of simulations",
            min_value=25,
            max_value=20_000,
            value=1_000,
            step=25,
        )

        use_calibration = col2.checkbox(
            "Use calibration",
            value=True,
        )

        neutral = col3.checkbox(
            "Neutral venues",
            value=True,
        )

        submitted = st.form_submit_button("Run simulation")

    if not submitted:
        return

    with st.spinner("Running Monte Carlo simulation..."):
        try:
            payload = api_post(
                endpoint="/simulate_tournament",
                payload={
                    "config_path": config_path,
                    "n_simulations": int(n_simulations),
                    "use_calibration": use_calibration,
                    "neutral": neutral,
                },
                api_base_url=api_base_url,
                timeout=120,
            )
        except requests.RequestException as error:
            st.error(f"Tournament simulation failed: {error}")
            return

    results = payload.get("results", [])

    if not results:
        st.warning("Simulation returned no results.")
        return

    results_df = pd.DataFrame(results)

    st.subheader("Simulation results")

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True,
    )

    if "tournament_win_probability" in results_df.columns:
        chart_df = (
            results_df[["team", "tournament_win_probability"]]
            .sort_values("tournament_win_probability", ascending=False)
            .head(12)
            .set_index("team")
        )

        st.subheader("Top tournament win probabilities")
        st.bar_chart(chart_df)


def main() -> None:
    st.title("⚽ World Cup Forecasting and Simulation Platform")

    st.caption(
        "Probabilistic international football forecasting with Elo, "
        "rolling features, calibration, FastAPI serving, and Monte Carlo simulation."
    )

    with st.sidebar:
        st.header("Settings")

        api_base_url = st.text_input(
            "API base URL",
            value=DEFAULT_API_BASE_URL,
        )

        if st.button("Check API health"):
            try:
                health = api_get(
                    endpoint="/health",
                    api_base_url=api_base_url,
                )
                st.success(f"API status: {health['status']}")
            except requests.RequestException as error:
                st.error(f"API unavailable: {error}")

        st.markdown("---")
        st.markdown("Run the API first:")
        st.code("uvicorn wc_forecast.api.main:app --reload")

    tab1, tab2, tab3 = st.tabs(
        [
            "Predict match",
            "Tournament simulation",
            "Model metrics",
        ]
    )

    with tab1:
        render_match_prediction(api_base_url)

    with tab2:
        render_tournament_simulation(api_base_url)

    with tab3:
        render_model_metrics(api_base_url)


if __name__ == "__main__":
    main()
