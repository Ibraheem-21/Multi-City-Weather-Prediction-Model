"""Streamlit UI for multi-city next-day temperature insights."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cities import get_city, list_cities
from src.data import load_city_weather
from src.features import ENGINEERED_PREDICTORS, prediction_features_from_history
from src.insights import (
    correlation_with_target,
    feature_importance_table,
    worst_prediction_days,
)
from src.model import artifact_paths, load_artifacts, predict_next_day, train_city

st.set_page_config(
    page_title="City Weather Insights",
    page_icon="🌡️",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def cached_weather(city_id: str, refresh: bool) -> pd.DataFrame:
    return load_city_weather(city_id, refresh=refresh)


def ensure_trained(city_id: str, weather: pd.DataFrame) -> dict:
    paths = artifact_paths(city_id)
    if not paths["model"].exists():
        with st.spinner(f"Training model for {get_city(city_id).name}..."):
            train_city(city_id, weather, persist=True)
    return load_artifacts(city_id)


def main() -> None:
    st.title("Multi-City Weather Model")
    st.caption(
        "Ridge regression predicting tomorrow’s max temperature (°F) from today’s "
        "conditions. Oakland uses the bundled airport CSV; other cities download "
        "NOAA GHCN-Daily station history (Open-Meteo as fallback)."
    )

    cities = list_cities()
    labels = {c.id: c.name for c in cities}
    default_idx = list(labels.keys()).index("oakland") if "oakland" in labels else 0

    with st.sidebar:
        st.header("City")
        city_id = st.selectbox(
            "Select city",
            options=list(labels.keys()),
            format_func=lambda x: labels[x],
            index=default_idx,
        )
        refresh = st.checkbox("Refresh remote weather data", value=False)
        if st.button("Retrain model", use_container_width=True):
            weather = cached_weather(city_id, refresh=True)
            cached_weather.clear()
            with st.spinner("Retraining..."):
                train_city(city_id, weather, persist=True)
            st.success("Model retrained.")
            st.rerun()

        st.divider()
        st.markdown("**Available cities**")
        for city in cities:
            src = "local CSV" if city.local_csv else f"NOAA {city.noaa_station}"
            st.write(f"- {city.name} (`{src}`)")

    city = get_city(city_id)

    try:
        weather = cached_weather(city_id, refresh=refresh)
        artifacts = ensure_trained(city_id, weather)
    except Exception as exc:  # noqa: BLE001 — show API/data failures in UI
        st.error(f"Could not load data or model for {city.name}: {exc}")
        st.stop()

    metrics = artifacts["metrics"]
    predictions = artifacts["predictions"]
    frame = artifacts["frame"]
    model = artifacts["model"]
    predictors = artifacts["predictors"]

    last = frame.iloc[-1]
    next_pred = predict_next_day(model, predictors, last)
    last_date = frame.index[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Next-day TMAX forecast", f"{next_pred:.1f} °F")
    c2.metric("Test RMSE", f"{metrics['rmse']:.2f} °F")
    c3.metric("Test MAE", f"{metrics['mae']:.2f} °F")
    c4.metric(
        "Data coverage",
        f"{weather.index.min().date()} → {weather.index.max().date()}",
    )

    st.info(
        f"Forecast uses conditions from **{last_date.date()}** in {city.name} "
        f"(precip={last['precip']:.2f} in, TMAX={last['temp_max']:.1f}°F, "
        f"TMIN={last['temp_min']:.1f}°F)."
    )

    tab_forecast, tab_perf, tab_insights, tab_history = st.tabs(
        ["Forecast what-if", "Performance", "Insights", "History"]
    )

    with tab_forecast:
        st.subheader("What-if next-day forecast")
        col_a, col_b, col_c = st.columns(3)
        precip = col_a.number_input(
            "Precip (inches)",
            min_value=0.0,
            max_value=10.0,
            value=float(last["precip"]),
            step=0.05,
        )
        temp_max = col_b.number_input(
            "Today TMAX (°F)",
            min_value=-40.0,
            max_value=130.0,
            value=float(last["temp_max"]),
            step=0.5,
        )
        temp_min = col_c.number_input(
            "Today TMIN (°F)",
            min_value=-50.0,
            max_value=100.0,
            value=float(last["temp_min"]),
            step=0.5,
        )

        window = weather.tail(40).copy()
        window.loc[window.index[-1], "precip"] = precip
        window.loc[window.index[-1], "temp_max"] = temp_max
        window.loc[window.index[-1], "temp_min"] = temp_min
        try:
            row = prediction_features_from_history(window)
            custom_pred = predict_next_day(model, predictors, row)
            st.metric("Predicted next-day TMAX", f"{custom_pred:.1f} °F")
            st.dataframe(
                pd.DataFrame(
                    {name: [float(row[name])] for name in ENGINEERED_PREDICTORS}
                ),
                use_container_width=True,
            )
        except ValueError as exc:
            st.warning(str(exc))

    with tab_perf:
        st.subheader("Actual vs predicted (held-out test)")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=predictions["actual"],
                name="Actual",
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=predictions["predictions"],
                name="Predicted",
                mode="lines",
            )
        )
        fig.update_layout(
            height=420,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="TMAX (°F)",
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)
        left.write("Largest absolute errors")
        left.dataframe(worst_prediction_days(predictions, 10), use_container_width=True)
        right.write("Summary")
        right.json(
            {
                "train_rows": metrics["train_rows"],
                "test_rows": metrics["test_rows"],
                "mse": round(metrics["mse"], 3),
                "rmse": round(metrics["rmse"], 3),
                "mae": round(metrics["mae"], 3),
                "train_end": metrics["train_end"],
                "test_start": metrics["test_start"],
            }
        )

    with tab_insights:
        st.subheader("Model coefficients")
        coef_table = feature_importance_table(metrics["coefficients"])
        fig_coef = px.bar(
            coef_table,
            x="feature",
            y="coefficient",
            title="Ridge coefficients",
        )
        st.plotly_chart(fig_coef, use_container_width=True)

        st.subheader("Correlation with next-day TMAX")
        corr = correlation_with_target(frame).reset_index()
        corr.columns = ["feature", "correlation"]
        fig_corr = px.bar(corr, x="feature", y="correlation")
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab_history:
        st.subheader("Temperature history")
        hist = weather.tail(365 * 5).copy()
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Scatter(x=hist.index, y=hist["temp_max"], name="TMAX", mode="lines")
        )
        fig_hist.add_trace(
            go.Scatter(x=hist.index, y=hist["temp_min"], name="TMIN", mode="lines")
        )
        fig_hist.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="°F",
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Annual precipitation")
        annual = weather["precip"].groupby(weather.index.year).sum().reset_index()
        annual.columns = ["year", "precip_inches"]
        fig_p = px.bar(annual.tail(40), x="year", y="precip_inches")
        st.plotly_chart(fig_p, use_container_width=True)


if __name__ == "__main__":
    main()
