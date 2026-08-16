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


def f_to_c(value: float) -> float:
    return (value - 32.0) * 5.0 / 9.0


def c_to_f(value: float) -> float:
    return value * 9.0 / 5.0 + 32.0


def convert_temp(value: float, *, to_unit: str) -> float:
    return f_to_c(value) if to_unit == "C" else value


def display_temp(value_f: float, unit: str, digits: int = 1) -> str:
    value = convert_temp(value_f, to_unit=unit)
    return f"{value:.{digits}f} °{unit}"


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
        "Ridge regression predicting tomorrow’s max temperature from today’s "
        "conditions. Oakland uses the bundled airport CSV; other cities download "
        "NOAA GHCN-Daily station history (Open-Meteo as fallback). "
        "The model always trains in °F; the unit toggle only changes display."
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
        unit = st.radio(
            "Temperature unit",
            options=["F", "C"],
            format_func=lambda u: "Fahrenheit (°F)" if u == "F" else "Celsius (°C)",
            horizontal=False,
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
    rmse_display = convert_temp(metrics["rmse"], to_unit=unit) if unit == "C" else metrics["rmse"]
    mae_display = convert_temp(metrics["mae"], to_unit=unit) if unit == "C" else metrics["mae"]
    # RMSE/MAE are temperature deltas; convert as differences, not absolute temps.
    if unit == "C":
        rmse_display = metrics["rmse"] * 5.0 / 9.0
        mae_display = metrics["mae"] * 5.0 / 9.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Next-day TMAX forecast", display_temp(next_pred, unit))
    c2.metric("Test RMSE", f"{rmse_display:.2f} °{unit}")
    c3.metric("Test MAE", f"{mae_display:.2f} °{unit}")
    c4.metric(
        "Data coverage",
        f"{weather.index.min().date()} → {weather.index.max().date()}",
    )

    st.info(
        f"Forecast uses conditions from **{last_date.date()}** in {city.name} "
        f"(precip={last['precip']:.2f} in, "
        f"TMAX={display_temp(float(last['temp_max']), unit)}, "
        f"TMIN={display_temp(float(last['temp_min']), unit)})."
    )

    tab_forecast, tab_perf, tab_insights, tab_history = st.tabs(
        ["Forecast what-if", "Performance", "Insights", "History"]
    )

    with tab_forecast:
        st.subheader("What-if next-day forecast")
        st.markdown(
            "This tab lets you **simulate a different today** and ask: *if precip and "
            "temperatures looked like this, what max temperature would the model "
            "expect tomorrow?* It does not pull a live forecast from a weather service — "
            "it scores your hypothetical inputs with the trained Ridge model. "
            "Engineered features (`month_day_max`, `max_min`) are rebuilt from recent "
            "history plus your overrides."
        )
        col_a, col_b, col_c = st.columns(3)
        precip = col_a.number_input(
            "Precip (inches)",
            min_value=0.0,
            max_value=10.0,
            value=float(last["precip"]),
            step=0.05,
        )

        default_tmax = convert_temp(float(last["temp_max"]), to_unit=unit)
        default_tmin = convert_temp(float(last["temp_min"]), to_unit=unit)
        if unit == "C":
            tmax_bounds = (-40.0, 55.0)
            tmin_bounds = (-45.0, 40.0)
            step = 0.5
        else:
            tmax_bounds = (-40.0, 130.0)
            tmin_bounds = (-50.0, 100.0)
            step = 0.5

        temp_max_input = col_b.number_input(
            f"Today TMAX (°{unit})",
            min_value=tmax_bounds[0],
            max_value=tmax_bounds[1],
            value=float(round(default_tmax, 2)),
            step=step,
        )
        temp_min_input = col_c.number_input(
            f"Today TMIN (°{unit})",
            min_value=tmin_bounds[0],
            max_value=tmin_bounds[1],
            value=float(round(default_tmin, 2)),
            step=step,
        )

        temp_max_f = c_to_f(temp_max_input) if unit == "C" else temp_max_input
        temp_min_f = c_to_f(temp_min_input) if unit == "C" else temp_min_input

        window = weather.tail(40).copy()
        window.loc[window.index[-1], "precip"] = precip
        window.loc[window.index[-1], "temp_max"] = temp_max_f
        window.loc[window.index[-1], "temp_min"] = temp_min_f
        try:
            row = prediction_features_from_history(window)
            custom_pred = predict_next_day(model, predictors, row)
            st.metric("Predicted next-day TMAX", display_temp(custom_pred, unit))
            feature_view = {
                "precip": [float(row["precip"])],
                "temp_max": [convert_temp(float(row["temp_max"]), to_unit=unit)],
                "temp_min": [convert_temp(float(row["temp_min"]), to_unit=unit)],
                "month_day_max": [float(row["month_day_max"])],
                "max_min": [float(row["max_min"])],
            }
            st.caption(
                f"Model features for this scenario "
                f"(temps shown in °{unit}; ratios stay unitless)."
            )
            st.dataframe(pd.DataFrame(feature_view), use_container_width=True)
        except ValueError as exc:
            st.warning(str(exc))

    with tab_perf:
        st.subheader("Actual vs predicted (held-out test)")
        plot_preds = predictions.copy()
        if unit == "C":
            plot_preds["actual"] = plot_preds["actual"].map(f_to_c)
            plot_preds["predictions"] = plot_preds["predictions"].map(f_to_c)
            plot_preds["diff"] = (plot_preds["actual"] - plot_preds["predictions"]).abs()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=plot_preds.index,
                y=plot_preds["actual"],
                name="Actual",
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_preds.index,
                y=plot_preds["predictions"],
                name="Predicted",
                mode="lines",
            )
        )
        fig.update_layout(
            height=420,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title=f"TMAX (°{unit})",
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)
        left.write("Largest absolute errors")
        left.dataframe(worst_prediction_days(plot_preds, 10), use_container_width=True)
        right.write("Summary")
        right.json(
            {
                "train_rows": metrics["train_rows"],
                "test_rows": metrics["test_rows"],
                "mse": round(metrics["mse"], 3),
                "rmse_F": round(metrics["rmse"], 3),
                "mae_F": round(metrics["mae"], 3),
                "rmse_display": round(rmse_display, 3),
                "mae_display": round(mae_display, 3),
                "display_unit": unit,
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
        if unit == "C":
            hist = hist.copy()
            hist["temp_max"] = hist["temp_max"].map(f_to_c)
            hist["temp_min"] = hist["temp_min"].map(f_to_c)
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
            yaxis_title=f"°{unit}",
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
