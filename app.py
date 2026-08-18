"""Streamlit UI for multi-city next-day temperature insights."""

from __future__ import annotations

from pathlib import Path
import sys

import calendar

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai_assistant import (
    SUGGESTED_QUESTIONS,
    answer_with_rules,
    generate_limitations_note,
    generate_model_summary,
)
from src.model_card import (
    FEATURE_DOCS,
    MODEL_ALGORITHM,
    MODEL_ALPHA,
    TARGET_VARIABLE,
    build_model_context,
    load_all_city_metrics,
)
from src.cities import get_city, list_cities
from src.data import load_city_weather
from src.features import prepare_modeling_frame, prediction_features_from_history
from src.insights import (
    calendar_day_predictions,
    correlation_with_target,
    feature_importance_table,
    worst_prediction_days,
)
from src.llm_providers import auto_resolve_backend, chat_with_provider
from src.prediction_explainer import explain_prediction
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
    if frame is None:
        frame = prepare_modeling_frame(weather)
    model = artifacts["model"]
    predictors = artifacts["predictors"]

    last = frame.iloc[-1]
    next_pred = predict_next_day(model, predictors, last)
    last_date = frame.index[-1]
    model_ctx = build_model_context(
        city,
        weather,
        metrics,
        frame,
        next_pred=next_pred,
        last_row=last,
        last_date=last_date,
    )
    all_metrics_df = load_all_city_metrics()
    all_metrics_list = all_metrics_df.to_dict("records") if not all_metrics_df.empty else []

    try:
        streamlit_secrets = {k: st.secrets[k] for k in st.secrets}  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        streamlit_secrets = {}
    provider_id, llm_key = auto_resolve_backend(streamlit_secrets)

    explanation = explain_prediction(
        model,
        predictors,
        last,
        model_ctx,
        provider_id=provider_id,
        api_key=llm_key,
        unit=unit,
    )

    # RMSE/MAE are temperature deltas; scale by 5/9 for °C (not absolute conversion).
    if unit == "C":
        rmse_display = metrics["rmse"] * 5.0 / 9.0
        mae_display = metrics["mae"] * 5.0 / 9.0
    else:
        rmse_display = metrics["rmse"]
        mae_display = metrics["mae"]

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

    with st.expander("Forecast insight", expanded=True):
        band = explanation.confidence_band_f
        if unit == "C":
            band = band * 5.0 / 9.0
        insight_text = explanation.rule_narrative
        if explanation.ai_narrative:
            insight_text = explanation.ai_narrative
        st.markdown(insight_text)
        st.caption(f"Typical error band ±{band:.1f} °{unit} (test MAE).")

        contrib = pd.DataFrame(
            {
                "feature": list(explanation.feature_contributions.keys()),
                "contribution_F": list(explanation.feature_contributions.values()),
            }
        )
        if unit == "C":
            contrib["contribution"] = contrib["contribution_F"] * 5.0 / 9.0
            y_col = "contribution"
        else:
            contrib["contribution"] = contrib["contribution_F"]
            y_col = "contribution"
        fig_contrib = px.bar(
            contrib,
            x="feature",
            y=y_col,
            title=f"Ridge feature contributions to forecast (°{unit})",
            labels={y_col: f"Contribution (°{unit})"},
        )
        st.plotly_chart(fig_contrib, use_container_width=True)
        st.caption(
            f"Intercept baseline: {explanation.intercept:.2f}°F · "
            "Each bar = coefficient × feature value."
        )

    tab_forecast, tab_by_date, tab_perf, tab_insights, tab_history, tab_model, tab_ai = st.tabs(
        [
            "Forecast what-if",
            "Predict by date",
            "Performance",
            "Insights",
            "History",
            "About the Model",
            "Ask",
        ]
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

    with tab_by_date:
        st.subheader("Predict by calendar day (e.g. June 25)")
        st.markdown(
            "Choose a **month and day only** — no year. The model runs on every "
            "matching date in the historical record: each year's forecast uses that "
            "year's weather from the day before. You get a typical prediction plus "
            "year-by-year actuals."
        )

        col_month, col_day = st.columns(2)
        month = col_month.selectbox(
            "Month",
            options=list(range(1, 13)),
            format_func=lambda m: calendar.month_name[m],
            index=5,
        )
        max_day = calendar.monthrange(2000, month)[1]
        day = col_day.number_input(
            "Day",
            min_value=1,
            max_value=max_day,
            value=min(25, max_day),
            step=1,
        )

        label = f"{calendar.month_name[month]} {day}"
        try:
            by_year = calendar_day_predictions(weather, model, predictors, month, day)
            pred_median = float(by_year["predicted_tmax"].median())
            actual_mean = float(by_year["actual_tmax"].mean())
            actual_median = float(by_year["actual_tmax"].median())

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Typical model prediction", display_temp(pred_median, unit))
            m2.metric("Historical avg actual", display_temp(actual_mean, unit))
            m3.metric("Historical median actual", display_temp(actual_median, unit))
            m4.metric("Years in record", len(by_year))

            display = by_year.copy()
            if unit == "C":
                display["predicted_tmax"] = display["predicted_tmax"].map(f_to_c)
                display["actual_tmax"] = display["actual_tmax"].map(f_to_c)
                display["error"] = display["error"] * 5.0 / 9.0

            display = display.rename(
                columns={
                    "year": "Year",
                    "predicted_tmax": f"Predicted TMAX (°{unit})",
                    "actual_tmax": f"Actual TMAX (°{unit})",
                    "prior_precip": "Prior-day precip (in)",
                    "error": f"Abs error (°{unit})",
                }
            )
            st.dataframe(display, use_container_width=True, hide_index=True)

            fig_day = go.Figure()
            fig_day.add_trace(
                go.Bar(
                    x=by_year["year"],
                    y=by_year["actual_tmax"].map(
                        lambda v: f_to_c(v) if unit == "C" else v
                    ),
                    name="Actual",
                )
            )
            fig_day.add_trace(
                go.Bar(
                    x=by_year["year"],
                    y=by_year["predicted_tmax"].map(
                        lambda v: f_to_c(v) if unit == "C" else v
                    ),
                    name="Predicted",
                )
            )
            fig_day.update_layout(
                title=f"{label} — predicted vs actual by year",
                barmode="group",
                height=380,
                yaxis_title=f"TMAX (°{unit})",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_day, use_container_width=True)
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

    with tab_model:
        st.subheader("Model card")
        st.markdown(
            f"A transparent summary of the **{city.name}** forecasting pipeline — "
            "useful for demos, interviews, and reproducibility."
        )

        overview_left, overview_right = st.columns(2)
        with overview_left:
            st.markdown("#### Algorithm")
            st.markdown(
                f"- **Method:** {MODEL_ALGORITHM}\n"
                f"- **Regularization (α):** {MODEL_ALPHA}\n"
                f"- **Target:** {TARGET_VARIABLE}\n"
                f"- **Intercept:** {metrics['intercept']:.3f}°F"
            )
            st.markdown("#### Train / test protocol")
            st.markdown(
                f"- **Train:** ≤ `{metrics['train_end']}` ({metrics['train_rows']:,} rows)\n"
                f"- **Test:** ≥ `{metrics['test_start']}` ({metrics['test_rows']:,} rows)\n"
                f"- **Split type:** Time-based (no random shuffle — avoids leakage)"
            )
        with overview_right:
            st.markdown("#### Data source")
            src = city.local_csv or f"NOAA GHCN `{city.noaa_station}`"
            st.markdown(
                f"- **City:** {city.name}\n"
                f"- **Source:** {src}\n"
                f"- **Coverage:** {weather.index.min().date()} → {weather.index.max().date()}\n"
                f"- **Days:** {len(weather):,}"
            )
            st.markdown("#### Held-out accuracy")
            st.markdown(
                f"- **RMSE:** {rmse_display:.2f} °{unit}\n"
                f"- **MAE:** {mae_display:.2f} °{unit}\n"
                f"- **MSE:** {metrics['mse']:.2f} (°F²)"
            )

        st.markdown("#### Pipeline")
        st.code(
            "NOAA / local CSV  →  clean & forward-fill  →  engineer features\n"
            "       →  Ridge(α=0.1)  →  next-day TMAX prediction  →  Streamlit UI / AI narrator",
            language=None,
        )

        st.markdown("#### Feature dictionary")
        feat_rows = [
            {
                "Feature": name,
                "In model": name in predictors,
                "Description": FEATURE_DOCS.get(name, "—"),
                "Coefficient": metrics["coefficients"].get(name),
            }
            for name in FEATURE_DOCS
        ]
        st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)

        st.markdown("#### Limitations")
        st.info(generate_limitations_note())

        if not all_metrics_df.empty:
            st.markdown("#### All cities (test RMSE)")
            compare = all_metrics_df.copy()
            if unit == "C":
                compare["rmse"] = compare["rmse"] * 5.0 / 9.0
                compare["mae"] = compare["mae"] * 5.0 / 9.0
            fig_all = px.bar(
                compare,
                x="city",
                y="rmse",
                title=f"Held-out RMSE by city (°{unit})",
                labels={"rmse": f"RMSE (°{unit})", "city": "City"},
            )
            fig_all.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_all, use_container_width=True)

        st.markdown("#### Tech stack")
        st.markdown(
            "- **ML:** scikit-learn Ridge, pandas feature engineering\n"
            "- **Data:** NOAA GHCN-Daily, bundled Oakland CSV\n"
            "- **App:** Streamlit + Plotly\n"
            "- **Insights:** Ridge contribution decomposition + narrative summaries"
        )

    with tab_ai:
        st.subheader("Ask about the forecast")
        st.markdown(
            "Questions about the current city's prediction, model behavior, and accuracy."
        )

        briefing = explanation.ai_narrative or explanation.rule_narrative
        st.success(briefing)
        st.markdown(generate_model_summary(model_ctx))

        st.markdown("##### Suggested questions")
        qcols = st.columns(2)
        picked: str | None = None
        for i, question in enumerate(SUGGESTED_QUESTIONS):
            if qcols[i % 2].button(question, key=f"suggest_{i}"):
                picked = question

        if "ai_messages" not in st.session_state:
            st.session_state.ai_messages = []

        user_q = st.chat_input("Ask about the forecast or model…")
        if picked:
            user_q = picked

        def respond(question: str) -> str:
            chat_ctx = dict(model_ctx)
            if all_metrics_list:
                chat_ctx["all_cities_metrics"] = all_metrics_list
            if provider_id != "rules" and llm_key:
                try:
                    return chat_with_provider(
                        provider_id, question, chat_ctx, llm_key, mode="chat"
                    )
                except Exception:  # noqa: BLE001
                    pass
            return answer_with_rules(question, chat_ctx, all_metrics_list)

        if user_q:
            st.session_state.ai_messages.append({"role": "user", "content": user_q})
            st.session_state.ai_messages.append({"role": "assistant", "content": respond(user_q)})

        for msg in st.session_state.ai_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])


if __name__ == "__main__":
    main()
