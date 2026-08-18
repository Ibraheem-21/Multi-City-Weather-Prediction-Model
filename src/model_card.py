"""Model documentation and cross-city comparison helpers."""

from __future__ import annotations

import json

import pandas as pd

from .cities import CITIES, City, list_cities
from .features import ENGINEERED_PREDICTORS
from .model import artifact_paths


FEATURE_DOCS: dict[str, str] = {
    "precip": "Daily precipitation (inches) on the prior day.",
    "temp_max": "Maximum temperature (°F) on the prior day.",
    "temp_min": "Minimum temperature (°F) on the prior day.",
    "month_day_max": "30-day rolling mean of TMAX divided by today's TMAX — captures recent warmth trend.",
    "max_min": "Ratio of prior-day TMAX to TMIN — diurnal range signal.",
}

MODEL_ALGORITHM = "Ridge Regression (L2-regularized linear regression)"
MODEL_ALPHA = 0.1
TARGET_VARIABLE = "Next-day maximum temperature (TMAX, °F)"


def load_all_city_metrics() -> pd.DataFrame:
    rows: list[dict] = []
    for city in list_cities():
        metrics_path = artifact_paths(city.id)["metrics"]
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "city": city.name,
                "city_id": city.id,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "train_rows": metrics["train_rows"],
                "test_rows": metrics["test_rows"],
            }
        )
    return pd.DataFrame(rows).sort_values("rmse")


def build_model_context(
    city: City,
    weather: pd.DataFrame,
    metrics: dict,
    frame: pd.DataFrame,
    *,
    next_pred: float,
    last_row: pd.Series,
    last_date,
) -> dict:
    coef = metrics["coefficients"]
    top_driver = max(coef.items(), key=lambda x: abs(x[1]))
    return {
        "city_name": city.name,
        "city_id": city.id,
        "data_source": city.local_csv or f"NOAA {city.noaa_station}",
        "date_min": str(weather.index.min().date()),
        "date_max": str(weather.index.max().date()),
        "n_days": len(weather),
        "algorithm": MODEL_ALGORITHM,
        "alpha": MODEL_ALPHA,
        "target": TARGET_VARIABLE,
        "predictors": metrics["predictors"],
        "train_rows": metrics["train_rows"],
        "test_rows": metrics["test_rows"],
        "train_end": metrics["train_end"],
        "test_start": metrics["test_start"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "mse": metrics["mse"],
        "intercept": metrics["intercept"],
        "coefficients": coef,
        "top_driver": top_driver[0],
        "top_driver_coef": top_driver[1],
        "next_pred": next_pred,
        "last_date": str(last_date.date()),
        "last_precip": float(last_row["precip"]),
        "last_tmax": float(last_row["temp_max"]),
        "last_tmin": float(last_row["temp_min"]),
        "feature_docs": FEATURE_DOCS,
        "engineered_predictors": ENGINEERED_PREDICTORS,
    }
