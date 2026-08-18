"""Feature engineering for next-day max temperature models."""

from __future__ import annotations

import pandas as pd

BASE_PREDICTORS = ["precip", "temp_max", "temp_min"]
ENGINEERED_PREDICTORS = [
    "precip",
    "temp_max",
    "temp_min",
    "month_day_max",
    "max_min",
]


def add_target(core_weather: pd.DataFrame) -> pd.DataFrame:
    frame = core_weather.copy()
    frame["target"] = frame["temp_max"].shift(-1)
    return frame.iloc[:-1].copy()


def add_engineered_features(core_weather: pd.DataFrame) -> pd.DataFrame:
    frame = core_weather.copy()
    frame["month_max"] = frame["temp_max"].rolling(30).mean()
    frame["month_day_max"] = frame["month_max"] / frame["temp_max"].replace(0, pd.NA)
    # Avoid divide-by-zero on rare 0°F mins; clip denominator.
    safe_min = frame["temp_min"].where(frame["temp_min"].abs() > 1e-6, pd.NA)
    frame["max_min"] = frame["temp_max"] / safe_min
    frame = frame.dropna()
    return frame


def prepare_modeling_frame(core_weather: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: target + engineered features, ready for train/test."""
    with_target = add_target(core_weather)
    return add_engineered_features(with_target)


def prediction_features_from_history(core_weather: pd.DataFrame) -> pd.Series:
    """
    Build engineered predictors for the latest day in a history frame.

    Does not create a training target, so the last calendar day is kept.
    """
    if len(core_weather) < 30:
        raise ValueError("Need at least 30 days of history to engineer features.")

    frame = core_weather.copy()
    frame["month_max"] = frame["temp_max"].rolling(30).mean()
    frame["month_day_max"] = frame["month_max"] / frame["temp_max"].replace(0, pd.NA)
    safe_min = frame["temp_min"].where(frame["temp_min"].abs() > 1e-6, pd.NA)
    frame["max_min"] = frame["temp_max"] / safe_min
    row = frame.iloc[-1]
    if row[ENGINEERED_PREDICTORS].isna().any():
        raise ValueError("Could not engineer features for the latest day.")
    return row


def prediction_features_for_target_date(
    core_weather: pd.DataFrame,
    target_date: pd.Timestamp,
) -> tuple[pd.Series, pd.Timestamp]:
    """
    Build predictors to forecast TMAX on target_date.

    Uses weather through the prior calendar day (model predicts next-day max).
    Returns the feature row and the prior date whose conditions were used.
    """
    target = pd.Timestamp(target_date).normalize()
    prior = target - pd.Timedelta(days=1)
    history = core_weather.loc[:prior]
    if history.empty or prior not in history.index:
        raise ValueError(
            f"No weather for {prior.date()}, the day before {target.date()}. "
            "Pick a date within the dataset range (plus one day after the last row)."
        )
    row = prediction_features_from_history(history.tail(40))
    return row, prior

