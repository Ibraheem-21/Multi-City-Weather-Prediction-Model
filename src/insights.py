"""Derived ML insights for UI display."""

from __future__ import annotations

import pandas as pd

from .features import prediction_features_for_target_date
from .model import predict_next_day


def correlation_with_target(frame: pd.DataFrame) -> pd.Series:
    cols = [c for c in frame.columns if c != "target"]
    return frame[cols + ["target"]].corr()["target"].drop("target").sort_values(
        ascending=False
    )


def worst_prediction_days(predictions: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return predictions.sort_values("diff", ascending=False).head(n)


def feature_importance_table(coefficients: dict[str, float]) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "feature": list(coefficients.keys()),
            "coefficient": list(coefficients.values()),
        }
    )
    table["abs_coefficient"] = table["coefficient"].abs()
    return table.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def calendar_day_predictions(
    weather: pd.DataFrame,
    model,
    predictors: list[str],
    month: int,
    day: int,
) -> pd.DataFrame:
    """
    Run the model for every historical occurrence of a calendar day (e.g. Jun 25).

    Each year uses that year's prior-day weather to predict that day's TMAX.
    """
    import calendar

    if day < 1 or day > calendar.monthrange(2000, month)[1]:
        raise ValueError(f"Invalid day {day} for month {month}.")

    mask = (weather.index.month == month) & (weather.index.day == day)
    records: list[dict] = []
    for target in weather.index[mask].sort_values():
        try:
            row, prior = prediction_features_for_target_date(weather, target)
            pred = predict_next_day(model, predictors, row)
            records.append(
                {
                    "year": int(target.year),
                    "predicted_tmax": pred,
                    "actual_tmax": float(weather.loc[target, "temp_max"]),
                    "prior_precip": float(weather.loc[prior, "precip"]),
                }
            )
        except ValueError:
            continue

    if not records:
        label = f"{calendar.month_name[month]} {day}"
        raise ValueError(f"No historical records for {label} in this city's dataset.")

    frame = pd.DataFrame(records)
    frame["error"] = (frame["actual_tmax"] - frame["predicted_tmax"]).abs()
    return frame.sort_values("year").reset_index(drop=True)
