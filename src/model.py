"""Train, evaluate, persist, and score Ridge weather models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .cities import ARTIFACTS_DIR
from .features import ENGINEERED_PREDICTORS, prepare_modeling_frame


@dataclass
class TrainResult:
    city_id: str
    predictors: list[str]
    train_rows: int
    test_rows: int
    mse: float
    rmse: float
    mae: float
    coefficients: dict[str, float]
    intercept: float
    train_end: str
    test_start: str


def time_split(
    frame: pd.DataFrame,
    train_end: str = "2020-12-31",
    test_start: str = "2021-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame.loc[:train_end]
    test = frame.loc[test_start:]
    if train.empty or test.empty:
        raise ValueError(
            f"Empty train/test split (train={len(train)}, test={len(test)}). "
            "Check date coverage for this city."
        )
    return train, test


def train_ridge(
    frame: pd.DataFrame,
    city_id: str,
    predictors: list[str] | None = None,
    alpha: float = 0.1,
    train_end: str = "2020-12-31",
    test_start: str = "2021-01-01",
) -> tuple[Ridge, TrainResult, pd.DataFrame]:
    predictors = predictors or ENGINEERED_PREDICTORS
    train, test = time_split(frame, train_end=train_end, test_start=test_start)

    model = Ridge(alpha=alpha)
    model.fit(train[predictors], train["target"])
    predictions = model.predict(test[predictors])

    mse = float(mean_squared_error(test["target"], predictions))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(test["target"], predictions))

    combined = pd.DataFrame(
        {
            "actual": test["target"],
            "predictions": predictions,
        },
        index=test.index,
    )
    combined["diff"] = (combined["actual"] - combined["predictions"]).abs()

    result = TrainResult(
        city_id=city_id,
        predictors=list(predictors),
        train_rows=len(train),
        test_rows=len(test),
        mse=mse,
        rmse=rmse,
        mae=mae,
        coefficients={
            name: float(coef) for name, coef in zip(predictors, model.coef_)
        },
        intercept=float(model.intercept_),
        train_end=train_end,
        test_start=test_start,
    )
    return model, result, combined


def artifact_paths(city_id: str) -> dict[str, Path]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "model": ARTIFACTS_DIR / f"{city_id}_ridge.joblib",
        "metrics": ARTIFACTS_DIR / f"{city_id}_metrics.json",
        "predictions": ARTIFACTS_DIR / f"{city_id}_test_predictions.csv",
        "frame": ARTIFACTS_DIR / f"{city_id}_modeling_frame.csv",
    }


def save_artifacts(
    city_id: str,
    model: Ridge,
    result: TrainResult,
    combined: pd.DataFrame,
    frame: pd.DataFrame,
) -> dict[str, Path]:
    paths = artifact_paths(city_id)
    joblib.dump(
        {"model": model, "predictors": result.predictors, "city_id": city_id},
        paths["model"],
    )
    paths["metrics"].write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    combined.to_csv(paths["predictions"])
    frame.to_csv(paths["frame"])
    return paths


def load_artifacts(city_id: str) -> dict:
    paths = artifact_paths(city_id)
    if not paths["model"].exists():
        raise FileNotFoundError(
            f"No trained model for '{city_id}'. Run: python train.py --city {city_id}"
        )
    bundle = joblib.load(paths["model"])
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    predictions = pd.read_csv(paths["predictions"], index_col=0, parse_dates=True)
    frame = pd.read_csv(paths["frame"], index_col=0, parse_dates=True)
    return {
        "model": bundle["model"],
        "predictors": bundle["predictors"],
        "metrics": metrics,
        "predictions": predictions,
        "frame": frame,
        "paths": paths,
    }


def predict_next_day(
    model: Ridge,
    predictors: list[str],
    row: pd.Series | dict,
) -> float:
    if isinstance(row, dict):
        values = [float(row[name]) for name in predictors]
    else:
        values = [float(row[name]) for name in predictors]
    return float(model.predict([values])[0])


def train_city(
    city_id: str,
    core_weather: pd.DataFrame,
    *,
    persist: bool = True,
) -> tuple[Ridge, TrainResult, pd.DataFrame, pd.DataFrame]:
    frame = prepare_modeling_frame(core_weather)
    model, result, combined = train_ridge(frame, city_id=city_id)
    if persist:
        save_artifacts(city_id, model, result, combined, frame)
    return model, result, combined, frame
