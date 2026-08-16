"""Derived ML insights for UI display."""

from __future__ import annotations

import pandas as pd


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
