"""ML prediction explanations — Ridge math + optional LLM narrative."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from sklearn.linear_model import Ridge

from .ai_assistant import generate_forecast_narrative
from .llm_providers import chat_with_provider


@dataclass
class PredictionExplanation:
    predicted_tmax_f: float
    intercept: float
    feature_contributions: dict[str, float]
    top_positive: str
    top_negative: str
    rule_narrative: str
    ai_narrative: str | None
    confidence_band_f: float
    provider_used: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ridge_contributions(
    model: Ridge,
    predictors: list[str],
    row: pd.Series | dict,
) -> tuple[float, dict[str, float]]:
    """Linear contribution of each feature: coefficient × value."""
    if isinstance(row, dict):
        values = {name: float(row[name]) for name in predictors}
    else:
        values = {name: float(row[name]) for name in predictors}

    intercept = float(model.intercept_)
    contribs = {
        name: float(model.coef_[i]) * values[name]
        for i, name in enumerate(predictors)
    }
    return intercept, contribs


def _top_drivers(contributions: dict[str, float]) -> tuple[str, str]:
    positive = max(contributions.items(), key=lambda x: x[1])
    negative = min(contributions.items(), key=lambda x: x[1])
    return positive[0], negative[0]


def explain_prediction(
    model: Ridge,
    predictors: list[str],
    row: pd.Series | dict,
    context: dict[str, Any],
    *,
    provider_id: str = "rules",
    api_key: str | None = None,
    unit: str = "F",
) -> PredictionExplanation:
    """
    Full ML explanation: Ridge decomposition + rule narrative + optional LLM.
    """
    if isinstance(row, dict):
        values = [float(row[name]) for name in predictors]
    else:
        values = [float(row[name]) for name in predictors]

    predicted = float(model.predict([values])[0])
    intercept, contribs = ridge_contributions(model, predictors, row)
    top_pos, top_neg = _top_drivers(contribs)

    ctx = dict(context)
    ctx["predicted_tmax"] = predicted
    ctx["feature_contributions"] = contribs
    ctx["intercept_contribution"] = intercept
    ctx["top_positive_driver"] = top_pos
    ctx["top_negative_driver"] = top_neg

    rule = generate_forecast_narrative(ctx, unit=unit)
    rule += (
        f" Largest upward pull: **{top_pos}** ({contribs[top_pos]:+.2f}°F). "
        f"Largest downward pull: **{top_neg}** ({contribs[top_neg]:+.2f}°F)."
    )

    ai_narrative: str | None = None
    if provider_id != "rules" and api_key:
        try:
            ai_narrative = chat_with_provider(
                provider_id,
                "Explain this next-day TMAX forecast for a general audience. "
                "Include the predicted high, what yesterday's weather implied, "
                "and the main feature drivers. Mention typical error (MAE).",
                ctx,
                api_key,
            )
        except Exception:
            ai_narrative = None

    mae = float(context.get("mae", 3.5))
    return PredictionExplanation(
        predicted_tmax_f=predicted,
        intercept=intercept,
        feature_contributions=contribs,
        top_positive=top_pos,
        top_negative=top_neg,
        rule_narrative=rule,
        ai_narrative=ai_narrative,
        confidence_band_f=mae,
        provider_used="insights" if ai_narrative else "rules",
    )
