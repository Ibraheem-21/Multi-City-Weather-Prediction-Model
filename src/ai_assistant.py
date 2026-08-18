"""AI-style narratives: rule-based by default, optional LLM when configured."""

from __future__ import annotations

import os


SUGGESTED_QUESTIONS = [
    "Explain tomorrow's forecast in plain English.",
    "What drives this model's predictions?",
    "Where does the model struggle most?",
    "How accurate is this city compared to others?",
]


def _format_coef_sentence(coefficients: dict[str, float]) -> str:
    parts = []
    for name, value in sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True):
        direction = "raises" if value > 0 else "lowers"
        parts.append(f"**{name}** ({direction} forecast, coef {value:+.2f})")
    return "; ".join(parts)


def generate_forecast_narrative(ctx: dict, unit: str = "F") -> str:
    temp = ctx["next_pred"]
    if unit == "C":
        temp = (temp - 32) * 5 / 9
    unit_sym = f"°{unit}"
    return (
        f"For **{ctx['city_name']}**, the Ridge model predicts a next-day high of "
        f"**{temp:.1f} {unit_sym}**, using weather from **{ctx['last_date']}** "
        f"(precip {ctx['last_precip']:.2f} in, TMAX {ctx['last_tmax']:.1f}°F, "
        f"TMIN {ctx['last_tmin']:.1f}°F). "
        f"The strongest linear driver in this city is **{ctx['top_driver']}** "
        f"(coefficient {ctx['top_driver_coef']:+.2f})."
    )


def generate_model_summary(ctx: dict) -> str:
    return (
        f"This is a **{ctx['algorithm']}** with α={ctx['alpha']}, trained on "
        f"**{ctx['train_rows']:,}** days (through {ctx['train_end']}) and evaluated on "
        f"**{ctx['test_rows']:,}** held-out days (from {ctx['test_start']}). "
        f"Target: {ctx['target']}. "
        f"Test **RMSE {ctx['rmse']:.2f}°F** / **MAE {ctx['mae']:.2f}°F**. "
        f"Features: {', '.join(f'`{p}`' for p in ctx['predictors'])}."
    )


def generate_limitations_note() -> str:
    return (
        "The model predicts **one number — next-day TMAX** — not full weather "
        "(no rain probability, wind, or hourly curves). It is **linear**, so extreme "
        "heat waves and cold snaps are often under-estimated. It uses **prior-day "
        "observations only**, not numerical weather prediction or satellite feeds."
    )


def answer_with_rules(question: str, ctx: dict, all_metrics: list[dict] | None = None) -> str:
    q = question.lower().strip()

    if any(w in q for w in ("tomorrow", "forecast", "next day", "predict")):
        return generate_forecast_narrative(ctx)

    if any(w in q for w in ("driver", "feature", "coefficient", "important", "why")):
        return (
            "Prediction drivers (Ridge coefficients): "
            + _format_coef_sentence(ctx["coefficients"])
            + ". "
            "Higher |coefficient| means a stronger effect on the next-day high, "
            "holding other features fixed."
        )

    if any(w in q for w in ("struggle", "error", "worst", "fail", "weak")):
        return (
            f"Typical error on held-out data is **±{ctx['mae']:.1f}°F** (MAE). "
            "Linear models miss **sudden heat spikes** and **marine layer reversals** "
            "because they extrapolate smoothly from yesterday. Check the **Performance** "
            "tab for the largest single-day misses."
        )

    if any(w in q for w in ("accurate", "compare", "other cit", "rmse", "best city")):
        if all_metrics:
            best = min(all_metrics, key=lambda r: r["rmse"])
            lines = [
                f"**{r['city']}**: RMSE {r['rmse']:.2f}°F, MAE {r['mae']:.2f}°F"
                for r in sorted(all_metrics, key=lambda x: x["rmse"])
            ]
            return (
                f"**{ctx['city_name']}** RMSE is **{ctx['rmse']:.2f}°F**. "
                f"Best in the portfolio: **{best['city']}** ({best['rmse']:.2f}°F). "
                "Full ranking:\n- " + "\n- ".join(lines)
            )
        return f"**{ctx['city_name']}** test RMSE: **{ctx['rmse']:.2f}°F**, MAE: **{ctx['mae']:.2f}°F**."

    if any(w in q for w in ("ridge", "model", "algorithm", "how work", "what is")):
        return generate_model_summary(ctx) + " " + generate_limitations_note()

    return (
        "I can explain **forecasts**, **model drivers**, **accuracy**, and **weak spots**. "
        "Try a suggested question above, or ask about "
        f"predictions for **{ctx['city_name']}**."
    )


def get_openai_api_key(user_key: str | None = None) -> str | None:
    if user_key and user_key.strip():
        return user_key.strip()
    return os.environ.get("OPENAI_API_KEY")


def ask_llm(
    question: str,
    ctx: dict,
    api_key: str,
    *,
    model: str = "gpt-4o-mini",
) -> str:
    from .llm_providers import chat_with_provider

    return chat_with_provider("openai", question, ctx, api_key, model=model)


def sanitize_api_key(key: str) -> bool:
    return bool(key.strip()) and len(key.strip()) >= 20
