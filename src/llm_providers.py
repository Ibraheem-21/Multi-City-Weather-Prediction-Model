"""LLM insight generation — used silently when keys are configured."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a concise weather analyst for a temperature forecasting app. "
    "Use ONLY the JSON context provided. Explain predictions in 2-4 complete sentences. "
    "Mention key drivers and typical error (RMSE/MAE). No hype. "
    "Do not use markdown code fences or backticks."
)

CHAT_SYSTEM_PROMPT = (
    "You are a helpful weather-ML assistant for a Ridge regression forecasting app. "
    "Use ONLY the JSON context provided. Answer the user's question in 2-6 complete "
    "sentences. Mention RMSE/MAE when discussing accuracy. "
    "Do not use markdown code fences or backticks. Always finish your answer."
)

# Gemini 3.x counts thinking + output against max_output_tokens; keep output headroom.
_GEMINI_MAX_OUTPUT_TOKENS = 1024

# Preferred models (newest first). _chat_gemini tries each until one works.
GEMINI_MODELS = (
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
)

# Priority order when auto-selecting a configured backend.
AUTO_PROVIDER_ORDER = ("gemini", "groq", "openrouter", "openai")

_PROVIDER_ENV = {
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
}

_PROVIDER_MODEL = {
    "gemini": GEMINI_MODELS[0],
    "groq": "llama-3.1-8b-instant",
    "openrouter": "google/gemma-2-9b-it:free",
    "openai": "gpt-4o-mini",
}

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


def resolve_api_key(
    provider_id: str,
    user_key: str | None = None,
    secrets: dict[str, str] | None = None,
) -> str | None:
    if provider_id == "rules":
        return None
    if user_key and user_key.strip():
        return user_key.strip()
    env_var = _PROVIDER_ENV.get(provider_id)
    if secrets and env_var and env_var in secrets:
        val = secrets[env_var]
        if val:
            return str(val).strip()
    raw = os.environ.get(env_var) if env_var else None
    return raw.strip() if raw else None


def auto_resolve_backend(
    secrets: dict[str, str] | None = None,
) -> tuple[str, str | None]:
    """Pick the first configured backend, else rule-based fallback."""
    for provider_id in AUTO_PROVIDER_ORDER:
        key = resolve_api_key(provider_id, secrets=secrets)
        if key:
            return provider_id, key
    return "rules", None


def _gemini_debug() -> bool:
    return os.environ.get("GEMINI_DEBUG", "").lower() in ("1", "true", "yes")


def _last_gemini_error() -> str | None:
    return getattr(_chat_gemini, "_last_error", None)


def _gemini_thinking_config():
    """Disable internal reasoning so output tokens are not eaten on Flash models."""
    from google.genai import types

    return types.ThinkingConfig(thinking_budget=0)


def _extract_gemini_text(response: Any) -> str:
    """Return visible answer text, skipping internal thought parts."""
    parts: list[str] = []
    try:
        for candidate in response.candidates or []:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in content.parts or []:
                if getattr(part, "thought", False):
                    continue
                text = getattr(part, "text", None)
                if text:
                    parts.append(text)
    except Exception:  # noqa: BLE001
        pass
    if parts:
        return "\n".join(parts).strip()
    return (getattr(response, "text", None) or "").strip()


def _gemini_generate_config(*, system_instruction: str):
    from google.genai import types

    return types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.35,
        max_output_tokens=_GEMINI_MAX_OUTPUT_TOKENS,
        thinking_config=_gemini_thinking_config(),
    )


def _chat_openai_compatible(
    url: str,
    api_key: str,
    model: str,
    user_content: str,
    *,
    extra_headers: dict[str, str] | None = None,
    system_instruction: str = SYSTEM_PROMPT,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    response = requests.post(
        url,
        headers=headers,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 1024,
            "temperature": 0.35,
        },
        timeout=60,
    )
    if not response.ok:
        raise RuntimeError(f"Insight service error ({response.status_code})")
    payload = response.json()
    return payload["choices"][0]["message"]["content"].strip()


def _chat_gemini_sdk(
    api_key: str, user_content: str, *, system_instruction: str
) -> str:
    """Official SDK — best compatibility with new AQ.* auth keys."""
    from google import genai

    client = genai.Client(api_key=api_key)
    config = _gemini_generate_config(system_instruction=system_instruction)
    last_err = ""
    for model in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_content,
                config=config,
            )
            text = _extract_gemini_text(response)
            if text:
                return text
        except Exception as exc:  # noqa: BLE001
            last_err = f"{model}: {type(exc).__name__}: {exc}"
            if _gemini_debug():
                logger.warning("Gemini SDK %s", last_err)
            continue
    raise RuntimeError(last_err or "Gemini SDK: no model responded")


def _chat_gemini_rest(
    api_key: str, user_content: str, *, system_instruction: str
) -> str:
    """Native REST fallback — x-goog-api-key header (AQ.* keys)."""
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    body = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"parts": [{"text": user_content}]}],
        "generationConfig": {
            "temperature": 0.35,
            "maxOutputTokens": _GEMINI_MAX_OUTPUT_TOKENS,
        },
        "thinkingConfig": {"thinkingBudget": 0},
    }
    last_err = ""
    for model in GEMINI_MODELS:
        url = f"{_GEMINI_BASE}/models/{model}:generateContent"
        try:
            response = requests.post(url, headers=headers, json=body, timeout=60)
            if not response.ok:
                err_msg = response.json().get("error", {}).get("message", response.text[:200])
                last_err = f"{model}: HTTP {response.status_code} — {err_msg}"
                if _gemini_debug():
                    logger.warning("Gemini REST %s", last_err)
                continue
            payload = response.json()
            parts = payload["candidates"][0]["content"]["parts"]
            text_parts = [
                p["text"].strip()
                for p in parts
                if p.get("text") and not p.get("thought")
            ]
            text = "\n".join(text_parts).strip()
            if text:
                return text
        except Exception as exc:  # noqa: BLE001
            last_err = f"{model}: {type(exc).__name__}: {exc}"
            if _gemini_debug():
                logger.warning("Gemini REST %s", last_err)
            continue
    raise RuntimeError(last_err or "Gemini REST: no model responded")


def _chat_gemini(api_key: str, user_content: str, *, system_instruction: str) -> str:
    """Try official SDK first, then native REST."""
    errors: list[str] = []
    for attempt in (_chat_gemini_sdk, _chat_gemini_rest):
        try:
            return attempt(api_key, user_content, system_instruction=system_instruction)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
            continue
    msg = "; ".join(errors) if errors else "Insight service unavailable"
    _chat_gemini._last_error = msg  # type: ignore[attr-defined]
    raise RuntimeError(msg)


def chat_with_provider(
    provider_id: str,
    question: str,
    context: dict[str, Any],
    api_key: str | None,
    *,
    model: str | None = None,
    mode: str = "brief",
) -> str:
    if provider_id == "rules":
        raise ValueError("rules backend does not use chat_with_provider")
    if not api_key:
        raise RuntimeError("Insight service not configured")

    system_instruction = CHAT_SYSTEM_PROMPT if mode == "chat" else SYSTEM_PROMPT
    user_content = (
        f"Context:\n{json.dumps(context, indent=2, default=str)}\n\nQuestion: {question}"
    )

    if provider_id == "gemini":
        return _chat_gemini(api_key, user_content, system_instruction=system_instruction)
    model = model or _PROVIDER_MODEL[provider_id]
    if provider_id == "groq":
        return _chat_openai_compatible(
            "https://api.groq.com/openai/v1/chat/completions",
            api_key,
            model,
            user_content,
            system_instruction=system_instruction,
        )
    if provider_id == "openrouter":
        return _chat_openai_compatible(
            "https://openrouter.ai/api/v1/chat/completions",
            api_key,
            model,
            user_content,
            extra_headers={
                "HTTP-Referer": "https://multi-city-weather-prediction-model.streamlit.app",
                "X-Title": "Multi-City Weather Model",
            },
            system_instruction=system_instruction,
        )
    if provider_id == "openai":
        return _chat_openai_compatible(
            "https://api.openai.com/v1/chat/completions",
            api_key,
            model,
            user_content,
            system_instruction=system_instruction,
        )
    raise ValueError(f"Unknown provider: {provider_id}")
