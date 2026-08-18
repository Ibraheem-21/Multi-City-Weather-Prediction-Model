"""Quick Gemini connectivity check — run from project root:

    python scripts/test_gemini.py

Reads GEMINI_API_KEY from .streamlit/secrets.toml or the environment.
Does NOT print your key.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_key() -> str | None:
    secrets_path = ROOT / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            import tomllib

            with secrets_path.open("rb") as fh:
                data = tomllib.load(fh)
            key = data.get("GEMINI_API_KEY")
            if key:
                return str(key).strip()
        except Exception as exc:  # noqa: BLE001
            print(f"Could not read secrets.toml: {exc}")
    env = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return env.strip() if env else None


def main() -> None:
    key = _load_key()
    if not key:
        print("FAIL: No GEMINI_API_KEY in .streamlit/secrets.toml or environment.")
        sys.exit(1)

    prefix = key[:3]
    print(f"Key loaded (prefix={prefix}..., length={len(key)})")

    os.environ["GEMINI_DEBUG"] = "1"

    from src.llm_providers import chat_with_provider

    ctx = {"city_name": "Oakland, CA", "next_pred": 74.0, "rmse": 4.55, "mae": 3.32}
    try:
        reply = chat_with_provider(
            "gemini",
            "Reply with exactly the word WORKING.",
            ctx,
            key,
        )
        print("SUCCESS")
        print(f"Reply: {reply.strip()[:120]}")
    except Exception as exc:
        print("FAIL")
        print(f"Error: {type(exc).__name__}: {exc}")
        print()
        print("If you see ACCESS_TOKEN_TYPE_UNSUPPORTED, this is a known Google")
        print("backend issue with new AQ.* keys — try regenerating in AI Studio")
        print("or report at https://discuss.ai.google.dev/")
        sys.exit(1)


if __name__ == "__main__":
    main()
