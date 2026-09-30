"""Plain-English explanations for a verdict.

Two providers, no vendor SDKs required:

* ``local`` (default) - deterministic, offline, built from the rule signals.
* ``openai`` / ``gemini`` - optional public AI APIs called over plain HTTPS.

The LLM is **never required**: when no API key is present, or the network/API
fails, the function falls back to the local narrative, so the prototype always
works out of the box. Keys are read from the environment only (see ``.env.example``)
and are never printed, logged or embedded in source code.
"""

from __future__ import annotations

import json

import requests

from app.config import Settings, get_settings, mask_secret, setup_logging
from app.model import Prediction

logger = setup_logging()

DEFAULT_TIMEOUT = 25

SYSTEM_PROMPT = (
    "You are a cyber-security analyst explaining an SMS/e-mail scam verdict to a "
    "non-technical reader. In at most 90 words: state the verdict, cite the strongest "
    "2-3 indicators, and give one concrete protective action. Never invent URLs or "
    "facts that are not in the input. Reply as plain text, no markdown headings."
)

LOCAL_FALLBACK = (
    "Offline explanation generated from {n} rule-based indicator(s): {indicators}. "
    "The classifier also scored this message {prob:.0%} likely to be spam. {advice}"
)


def local_explanation(prediction: Prediction) -> str:
    """Build a deterministic explanation from the rule signals (always available)."""
    indicators = ", ".join(signal["label"] for signal in prediction.signals[:3])
    if not indicators:
        indicators = "no strong indicators (no links, no urgency or credential language)"
    return LOCAL_FALLBACK.format(
        n=len(prediction.signals),
        indicators=indicators,
        prob=prediction.spam_probability,
        advice=prediction.advice,
    )


def _openai_request(prompt: str, settings: Settings) -> str:
    """Call the OpenAI Chat Completions API (plain HTTPS, no SDK)."""
    url = (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {settings.openai_api_key}",
                 "Content-Type": "application/json"},
        json={
            "model": settings.openai_model,
            "temperature": 0.2,
            "max_tokens": 220,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": prompt}],
        },
        timeout=settings.ai_timeout or DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def _gemini_request(prompt: str, settings: Settings) -> str:
    """Call the Google Gemini ``generateContent`` endpoint (plain HTTPS, no SDK)."""
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{settings.gemini_model}:generateContent")
    response = requests.post(
        url,
        headers={"x-goog-api-key": settings.gemini_api_key,
                 "Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": f"{SYSTEM_PROMPT}\n\n{prompt}"}]}],
              "generationConfig": {"temperature": 0.2, "maxOutputTokens": 220}},
        timeout=settings.ai_timeout or DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def build_prompt(prediction: Prediction) -> str:
    """Assemble the user message sent to the LLM (message may contain a link)."""
    payload = {
        "verdict": prediction.verdict,
        "risk_score": prediction.risk_score,
        "model_probability_spam": prediction.spam_probability,
        "rule_score": prediction.signal_score,
        "indicators": [s["label"] for s in prediction.signals],
        "message": prediction.text[:1200],
    }
    return json.dumps(payload, ensure_ascii=False)


def explain(prediction: Prediction, settings: Settings | None = None) -> tuple[str, str]:
    """Return ``(explanation_text, source)`` where source is ``openai|gemini|local``.

    Any network or authentication problem is logged and silently downgraded to
    the local explanation - a bad API key must never break the prototype.
    """
    settings = settings or get_settings()
    if not settings.ai_configured:
        return local_explanation(prediction), "local"

    prompt = build_prompt(prediction)
    try:
        if settings.ai_provider == "openai":
            return _openai_request(prompt, settings), "openai"
        return _gemini_request(prompt, settings), "gemini"
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        logger.warning(
            "AI explanation via %s failed (%s: %s) -> using local explanation",
            settings.ai_provider, type(exc).__name__, str(exc)[:200],
        )
        return local_explanation(prediction), "local (fallback)"


def ai_status(settings: Settings | None = None) -> str:
    """Short human-readable status line used in the sidebar/footer."""
    settings = settings or get_settings()
    if settings.ai_configured:
        model = settings.openai_model if settings.ai_provider == "openai" else settings.gemini_model
        return f"AI explanations: {settings.ai_provider} ({model}), key {mask_secret(settings.ai_key)}"
    return "AI explanations: off (offline rule-based explanations; add a key in .env to enable)"
