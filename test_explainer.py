"""Explanation layer tests: offline narrative always, network never required."""

from __future__ import annotations

from app.config import Settings
from app.explainer import ai_status, build_prompt, explain, local_explanation
from app.model import SpamDetector


def test_local_explanation_is_always_available(detector: SpamDetector):
    text = local_explanation(detector.predict("Congratulations you won a prize!"))
    assert len(text) > 40
    assert "rule-based indicator" in text


def test_explain_with_provider_none_uses_local(detector: SpamDetector):
    settings = Settings()  # dataclass defaults -> AI off
    text, source = explain(detector.predict("URGENT: verify http://a.tk/x"), settings)
    assert source == "local"
    assert text


def test_explain_with_unconfigured_provider_never_raises(detector: SpamDetector):
    # Invalid key configured -> must fall back to local instead of crashing.
    settings = Settings(ai_provider="openai", openai_api_key="put-your-own-key-here")
    text, source = explain(detector.predict("hi"), settings)
    assert source.startswith("local")
    assert text


def test_prompt_contains_scores_and_message(detector: SpamDetector):
    payload = build_prompt(detector.predict("test message"))
    assert "verdict" in payload and "test message" in payload


def test_ai_status_hides_key():
    from app.config import get_settings

    status = ai_status(get_settings())
    assert "put-your-own" not in status
