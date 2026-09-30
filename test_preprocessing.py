"""Tests for the shared text normalisation helpers."""

from __future__ import annotations

from app.preprocessing import (
    base_domain,
    contains_emoji,
    extract_urls,
    looks_like_shortener,
    normalise_text,
    url_host,
)


def test_empty_input_is_safe():
    assert normalise_text("") == ""
    assert normalise_text(None) == ""  # type: ignore[arg-type]


def test_urls_phones_emails_become_placeholders():
    out = normalise_text("Pay via http://evil.example.com or mail a@b.com or call 07700 900123")
    assert "<url>" in out
    assert "<email>" in out
    assert "<number>" in out
    assert "evil.example.com" not in out


def test_whitespace_and_repeated_punctuation_are_collapsed():
    assert normalise_text("hello     world") == "hello world"
    assert normalise_text("wow!!! amazing???") == "wow! amazing?"


def test_extract_urls_and_hosts():
    urls = extract_urls("see bit.ly/abc and https://paypal.com/verify")
    assert "bit.ly/abc" in urls
    assert url_host("https://paypal.com/verify") == "paypal.com"
    assert base_domain("https://login.sub.example.com/x") == "example.com"


def test_shortener_detection():
    assert looks_like_shortener("http://bit.ly/xyz")
    assert not looks_like_shortener("https://www.bbc.co.uk/news")


def test_emoji_detector():
    assert contains_emoji("you won 🎉")
    assert not contains_emoji("you won")
