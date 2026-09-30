"""Tests for the deterministic explainable-signal layer (offline, no model)."""

from __future__ import annotations

import pytest

from app.signals import analyse


def test_phishing_url_gets_high_score():
    report = analyse("URGENT: verify your account now at http://192.168.1.10/secure-login")
    assert report.score > 0.4
    assert "credential_request" in {s.key for s in report.signals}
    assert "url_ip_literal" in {s.key for s in report.signals}
    assert "http://192.168.1.10/secure-login" in report.urls
    assert report.riskiest_url is not None


def test_brand_impersonation_outside_real_domain():
    report = analyse("Sign in to unlock your PayPal: http://paypal-secure.tk/login")
    assert "brand_mismatch" in {s.key for s in report.signals}
    assert report.score > 0.5


def test_mundane_message_has_low_score():
    report = analyse("Hi, can we move standup to 10:30 tomorrow? Thanks!")
    assert report.score < 0.15
    assert report.urls == []


def test_real_brand_domain_is_not_flagged_as_brand_mismatch():
    report = analyse("View your order at https://paypal.com/myaccount")
    assert "brand_mismatch" not in {s.key for s in report.signals}


def test_shortener_and_ip_urls_raise_url_risk():
    report = analyse("claim now bit.ly/xyz and http://8.8.8.8/pay")
    found = {s.key for s in report.signals}
    assert "url_shortener" in found
    assert "url_ip_literal" in found


def test_score_is_bounded():
    for message in ["FREE WINNER CLICK http://a.tk/x", "", "ok"]:
        assert 0.0 <= analyse(message).score <= 1.0


def test_empty_and_whitespace_input():
    assert analyse("").signals == []
    assert analyse("   ").signals == []


@pytest.mark.parametrize(
    "message,expected",
    [
        ("text 09069775139 to claim your prize", "premium_action"),
        ("CONGRATULATIONS!!! YOU WON", "repeated_punctuation"),
        ("cl1ck here to verify", "leet_obfuscation"),
        ("your one time password is 4421", "credential_request"),
    ],
)
def test_specific_detectors(message, expected):
    assert expected in {s.key for s in analyse(message).signals}
