"""Shared pytest fixtures: one trained detector reused by the whole suite."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repository root importable when pytest is invoked from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.model import SpamDetector

PHISHING = (
    "URGENT: We detected a suspicious login. Your account will be closed in 24 hours. "
    "Verify your identity now http://192.168.1.10/secure-login"
)
PRIZE_SCAM = (
    "Congratulations! You have WON a 500 gift voucher. Click here to claim your FREE "
    "prize now! Reply STOP to unsubscribe"
)
NORMAL_MSG = "Hi team, can we move standup to 10:30 tomorrow? I will send the invite"


@pytest.fixture(scope="session")
def detector() -> SpamDetector:
    """Load the model artifact, training it on first use (once per pytest run)."""
    return SpamDetector.load(auto_train=True)
