"""End-to-end model tests: obvious spam is caught, normal mail passes."""

from __future__ import annotations

import pytest

from app.model import SpamDetector, build_pipeline, train_and_evaluate
from conftest import NORMAL_MSG, PHISHING, PRIZE_SCAM


def test_pipeline_builds():
    pipeline = build_pipeline()
    assert "features" in pipeline.named_steps and "clf" in pipeline.named_steps


def test_obvious_phishing_is_spam(detector: SpamDetector):
    prediction = detector.predict(PHISHING)
    assert prediction.verdict == "spam"
    assert prediction.risk_score >= detector.settings.spam_threshold
    assert prediction.spam_probability > 0.5
    assert prediction.reasons, "spam verdict must come with explanations"


def test_prize_scam_is_spam(detector: SpamDetector):
    assert detector.predict(PRIZE_SCAM).verdict == "spam"


def test_normal_message_is_ham(detector: SpamDetector):
    prediction = detector.predict(NORMAL_MSG)
    assert prediction.verdict == "ham"
    assert prediction.risk_score < detector.settings.spam_threshold


def test_empty_input_is_ham_and_never_raises(detector: SpamDetector):
    prediction = detector.predict("")
    assert prediction.verdict == "ham"
    assert prediction.risk_score == 0.0


def test_prediction_is_json_serialisable(detector: SpamDetector):
    import json

    payload = json.loads(detector.predict(PHISHING).to_json())
    for key in ("verdict", "risk_score", "spam_probability", "signal_score",
                "reasons", "advice", "confidence"):
        assert key in payload


def test_every_verdict_has_advice(detector: SpamDetector):
    for message in (PHISHING, NORMAL_MSG, "maybe"):
        assert detector.predict(message).advice.strip()


def test_load_does_not_retrain_when_artifact_exists(detector: SpamDetector):
    again = SpamDetector.load(auto_train=False)
    assert again.pipeline is not None


def test_train_and_evaluate_reports_expected_metrics():
    _, metrics = train_and_evaluate()
    test = metrics["test_metrics"]
    assert set(test) == {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
    assert 0.9 <= test["accuracy"] <= 1.0
    assert test["roc_auc"] >= 0.95
    assert len(metrics["confusion_matrix"]) == 2


@pytest.mark.parametrize("threshold_check", [0.0, 1.0])
def test_threshold_bounds_are_valid(threshold_check):
    assert 0.0 <= threshold_check <= 1.0
