"""Scikit-learn pipeline: training, evaluation, persistence and prediction.

Model design
------------
``FeatureUnion`` of two complementary text vectorisers feeding a calibrated
logistic regression classifier:

* **word n-grams (1-2)**  -> topical vocabulary ("free entry", "claim prize")
* **char n-grams (3-5)**  -> obfuscations and typos ("cl1ck", "paypal", "baŋk")

``class_weight="balanced"`` keeps the minority spam class honest (only ~13% of
messages are spam). A tiny grid search picks the regularisation strength.

The learned probability is then blended with the deterministic rule score from
:mod:`app.signals`::

    risk = 0.8 * P(spam | text) + 0.2 * signal_score

so the ranking stays data driven while the *explanations* stay human readable.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder

from app.config import Settings, get_settings, setup_logging
from app.dataset import load_dataset, summarise
from app.preprocessing import normalise_text
from app.signals import HAM, SPAM, SignalReport, analyse

logger = setup_logging()

MODEL_VERSION = "1.0.0"
ML_WEIGHT = 0.80  # documented blend between the learned model and the rule layer


@dataclass
class Prediction:
    """Everything the interfaces (CLI / UI / batch) need to render one verdict."""

    text: str
    verdict: str                 # "spam" | "review" | "ham"
    risk_score: float            # blended 0-1 score
    spam_probability: float      # model P(spam | text)
    signal_score: float          # rule layer 0-1 score
    confidence: str              # high | medium | low
    reasons: list[str] = field(default_factory=list)
    signals: list[dict[str, Any]] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    riskiest_url: str | None = None
    advice: str = ""
    model_version: str = MODEL_VERSION
    latency_ms: float = 0.0

    @property
    def emoji(self) -> str:
        """Single glyph used by the CLI and the Streamlit banner."""
        return {"spam": "🚨", "review": "⚠️", "ham": "✅"}.get(self.verdict, "•")

    @property
    def colour(self) -> str:
        """Hex colour used by the Streamlit progress bar."""
        return {"spam": "#e74c3c", "review": "#f39c12", "ham": "#27ae60"}[self.verdict]

    def as_dict(self) -> dict[str, Any]:
        """Flat dict for CSV export (signals and reasons joined as text)."""
        data = asdict(self)
        data["signals"] = "; ".join(str(s["label"]) for s in self.signals)
        data["reasons"] = "; ".join(self.reasons)
        return data

    def to_json(self) -> str:
        """Compact JSON used by ``cli.py --json``."""
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def build_pipeline(c: float = 1.0, random_state: int = 42) -> Pipeline:
    """Create the untrained classifier pipeline described in the module docstring."""
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    max_features=20_000,
                    preprocessor=normalise_text,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=30_000,
                ),
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("features", features),
            (
                "clf",
                LogisticRegression(
                    C=c,
                    class_weight="balanced",
                    max_iter=1000,
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )


def _top_features(pipeline: Pipeline, n: int = 12) -> list[dict[str, Any]]:
    """Extract the highest positive/negative coefficient n-grams for reporting."""
    vectorizer: FeatureUnion = pipeline.named_steps["features"]
    classifier: LogisticRegression = pipeline.named_steps["clf"]
    names: list[str] = []
    for _, transformer in vectorizer.transformer_list:
        names.extend(list(transformer.get_feature_names_out()))
    weights = classifier.coef_[0]
    order = np.argsort(weights)
    top_spam = [{"feature": names[i], "weight": round(float(weights[i]), 4)}
                for i in order[-n:][::-1]]
    top_ham = [{"feature": names[i], "weight": round(float(weights[i]), 4)}
               for i in order[:n]]
    return top_spam + top_ham


def train_and_evaluate(
    frame: pd.DataFrame | None = None,
    settings: Settings | None = None,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train the pipeline and return ``(fitted_pipeline, metrics)``.

    Args:
        frame: Labeled dataset; loaded from disk when omitted.
        settings: Runtime settings; defaults to :func:`app.config.get_settings`.

    Returns:
        The fitted pipeline plus JSON serialisable metrics (accuracy, F1,
        ROC-AUC, PR-AUC, confusion matrix, top features, hardest test rows).
    """
    settings = settings or get_settings()
    if frame is None:
        frame = load_dataset()

    texts = frame["text"].astype(str).tolist()
    labels = LabelEncoder().fit_transform(frame["label"].tolist())  # ham=0, spam=1
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=labels,
    )
    logger.info("Training on %d messages, evaluating on %d", len(X_train), len(X_test))

    search = GridSearchCV(
        estimator=build_pipeline(random_state=settings.random_state),
        param_grid={"clf__C": list(settings.c_grid)},
        scoring="f1",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.random_state),
        n_jobs=settings.n_jobs,
        refit=True,
        verbose=0,
    )
    started = time.perf_counter()
    search.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    pipeline: Pipeline = search.best_estimator_

    proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (proba >= settings.spam_threshold).astype(int)

    metrics: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": MODEL_VERSION,
        "algorithm": "Tfidf(word 1-2 + char 3-5) -> LogisticRegression(balanced)",
        "best_C": float(search.best_params_["clf__C"]),
        "cv_f1_mean": round(float(search.best_score_), 4),
        "cv_f1_std": round(float(search.cv_results_["std_test_score"][search.best_index_]), 4),
        "test_size": settings.test_size,
        "random_state": settings.random_state,
        "fit_seconds": round(fit_seconds, 2),
        "samples_total": len(frame),
        "samples_train": len(X_train),
        "samples_test": len(X_test),
        "dataset_summary": summarise(frame),
        "threshold": settings.spam_threshold,
        "test_metrics": {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        },
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        # Not JSON: raw arrays kept for the report figures, stripped by
        # app.reporting.write_metrics() before anything is written to disk.
        "_raw": {"y_test": y_test.tolist(), "y_pred": y_pred.tolist(),
                 "y_score": proba.tolist()},
        "top_features": _top_features(pipeline),
        "hardest_test_rows": [
            {"text": t[:120], "actual": int(a), "pred": int(p), "p_spam": round(float(s), 4)}
            for t, a, p, s in sorted(
                zip(X_test, y_test, y_pred, proba),
                key=lambda row: abs(row[3] - 0.5),
                reverse=True,
            )[:15]
        ],
    }
    logger.info(
        "Trained in %.1fs | F1=%.3f accuracy=%.3f roc_auc=%.3f",
        fit_seconds,
        metrics["test_metrics"]["f1"],
        metrics["test_metrics"]["accuracy"],
        metrics["test_metrics"]["roc_auc"],
    )
    return pipeline, metrics


class SpamDetector:
    """Loads the trained pipeline and turns a message into a :class:`Prediction`."""

    def __init__(self, pipeline: Pipeline | None = None,
                 settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.pipeline = pipeline
        self.trained_at: str | None = None
        self.metrics: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, settings: Settings | None = None, auto_train: bool = True) -> "SpamDetector":
        """Load the model artifact from disk.

        Args:
            settings: Runtime settings (defaults to the process settings).
            auto_train: Train and persist a fresh model when none exists yet.

        Raises:
            FileNotFoundError: ``auto_train`` is False and no model artifact exists.
        """
        settings = settings or get_settings()
        path = settings.model_path
        if path.exists():
            artifact = joblib.load(path)
            detector = cls(pipeline=artifact["pipeline"], settings=settings)
            detector.trained_at = artifact.get("trained_at")
            detector.metrics = artifact.get("metrics", {})
            logger.debug("Loaded model artifact %s", path)
            return detector
        if not auto_train:
            raise FileNotFoundError(
                f"Model artifact not found at {path}. Train it with: python -m app.cli train"
            )
        logger.info("No model at %s -> training one now (first run only) ...", path)
        return cls.train(settings=settings)

    def save(self, path: Path | None = None) -> Path:
        """Persist the pipeline, its metrics and a version stamp as one artifact."""
        if self.pipeline is None:
            raise RuntimeError("Nothing to save: pipeline has not been trained.")
        target = Path(path or self.settings.model_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "pipeline": self.pipeline,
            "version": MODEL_VERSION,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metrics": self.metrics,
            "settings": {"threshold": self.settings.spam_threshold,
                         "random_state": self.settings.random_state},
        }
        joblib.dump(artifact, target, compress=3)
        logger.info("Saved model artifact -> %s (%.1f KiB)",
                    target, target.stat().st_size / 1024)
        return target

    @classmethod
    def train(cls, settings: Settings | None = None, save: bool = True) -> "SpamDetector":
        """Train from the dataset, optionally persist, and return the detector."""
        pipeline, metrics = train_and_evaluate(settings=settings)
        detector = cls(pipeline=pipeline, settings=settings)
        detector.metrics = metrics
        detector.trained_at = metrics["generated_at_utc"]
        if save:
            detector.save()
        return detector

    # ------------------------------------------------------------------ #
    # Inference                                                          #
    # ------------------------------------------------------------------ #
    def predict(self, text: str) -> Prediction:
        """Classify one message and attach transparent, human readable evidence.

        Args:
            text: Raw message (SMS body, e-mail body/snippet, chat text...).

        Returns:
            A :class:`Prediction` with verdict, blended scores, reasons and advice.
        """
        started = time.perf_counter()
        if self.pipeline is None:
            raise RuntimeError("Detector has no pipeline. Call SpamDetector.load() first.")

        clean = normalise_text(text)
        if not clean:
            return Prediction(
                text=text or "", verdict=HAM, risk_score=0.0, spam_probability=0.0,
                signal_score=0.0, confidence="low",
                reasons=["Empty input - nothing to classify."],
                advice="Paste an SMS, e-mail or chat message to check it.",
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )

        spam_probability = float(self.pipeline.predict_proba([text])[0][1])
        report: SignalReport = analyse(text)
        risk = ML_WEIGHT * spam_probability + (1 - ML_WEIGHT) * report.score

        if risk >= self.settings.spam_threshold:
            verdict = SPAM
            advice = ("Do not click or reply. Delete the message, and if you already shared "
                      "details, contact your bank/security team immediately.")
        elif risk >= self.settings.review_threshold:
            verdict = "review"
            advice = ("Uncertain. Verify the sender through an official channel (never the "
                      "link inside the message) before taking any action.")
        else:
            verdict = HAM
            advice = ("Looks like ordinary correspondence. Normal caution still applies - "
                      "legitimate organisations never ask for OTPs by text.")

        margin = min(risk, 1 - risk)
        confidence = "high" if margin <= 0.10 else ("medium" if margin <= 0.30 else "low")

        reasons = [f"{signal.label} - {signal.detail}" for signal in report.top]
        if verdict == HAM and not reasons:
            reasons = ["No spam indicators: no links, no urgency/credential/prize language, "
                       "normal capitalisation and length."]

        return Prediction(
            text=text,
            verdict=verdict,
            risk_score=round(risk, 4),
            spam_probability=round(spam_probability, 4),
            signal_score=report.score,
            confidence=confidence,
            reasons=reasons[:8],
            signals=[signal.as_dict() for signal in report.top],
            urls=report.urls,
            riskiest_url=report.riskiest_url,
            advice=advice,
            model_version=MODEL_VERSION,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def predict_many(self, texts: list[str]) -> list[Prediction]:
        """Classify a list of messages, preserving order."""
        return [self.predict(text) for text in texts]




