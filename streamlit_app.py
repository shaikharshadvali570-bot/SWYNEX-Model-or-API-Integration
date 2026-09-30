"""Streamlit web user interface for SpamGuard AI.

Run it with::

    streamlit run app/streamlit_app.py     # or simply: python run.py web

The page is deliberately small: one text box, one button, one verdict panel with
explanations, plus an optional AI narrative and a metrics sidebar.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow "streamlit run app/streamlit_app.py" from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app import __version__
from app.config import get_settings
from app.dataset import load_samples
from app.explainer import ai_status, explain
from app.model import SpamDetector
from app.reporting import serialisable

st.set_page_config(page_title="SpamGuard AI", page_icon="🛡️", layout="wide")

VERDICT_META = {
    "spam": ("🚨 SPAM / PHISHING", "This message shows strong scam patterns."),
    "review": ("⚠️ NEEDS REVIEW", "Mixed signals - verify the sender before acting."),
    "ham": ("✅ LIKELY GENUINE", "No notable scam indicators were found."),
}

PROMPTS = {
    "1. Bank phishing (spam)": (
        "URGENT: We detected a suspicious login. Your account will be closed in 24 hrs. "
        "Verify your identity now: http://192.168.1.10/secure-login"
    ),
    "2. Prize scam (spam)": (
        "Congratulations! You won a 500 shopping voucher. Claim your FREE gift now, "
        "reply with your card details. STOP to unsubscribe"
    ),
    "3. Meeting change (ham)": (
        "Hi, can we move standup to 10:30 tomorrow? I'll send a new calendar invite. "
        "Thanks!"
    ),
    "4. Delivery notice (spam)": (
        "Your parcel could not be delivered. Pay the 1.99 fee within 12 hours at "
        "bit.ly/dlv-redir to reschedule."
    ),
    "5. Friend reply (ham)": (
        "lol that film was terrible, see you at the pub at 8 🍕"
    ),
}


@st.cache_resource(show_spinner="Loading / training the model ...")
def load_detector() -> SpamDetector:
    """Load the trained pipeline once per app session (auto-trains on first run)."""
    return SpamDetector.load(auto_train=True)


def show_metrics(detector: SpamDetector) -> None:
    """Render the model report in the sidebar."""
    metrics = detector.metrics or {}
    if not metrics:
        st.sidebar.caption("No training metrics stored in this artifact yet.")
        return
    test = metrics.get("test_metrics", {})
    st.sidebar.subheader("Model report")
    st.sidebar.metric("Accuracy", f"{test.get('accuracy', 0):.3f}")
    st.sidebar.metric("F1", f"{test.get('f1', 0):.3f}")
    st.sidebar.metric("ROC-AUC", f"{test.get('roc_auc', 0):.3f}")
    st.sidebar.metric("Precision / Recall",
                      f"{test.get('precision', 0):.2f} / {test.get('recall', 0):.2f}")
    st.sidebar.caption(
        f"held-out test set ({metrics.get('samples_test', 0)} messages), "
        f"C={metrics.get('best_C')}, CV F1 {metrics.get('cv_f1_mean', 0):.3f}"
    )
    with st.sidebar.expander("Full metrics (JSON)"):
        st.json(serialisable(metrics))


def main() -> None:
    """Build and run the single-page app."""
    settings = get_settings()
    detector = load_detector()

    st.title("🛡️ SpamGuard AI")
    st.caption("Explainable SMS / e-mail spam & phishing detector - scikit-learn "
               f"TF-IDF + logistic regression blended with transparent rules (v{__version__})")

    st.sidebar.header("Status")
    st.sidebar.write(ai_status(settings))
    st.sidebar.caption(f"thresholds: spam >= {settings.spam_threshold:.2f}, "
                       f"review >= {settings.review_threshold:.2f}")
    if detector.trained_at:
        st.sidebar.caption(f"model trained at {detector.trained_at}")
    show_metrics(detector)

    message = st.text_area(
        "Paste the suspicious message",
        height=170,
        placeholder="e.g. URGENT: your bank account will be suspended...",
    )

    sample_label = st.selectbox("...or start from an example", ["(type your own)", *PROMPTS])
    if sample_label != "(type your own)":
        message = PROMPTS[sample_label]

    col_run, col_hint = st.columns([1, 3])
    analyse = col_run.button("Analyse message", type="primary", use_container_width=True)
    col_hint.caption("Runs fully offline - your message never leaves this machine.")

    if analyse and message.strip():
        with st.spinner("Scoring message ..."):
            prediction = detector.predict(message)
            narration, source = explain(prediction, settings)

        label, blurb = VERDICT_META[prediction.verdict]
        st.subheader(f"{prediction.emoji} {label}")
        st.caption(blurb)

        left, right = st.columns([3, 2])
        with left:
            st.progress(prediction.risk_score, text=f"Risk score {prediction.risk_score:.2f}")
            st.markdown(
                "| Model P(spam) | Rule signals | Confidence | Latency |\n"
                "|---:|---:|---|---:|\n"
                f"| **{prediction.spam_probability:.3f}** | {prediction.signal_score:.3f} "
                f"| {prediction.confidence} | {prediction.latency_ms:.1f} ms |"
            )
        with right:
            if prediction.urls:
                st.markdown("**Links found**")
                for url in prediction.urls[:5]:
                    st.code(url, language=None)
                if prediction.riskiest_url:
                    st.error(f"Riskiest link: {prediction.riskiest_url}")
            else:
                st.success("No links detected in this message")

        st.markdown("#### Why?")
        if prediction.reasons:
            for reason in prediction.reasons:
                st.markdown(f"- {reason}")
        else:
            st.markdown("- No spam indicators detected.")

        st.info(f"**Recommended action:** {prediction.advice}")

        with st.expander("AI explanation"):
            st.markdown(f"*Source: {source}*")
            st.write(narration)

        with st.expander("Machine-readable result"):
            st.json(json.loads(prediction.to_json()))
        return

    if analyse:
        st.warning("Type a message first (or pick an example).")
        return

    st.markdown(
        """
##### Try these
| Input | Expected |
|---|---|
| `URGENT: We detected a suspicious login... verify now` | SPAM |
| `Congratulations! You won a 500 voucher... reply STOP` | SPAM |
| `Hi, can we move standup to 10:30 tomorrow?` | HAM |
"""
    )
    samples = load_samples()
    if not samples.empty:
        with st.expander(f"All {len(samples)} sample inputs (data/sample_messages.csv)"):
            st.dataframe(samples, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

