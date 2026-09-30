"""SpamGuard AI - a small, explainable SMS / email spam and phishing detector.

The package is organised in small, independently testable modules:

* :mod:`app.config`        - environment driven settings and file paths.
* :mod:`app.preprocessing` - text normalisation shared by training and inference.
* :mod:`app.signals`       - transparent rule based risk signals (hybrid AI).
* :mod:`app.dataset`       - dataset loading helpers.
* :mod:`app.model`         - scikit-learn pipeline: train, evaluate, predict.
* :mod:`app.explainer`     - optional LLM narrative via a public AI API.
* :mod:`app.cli`           - command line interface.
* :mod:`app.streamlit_app` - web user interface.
"""

from __future__ import annotations

__version__ = "1.0.0"
__app_name__ = "SpamGuard AI"
__all__ = ["__version__", "__app_name__"]
