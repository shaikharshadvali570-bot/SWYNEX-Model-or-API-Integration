"""Environment driven configuration for SpamGuard AI.

All tunables have safe defaults so the prototype runs with zero configuration.
Secrets (optional LLM API keys) are read exclusively from the environment /
a local ``.env`` file and are **never** written into source control.

Copy ``.env.example`` to ``.env`` to change anything, for example::

    SPAMGUARD_SPAM_THRESHOLD=0.65
    AI_PROVIDER=openai
    OPENAI_API_KEY=put-your-own-key-here
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

try:  # python-dotenv is a runtime dependency, but keep import failure friendly.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - only when deps are not installed
    def load_dotenv(*_args, **_kwargs):  # type: ignore[misc]
        """No-op fallback used when ``python-dotenv`` is unavailable."""
        return False


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DATA = DATA_DIR / "processed" / "sms_spam.csv"
SAMPLE_DATA = DATA_DIR / "sample_messages.csv"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ENV_FILE = ROOT_DIR / ".env"

VALID_PROVIDERS = ("none", "openai", "gemini")


def _env_str(name: str, default: str = "") -> str:
    """Read a string environment variable, trimming whitespace."""
    return os.getenv(name, default).strip()


def _env_float(name: str, default: float, low: float, high: float) -> float:
    """Read a float environment variable, clamped to ``[low, high]``."""
    raw = _env_str(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        _get_logger().warning("Ignoring %s=%r (not a number); using %s", name, raw, default)
        return default
    return min(max(value, low), high)


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to ``default``."""
    raw = _env_str(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        _get_logger().warning("Ignoring %s=%r (not an integer); using %s", name, raw, default)
        return default


def _get_logger() -> logging.Logger:
    """Return the package logger (kept local to avoid circular imports)."""
    return logging.getLogger("spamguard")


def mask_secret(value: str | None, visible: int = 4) -> str:
    """Return a redacted representation of a secret, safe to log or display.

    >>> mask_secret("sk-1234567890abcdef")
    'sk-1...cdef (18 chars)'
    >>> mask_secret("")
    '(not set)'
    """
    if not value:
        return "(not set)"
    if len(value) <= visible * 2:
        return "*" * len(value)
    return f"{value[:visible]}...{value[-visible:]} ({len(value)} chars)"


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the prototype (immutable, built from the environment)."""

    model_path: Path = MODELS_DIR / "sms_spam_pipeline.joblib"
    metrics_path: Path = REPORTS_DIR / "metrics.json"
    dataset_path: Path = PROCESSED_DATA
    random_state: int = 42
    test_size: float = 0.2
    spam_threshold: float = 0.50
    review_threshold: float = 0.30
    c_grid: tuple[float, ...] = (0.3, 1.0, 3.0)
    n_jobs: int = 1
    log_level: str = "INFO"
    ai_provider: str = "none"
    openai_api_key: str = field(default="", repr=False)
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = ""
    gemini_api_key: str = field(default="", repr=False)
    gemini_model: str = "gemini-2.5-flash"
    ai_timeout: int = 25
    dotenv_loaded: bool = False

    @property
    def ai_configured(self) -> bool:
        """True when an LLM provider was selected *and* its key is present."""
        if self.ai_provider == "openai":
            return bool(self.openai_api_key) and not self.openai_api_key.startswith("put-")
        if self.ai_provider == "gemini":
            return bool(self.gemini_api_key) and not self.gemini_api_key.startswith("put-")
        return False

    @property
    def ai_key(self) -> str:
        """The active provider key (never log directly - use ``mask_secret``)."""
        return self.openai_api_key if self.ai_provider == "openai" else self.gemini_api_key

    def safe_public_view(self) -> dict[str, str]:
        """Diagnostics safe to show in the UI: secrets are always redacted."""
        return {
            "ai_provider": self.ai_provider,
            "ai_model": self.openai_model if self.ai_provider == "openai"
            else (self.gemini_model if self.ai_provider == "gemini" else "-"),
            "ai_status": "enabled" if self.ai_configured else "disabled (offline heuristics)",
            "ai_key": mask_secret(self.ai_key if self.ai_configured else ""),
            "model_file": self.model_path.name,
            "spam_threshold": f"{self.spam_threshold:.2f}",
            "review_threshold": f"{self.review_threshold:.2f}",
            "dotenv_loaded": str(self.dotenv_loaded),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process from ``.env`` plus real environment vars."""
    loaded = bool(load_dotenv(ENV_FILE, override=False))
    provider = _env_str("AI_PROVIDER", "none").lower()
    if provider not in VALID_PROVIDERS:
        _get_logger().warning(
            "Unknown AI_PROVIDER=%r (expected one of %s); disabling AI explanations",
            provider, ", ".join(VALID_PROVIDERS),
        )
        provider = "none"

    model_override = _env_str("SPAMGUARD_MODEL_PATH")
    settings = Settings(
        model_path=Path(model_override) if model_override
        else MODELS_DIR / "sms_spam_pipeline.joblib",
        dataset_path=Path(_env_str("SPAMGUARD_DATASET_PATH") or PROCESSED_DATA),
        random_state=_env_int("SPAMGUARD_RANDOM_STATE", 42),
        test_size=_env_float("SPAMGUARD_TEST_SIZE", 0.2, 0.05, 0.5),
        spam_threshold=_env_float("SPAMGUARD_SPAM_THRESHOLD", 0.50, 0.05, 0.95),
        review_threshold=_env_float("SPAMGUARD_REVIEW_THRESHOLD", 0.30, 0.01, 0.9),
        log_level=_env_str("SPAMGUARD_LOG_LEVEL", "INFO").upper(),
        ai_provider=provider,
        openai_api_key=_env_str("OPENAI_API_KEY"),
        openai_model=_env_str("OPENAI_MODEL", "gpt-4o-mini"),
        openai_base_url=_env_str("OPENAI_BASE_URL"),
        gemini_api_key=_env_str("GEMINI_API_KEY"),
        gemini_model=_env_str("GEMINI_MODEL", "gemini-2.5-flash"),
        ai_timeout=_env_int("AI_TIMEOUT_SECONDS", 25),
        dotenv_loaded=loaded,
    )
    if settings.review_threshold >= settings.spam_threshold:
        # Keep the "needs review" band meaningful: always below the spam cut-off.
        settings = replace(settings, review_threshold=round(settings.spam_threshold * 0.6, 3))
    return settings


def setup_logging(level: str | None = None) -> logging.Logger:
    """Configure the ``spamguard`` logger once and return it."""
    logger = logging.getLogger("spamguard")
    if logger.handlers:  # already configured (e.g. on Streamlit reruns)
        return logger
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, (level or get_settings().log_level), logging.INFO))
    logger.propagate = False
    return logger

