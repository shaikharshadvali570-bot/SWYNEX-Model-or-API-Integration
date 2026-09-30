"""Dataset loading helpers with friendly, actionable error messages."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.config import SAMPLE_DATA, get_settings

EXPECTED_COLUMNS = ("label", "label_code", "text")
LABEL_NAMES = {0: "ham", 1: "spam"}


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the training data is missing."""

    def __init__(self, path: Path) -> None:
        super().__init__(
            f"Dataset not found at {path}\n\n"
            "Fetch it with:  python scripts/download_data.py\n"
            "(or copy your own CSV with columns: label,label_code,text)"
        )
        self.path = path


def load_dataset(path: Path | None = None, *, require_labels: bool = True) -> pd.DataFrame:
    """Load and validate the message dataset.

    Args:
        path: CSV location; defaults to ``SPAMGUARD_DATASET_PATH`` or ``data/processed/sms_spam.csv``.
        require_labels: When False, only ``text`` (or the first column) is needed.

    Returns:
        DataFrame with at least a ``text`` column, plus labels when required.

    Raises:
        DatasetNotFoundError: The file does not exist.
        ValueError: Required columns are missing or labels are invalid.
    """
    target = Path(path or get_settings().dataset_path)
    if not target.exists():
        raise DatasetNotFoundError(target)

    frame = pd.read_csv(target)
    if require_labels:
        missing = [c for c in EXPECTED_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(
                f"{target} is missing required column(s): {', '.join(missing)}. "
                "Expected schema: label,label_code,text"
            )
        if set(frame["label"].unique()) - {"ham", "spam"}:
            raise ValueError(f"{target} contains labels other than 'ham'/'spam'.")
        if not set(frame["label_code"].unique()) <= {0, 1}:
            raise ValueError(f"{target}: label_code must contain only 0/1.")
    elif "text" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "text"})

    return frame


def load_samples(path: Path | None = None) -> pd.DataFrame:
    """Load the small curated sample set used by the demo modes.

    Falls back to an empty frame with the expected schema so callers never fail
    on a missing optional file.
    """
    target = Path(path or SAMPLE_DATA)
    if not target.exists():
        return pd.DataFrame(columns=["id", "text", "expected"])
    return pd.read_csv(target)


def summarise(frame: pd.DataFrame) -> dict[str, float | int | str]:
    """Return basic dataset statistics used in the README and UI header."""
    spam = int((frame["label"] == "spam").sum())
    total = len(frame)
    return {
        "total": total,
        "ham": total - spam,
        "spam": spam,
        "spam_pct": round(spam / total * 100, 2) if total else 0.0,
        "avg_chars": int(frame["text"].astype(str).str.len().mean()) if total else 0,
        "class_weight_ratio": round((total - spam) / spam, 2) if spam else 0.0,
    }
