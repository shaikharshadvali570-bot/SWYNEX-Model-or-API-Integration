"""Persist evaluation artefacts: ``reports/metrics.json`` and PNG figures.

Figures are generated with the non-interactive ``Agg`` backend so this module
also works on headless CI machines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # must precede any pyplot import
import matplotlib.pyplot as plt  # noqa: E402

from app.config import FIGURES_DIR, REPORTS_DIR, setup_logging

logger = setup_logging()

PALETTE = {"spam": "#e74c3c", "ham": "#27ae60", "accent": "#2c3e50"}


def serialisable(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``metrics`` without the non-JSON ``_raw`` arrays."""
    clean = {k: v for k, v in metrics.items() if k != "_raw"}
    return json.loads(json.dumps(clean))  # normalise numpy -> python types


def write_metrics(metrics: dict[str, Any], path: Path | None = None) -> Path:
    """Write the metrics dictionary to ``reports/metrics.json``."""
    target = Path(path or REPORTS_DIR / "metrics.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(serialisable(metrics), indent=2) + "\n", encoding="utf-8")
    logger.info("Metrics written -> %s", target)
    return target


def make_figures(metrics: dict[str, Any], figures_dir: Path | None = None) -> list[Path]:
    """Render the confusion matrix, top features and score distribution figures.

    Args:
        metrics: Result of :func:`app.model.train_and_evaluate`.
        figures_dir: Output directory (defaults to ``reports/figures``).

    Returns:
        Paths of the PNG files that were written.
    """
    out = Path(figures_dir or FIGURES_DIR)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # 1) Confusion matrix ---------------------------------------------------
    matrix = metrics.get("confusion_matrix")
    if matrix:
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(max(r) for r in matrix))
        for row in range(len(matrix)):
            for col in range(len(matrix[0])):
                ax.text(col, row, f"{matrix[row][col]}",
                        ha="center", va="center",
                        color="white" if matrix[row][col] > matrix[0][0] * 0.6 else PALETTE["accent"],
                        fontsize=16, fontweight="bold")
        ax.set_xticks([0, 1], ["Predicted ham", "Predicted spam"])
        ax.set_yticks([0, 1], ["Actual ham", "Actual spam"], rotation=90, va="center")
        acc = metrics["test_metrics"]["accuracy"]
        ax.set_title(f"Confusion matrix (hold-out set, accuracy={acc:.3f})")
        fig.colorbar(image, fraction=0.046)
        fig.tight_layout()
        path = out / "confusion_matrix.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)

    # 2) Top predictive features -------------------------------------------
    features = metrics.get("top_features") or []
    if features:
        half = len(features) // 2
        spam_terms, ham_terms = features[:half], features[half:]
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
        for ax, rows, title, colour in (
            (axes[0], spam_terms, "Top SPAM indicators", PALETTE["spam"]),
            (axes[1], ham_terms, "Top HAM indicators", PALETTE["ham"]),
        ):
            rows = sorted(rows, key=lambda r: abs(r["weight"]))
            ax.barh([r["feature"] for r in rows], [r["weight"] for r in rows], color=colour)
            ax.set_title(title)
            ax.set_xlabel("log-odds weight")
            ax.tick_params(labelsize=9)
        fig.suptitle("What the model learned (TF-IDF coefficients)")
        fig.tight_layout()
        path = out / "top_features.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)

    # 3) Risk score distribution on the hold-out set -------------------------
    raw = metrics.get("_raw") or {}
    scores = raw.get("y_score")
    if scores:
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        ax.hist(
            [s for s, y in zip(scores, raw["y_test"]) if y == 0],
            bins=30, alpha=0.75, label="ham", color=PALETTE["ham"],
        )
        ax.hist(
            [s for s, y in zip(scores, raw["y_test"]) if y == 1],
            bins=30, alpha=0.75, label="spam", color=PALETTE["spam"],
        )
        threshold = metrics.get("threshold", 0.5)
        ax.axvline(threshold, color=PALETTE["accent"], linestyle="--",
                   label=f"threshold {threshold:.2f}")
        ax.set_xlabel("Predicted P(spam)")
        ax.set_ylabel("Messages")
        ax.set_title("Model confidence on the hold-out set")
        ax.legend()
        fig.tight_layout()
        path = out / "score_distribution.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)

    if written:
        logger.info("Figures written: %s", ", ".join(p.name for p in written))
    return written
