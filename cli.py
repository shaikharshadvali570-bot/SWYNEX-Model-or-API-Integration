"""Command line interface for SpamGuard AI.

Examples
--------
    python -m app.cli train                          # train + reports/figures
    python -m app.cli check                          # environment / data sanity
    python -m app.cli predict "URGENT: your bank ..."
    python -m app.cli predict --file message.txt --explain
    python -m app.cli batch --input data/sample_messages.csv --output out.csv
    python -m app.cli demo                           # the documented examples

Add ``--json`` anywhere for machine-readable output (nice for CI/e2e tests).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from app import __version__
from app.config import ROOT_DIR, get_settings, mask_secret, setup_logging
from app.dataset import load_dataset, load_samples, summarise
from app.explainer import ai_status, explain
from app.model import SpamDetector, train_and_evaluate
from app.reporting import make_figures, write_metrics

logger = setup_logging()

ANSI = {"spam": "\033[91m", "review": "\033[93m", "ham": "\033[92m",
        "dim": "\033[2m", "bold": "\033[1m", "reset": "\033[0m"}


def paint(text: str, style: str) -> str:
    """Wrap ``text`` in an ANSI colour when the terminal supports it."""
    if not sys.stdout.isatty():
        return text
    return f"{ANSI[style]}{text}{ANSI['reset']}"


def render(prediction: Any, explain_it: bool = False, settings=None) -> str:
    """Render one prediction as a human readable block of text."""
    icon = {"spam": "[SPAM]", "review": "[REVIEW]", "ham": "[HAM]"}[prediction.verdict]
    lines = [
        paint(f"{icon} verdict: {prediction.verdict.upper()}", prediction.verdict),
        f"  risk score      : {prediction.risk_score:.3f}",
        f"  model P(spam)   : {prediction.spam_probability:.3f}",
        f"  rule signals    : {prediction.signal_score:.3f}",
        f"  confidence      : {prediction.confidence}",
        f"  latency         : {prediction.latency_ms:.1f} ms",
    ]
    if prediction.urls:
        lines.append(f"  links found     : {', '.join(prediction.urls)}")
        if prediction.riskiest_url:
            lines.append(f"  riskiest link   : {prediction.riskiest_url}")
    if prediction.reasons:
        lines.append("  why:")
        lines.extend(f"    - {reason}" for reason in prediction.reasons)
    lines.append(f"  advice          : {prediction.advice}")
    if explain_it:
        text, source = explain(prediction, settings)
        lines.append(f"  ai explanation ({source}):")
        lines.append(f"    {text}")
    return "\n".join(lines)


def cmd_check(_args: argparse.Namespace) -> int:
    """Print environment, dataset and model diagnostics (never any secret)."""
    settings = get_settings()
    summary: dict[str, Any] = {
        "app": f"SpamGuard AI v{__version__}",
        "root": str(ROOT_DIR),
        "python": sys.version.split()[0],
        "settings": settings.safe_public_view(),
        "ai_status": ai_status(settings),
    }
    try:
        summary["dataset"] = summarise(load_dataset())
    except (FileNotFoundError, ValueError) as exc:
        summary["dataset"] = f"NOT READY -> {exc}"

    if settings.model_path.exists():
        detector = SpamDetector.load(settings=settings, auto_train=False)
        test_metrics = detector.metrics.get("test_metrics", {})
        summary["model"] = {"artifact": str(settings.model_path),
                            "trained_at": detector.trained_at,
                            **test_metrics}
        summary["smoke_test"] = detector.predict(
            "FINAL WARNING: your bank account closes today. Verify now at http://192.168.0.1/login"
        ).as_dict()
    else:
        summary["model"] = "not trained yet -> run: python -m app.cli train"

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Retrain the model, then write reports/metrics.json and the figures."""
    settings = get_settings()
    detector = SpamDetector.train(settings=settings, save=True)
    metrics = detector.metrics
    write_metrics(metrics, settings.metrics_path)
    figures = make_figures(metrics)

    if args.json:
        printable = {k: v for k, v in metrics.items() if k != "_raw"}
        print(json.dumps(printable, indent=2, ensure_ascii=False))
        return 0

    test = metrics["test_metrics"]
    data = metrics["dataset_summary"]
    print(paint("Training complete", "bold"))
    print(f"  dataset          : {data['total']:,} msgs "
          f"({data['ham']:,} ham / {data['spam']:,} spam = {data['spam_pct']}% spam)")
    print(f"  split            : {metrics['samples_train']:,} train / "
          f"{metrics['samples_test']:,} test (test_size={metrics['test_size']})")
    print(f"  algorithm        : {metrics['algorithm']} (C={metrics['best_C']})")
    print(f"  CV F1 (3-fold)   : {metrics['cv_f1_mean']:.3f} (+/- {metrics['cv_f1_std']:.3f})")
    print(f"  test accuracy    : {test['accuracy']:.3f}")
    print(f"  test precision   : {test['precision']:.3f}")
    print(f"  test recall      : {test['recall']:.3f}")
    print(f"  test F1          : {test['f1']:.3f}")
    print(f"  ROC-AUC / PR-AUC : {test['roc_auc']:.3f} / {test['pr_auc']:.3f}")
    print(f"  confusion matrix : {metrics['confusion_matrix']}")
    print(f"  trained in       : {metrics['fit_seconds']} s")
    print(f"  model artifact   : {settings.model_path}")
    print(f"  metrics          : {settings.metrics_path}")
    for figure in figures:
        print(f"  figure           : {figure}")
    return 0


def _detector(auto_train: bool = True) -> SpamDetector:
    """Shared loader used by the predict/batch/demo commands."""
    return SpamDetector.load(auto_train=auto_train)


def cmd_predict(args: argparse.Namespace) -> int:
    """Classify one or more messages given as arguments, ``--file`` or stdin."""
    settings = get_settings()
    messages: list[str] = list(args.text)

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            return 1
        messages.append(path.read_text(encoding="utf-8"))
    if not messages and not sys.stdin.isatty():
        messages.append(sys.stdin.read())
    if not messages:
        print("ERROR: no input. Pass a message, --file or pipe text on stdin.",
              file=sys.stderr)
        return 1

    detector = _detector(auto_train=not args.no_auto_train)
    predictions = [detector.predict(message) for message in messages]

    if args.json:
        if len(predictions) == 1:
            print(predictions[0].to_json())
        else:
            print(json.dumps([json.loads(p.to_json()) for p in predictions],
                             indent=2, ensure_ascii=False))
        return 0

    for index, prediction in enumerate(predictions):
        if index:
            print()
        print(render(prediction, explain_it=args.explain, settings=settings))
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    """Score every row of a CSV and write an annotated copy."""
    settings = get_settings()
    source = Path(args.input)
    if not source.exists():
        print(f"ERROR: input CSV not found: {source}", file=sys.stderr)
        return 1

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or args.text_col not in rows[0]:
        print(f"ERROR: column '{args.text_col}' not found in {source}", file=sys.stderr)
        return 1

    detector = _detector(auto_train=not args.no_auto_train)
    for row in rows:
        prediction = detector.predict(str(row[args.text_col]))
        row.update({
            "verdict": prediction.verdict,
            "risk_score": prediction.risk_score,
            "spam_probability": prediction.spam_probability,
            "signal_score": prediction.signal_score,
            "confidence": prediction.confidence,
            "reasons": "; ".join(prediction.reasons),
        })

    target = Path(args.output or source.with_name(f"{source.stem}_scored.csv"))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts = {"spam": 0, "review": 0, "ham": 0}
    for row in rows:
        counts[row["verdict"]] += 1
    print(f"Scored {len(rows)} message(s) -> {target}")
    print(f"  spam={counts['spam']}  review={counts['review']}  ham={counts['ham']}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Score the curated examples from ``data/sample_messages.csv``."""
    settings = get_settings()
    samples = load_samples()
    if samples.empty:
        print("ERROR: data/sample_messages.csv not found.", file=sys.stderr)
        return 1

    detector = _detector(auto_train=not args.no_auto_train)
    if args.json:
        results = []
        for _, row in samples.iterrows():
            prediction = detector.predict(str(row["text"]))
            results.append({"input": row["text"],
                            "expected": row.get("expected", ""),
                            "prediction": json.loads(prediction.to_json())})
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return 0

    print(paint("SpamGuard AI - demo run (documented examples)", "bold"))
    print(ai_status(settings))
    for index, row in samples.iterrows():
        prediction = detector.predict(str(row["text"]))
        print(f"\n--- example {index + 1} | expected={row.get('expected', '?')} ---")
        print(f"    input : {row['text']}")
        print(render(prediction, explain_it=args.explain, settings=settings))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser (one subcommand per capability)."""
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="SpamGuard AI - explainable SMS/e-mail spam & phishing detection.",
    )
    parser.add_argument("--version", action="version", version=f"SpamGuard AI {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check", help="print environment / data / model diagnostics")

    train = subparsers.add_parser("train", help="train the model and write reports/")
    train.add_argument("--json", action="store_true", help="dump raw metrics as JSON")

    predict = subparsers.add_parser("predict", help="classify a message")
    predict.add_argument("text", nargs="*", help="message(s) to classify")
    predict.add_argument("--file", help="read the message from a text file")
    predict.add_argument("--json", action="store_true", help="machine-readable output")
    predict.add_argument("--explain", action="store_true",
                         help="add an AI explanation (LLM if configured, else offline)")
    predict.add_argument("--no-auto-train", action="store_true",
                         help="fail instead of training when no model exists")

    batch = subparsers.add_parser("batch", help="score a CSV file")
    batch.add_argument("--input", required=True, help="input CSV path")
    batch.add_argument("--output", help="output CSV path (default: <input>_scored.csv)")
    batch.add_argument("--text-col", default="text", help="column containing the message")
    batch.add_argument("--no-auto-train", action="store_true")

    demo = subparsers.add_parser("demo", help="run the documented example inputs")
    demo.add_argument("--json", action="store_true")
    demo.add_argument("--explain", action="store_true")
    demo.add_argument("--no-auto-train", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point shared by ``python -m app.cli`` and ``python app/cli.py``."""
    args = build_parser().parse_args(argv)
    handlers = {"check": cmd_check, "train": cmd_train, "predict": cmd_predict,
                "batch": cmd_batch, "demo": cmd_demo}
    try:
        return handlers[args.command](args)
    except FileNotFoundError as exc:  # missing dataset/model -> actionable hint
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


