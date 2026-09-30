"""CLI contract tests (exit codes, JSON output, batch CSV)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.cli import main
from conftest import NORMAL_MSG, PHISHING


def test_predict_json_outputs_parseable_result(capsys):
    assert main(["predict", PHISHING, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "spam"
    assert 0.0 <= payload["risk_score"] <= 1.0


def test_predict_two_messages(capsys):
    assert main(["predict", PHISHING, NORMAL_MSG, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list) and len(payload) == 2


def test_check_reports_configuration(capsys):
    assert main(["check"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["settings"]["ai_provider"] in {"none", "openai", "gemini"}
    assert "sk-" not in json.dumps(payload), "no raw secret may be printed"


def test_demo_runs_all_samples(capsys):
    assert main(["demo", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload, "sample_messages.csv must not be empty"
    assert all(row["prediction"]["verdict"] in {"spam", "review", "ham"} for row in payload)


def test_batch_writes_scored_csv(tmp_path: Path, capsys):
    source = tmp_path / "in.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text"])
        writer.writerow([PHISHING])
        writer.writerow([NORMAL_MSG])
    output = tmp_path / "out.csv"

    assert main(["batch", "--input", str(source), "--output", str(output)]) == 0
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["verdict"] == "spam"
    assert rows[1]["verdict"] == "ham"
    assert float(rows[0]["risk_score"]) >= float(rows[1]["risk_score"])


def test_predict_without_input_returns_error(capsys, monkeypatch):
    class _Tty:
        @staticmethod
        def isatty():
            return True

    monkeypatch.setattr("sys.stdin", _Tty())
    assert main(["predict"]) == 1
