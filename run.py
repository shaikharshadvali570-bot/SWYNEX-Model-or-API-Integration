#!/usr/bin/env python3
"""Single entry point for SpamGuard AI.

    python run.py web     # launch the Streamlit UI (default)
    python run.py cli ... # any CLI command, e.g. python run.py cli train
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMANDS = {"web", "ui", "cli", "train", "check", "demo"}


def python_exe() -> str:
    """Return the interpreter to run (the active venv when present)."""
    venv = ROOT / (".venv" / "Scripts" if sys.platform == "win32" else ".venv" / "bin")
    candidate = venv / ("python.exe" if sys.platform == "win32" else "python")
    return str(candidate) if candidate.exists() else sys.executable


def main() -> int:
    """Dispatch to Streamlit or the CLI."""
    target = (sys.argv[1] if len(sys.argv) > 1 else "web").lower()
    rest = sys.argv[2:]

    if target in {"web", "ui"}:
        app = ROOT / "app" / "streamlit_app.py"
        return subprocess.call([python_exe(), "-m", "streamlit", "run", str(app), *rest])

    if target == "cli":
        return subprocess.call([python_exe(), "-m", "app.cli", *rest], cwd=str(ROOT))

    if target in {"train", "check", "demo"}:
        return subprocess.call([python_exe(), "-m", "app.cli", target, *rest], cwd=str(ROOT))

    print(__doc__)
    print(f"Unknown command {target!r}; expected one of: {', '.join(sorted(COMMANDS))}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
