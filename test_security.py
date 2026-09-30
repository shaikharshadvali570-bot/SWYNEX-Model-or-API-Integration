"""Guard rails: no secrets in source, `.env` never committed, safe masking."""

from __future__ import annotations

import re
from pathlib import Path

from app.config import get_settings, mask_secret

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("app", "scripts", "tests")
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),            # OpenAI style keys
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),        # Google API keys
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),           # GitHub tokens
]


def _source_files():
    for name in SOURCE_DIRS:
        yield from (ROOT / name).rglob("*.py")


def test_no_hardcoded_api_keys_in_source():
    for path in list(_source_files()) + [ROOT / ".env.example"]:
        text = path.read_text(encoding="utf-8")
        for pattern in SECRET_PATTERNS:
            assert not pattern.search(text), f"possible secret committed in {path}"


def test_env_file_is_gitignored():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in ignore and "!.env.example" in ignore
    assert not (ROOT / ".env").exists() or ".env" in ignore


def test_masking_never_reveals_key():
    key = "sk-proj-abcdefghijklmnop"
    masked = mask_secret(key)
    assert key not in masked
    assert mask_secret("") == "(not set)"
    assert mask_secret("short") == "*****"


def test_settings_public_view_is_safe():
    view = get_settings().safe_public_view()
    assert view["ai_key"] in {"(not set)"} or "..." in view["ai_key"]
    assert "ai_provider" in view and "spam_threshold" in view
