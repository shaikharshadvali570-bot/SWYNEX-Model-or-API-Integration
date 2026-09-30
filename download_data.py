"""Fetch and normalise the public SMS Spam Collection dataset (UCI ML Repository).

The dataset used by SpamGuard AI is the well known "SMS Spam Collection"
(Almeida, Gomez & Hidalgo, 2011) published in the UCI Machine Learning
Repository under a CC BY 4.0 licence.

This script is intentionally dependency free (standard library only) so it can
be run before ``pip install -r requirements.txt`` if desired.

What it does
------------
1. Downloads ``sms_spam_collection.zip`` from the first reachable mirror.
2. Extracts the ``SMSSpamCollection`` flat file (``label<TAB>message``).
3. Normalises it into ``data/processed/sms_spam.csv`` with the columns
   ``label`` (``ham`` / ``spam``), ``label_code`` (0 / 1) and ``text``.
4. Writes ``data/data_card.json`` with provenance, size and checksums.

Usage
-----
    python scripts/download_data.py            # download only if needed
    python scripts/download_data.py --force    # re-download and rebuild
    python scripts/download_data.py --quiet
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARCHIVE_PATH = RAW_DIR / "sms_spam_collection.zip"
EXTRACT_DIR = RAW_DIR / "_extracted"
RAW_FILE_PATH = RAW_DIR / "SMSSpamCollection"
PROCESSED_CSV = PROCESSED_DIR / "sms_spam.csv"
DATA_CARD = DATA_DIR / "data_card.json"

DATASET_TITLE = "SMS Spam Collection"
DATASET_URL = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"

# Ordered by preference. The first reachable mirror wins.
MIRRORS: tuple[str, ...] = (
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
)

USER_AGENT = "SpamGuardAI/1.0 (research prototype; +https://github.com/)"
LABEL_MAP = {"ham": 0, "spam": 1}


class DatasetStats(NamedTuple):
    """Simple container for the row counts of the normalised dataset."""

    total: int
    ham: int
    spam: int
    skipped: int


def _log(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(message)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file, reading it in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(quiet: bool = False, force: bool = False) -> Path:
    """Download the dataset archive, trying each mirror in turn.

    Args:
        quiet: Suppress progress output.
        force: Re-download even when the archive already exists.

    Returns:
        Path to the downloaded zip archive.

    Raises:
        RuntimeError: If no mirror could be reached.
    """
    if ARCHIVE_PATH.exists() and not force:
        _log(f"[skip] archive already present: {ARCHIVE_PATH}", quiet)
        return ARCHIVE_PATH

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    for url in MIRRORS:
        try:
            _log(f"[get ] {url}", quiet)
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=90) as response, \
                    ARCHIVE_PATH.open("wb") as sink:
                shutil.copyfileobj(response, sink)
            _log(f"[ok  ] saved {ARCHIVE_PATH.stat().st_size / 1024:,.1f} KiB", quiet)
            return ARCHIVE_PATH
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            errors.append(f"{url}: {exc}")
            _log(f"[warn] mirror failed ({exc})", quiet)

    raise RuntimeError(
        "Could not download the SMS Spam Collection dataset from any mirror.\n"
        + "\n".join(errors)
        + "\n\nOffline workaround: download the archive manually from\n"
        f"  {DATASET_URL}\n"
        f"and store it as {ARCHIVE_PATH}, then re-run this script."
    )


def extract(quiet: bool = False) -> Path:
    """Extract ``SMSSpamCollection`` from the downloaded archive.

    Returns:
        Path to the extracted raw flat file.

    Raises:
        FileNotFoundError: If the archive is missing or lacks the data file.
    """
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(f"Missing archive: {ARCHIVE_PATH}")

    if RAW_FILE_PATH.exists():
        return RAW_FILE_PATH

    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        candidate = next(
            (
                info
                for info in archive.infolist()
                if not info.is_dir()
                and info.filename.rsplit("/", 1)[-1].lower() == "smsspamcollection"
            ),
            None,
        )
        if candidate is None:
            raise FileNotFoundError(
                "SMSSpamCollection not found inside "
                f"{ARCHIVE_PATH} (first entries: {archive.namelist()[:5]})"
            )
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        target = EXTRACT_DIR / "SMSSpamCollection"
        with archive.open(candidate) as source, target.open("wb") as sink:
            shutil.copyfileobj(source, sink)
        shutil.copyfile(target, RAW_FILE_PATH)

    shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
    _log(f"[ok  ] extracted -> {RAW_FILE_PATH}", quiet)
    return RAW_FILE_PATH


def normalise(quiet: bool = False) -> DatasetStats:
    """Convert the raw tab separated file into a clean UTF-8 CSV.

    The raw file uses ``label<TAB>message``. A few rows in the public release
    contain no tab or an empty message; those are counted as ``skipped`` so
    that training never sees malformed input.

    Returns:
        Row counts of the normalised dataset.
    """
    if not RAW_FILE_PATH.exists():
        extract(quiet=quiet)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    total = ham = spam = skipped = 0

    with RAW_FILE_PATH.open("r", encoding="latin-1", newline="") as source, \
            PROCESSED_CSV.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["label", "label_code", "text"])
        for raw_line in source:
            line = raw_line.strip("\ufeff").strip()
            if not line:
                skipped += 1
                continue
            if "\t" in line:
                label, _, text = line.partition("\t")
            else:  # rare fallback: whitespace separated label + message
                label, _, text = line.partition(" ")
            label, text = label.strip().lower(), " ".join(text.split())
            if label not in LABEL_MAP or not text:
                skipped += 1
                continue
            writer.writerow([label, LABEL_MAP[label], text])
            total += 1
            if label == "spam":
                spam += 1
            else:
                ham += 1

    return DatasetStats(total=total, ham=ham, spam=spam, skipped=skipped)


def write_data_card(stats: DatasetStats, quiet: bool = False) -> Path:
    """Persist provenance and checksum information next to the data."""
    card = {
        "title": DATASET_TITLE,
        "source": DATASET_URL,
        "licence": "CC BY 4.0 (UCI Machine Learning Repository)",
        "citation": (
            "Almeida, T., Gomez, J., & Hidalgo, J. (2011). Turn your spam filter "
            "on: Techniques to parse spamdaka. IDAAC 2011."
        ),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "archive": {
            "path": str(ARCHIVE_PATH.relative_to(REPO_ROOT)),
            "sha256": _sha256(ARCHIVE_PATH) if ARCHIVE_PATH.exists() else None,
            "mirrors": list(MIRRORS),
        },
        "processed": {
            "path": str(PROCESSED_CSV.relative_to(REPO_ROOT)),
            "sha256": _sha256(PROCESSED_CSV) if PROCESSED_CSV.exists() else None,
            "columns": ["label", "label_code", "text"],
        },
        "rows": stats._asdict(),
        "label_map": LABEL_MAP,
        "notes": "Messages kept verbatim except for whitespace normalisation.",
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_CARD.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    _log(f"[ok  ] data card -> {DATA_CARD}", quiet)
    return DATA_CARD


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download the archive")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    args = parser.parse_args(argv)

    try:
        download(quiet=args.quiet, force=args.force)
        extract(quiet=args.quiet)
        stats = normalise(quiet=args.quiet)
        write_data_card(stats, quiet=args.quiet)
    except (RuntimeError, FileNotFoundError) as exc:  # network/zip failure
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    spam_pct = stats.spam / stats.total * 100 if stats.total else 0.0
    _log(
        f"[ok  ] {stats.total:,} messages "
        f"(ham={stats.ham:,}, spam={stats.spam:,} = {spam_pct:.1f}% spam) "
        f"-> {PROCESSED_CSV}",
        args.quiet,
    )
    if stats.skipped:
        _log(f"[note] skipped {stats.skipped} malformed line(s)", args.quiet)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

