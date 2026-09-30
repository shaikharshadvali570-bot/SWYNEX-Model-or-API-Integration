"""Text normalisation shared by training and inference.

Keeping preprocessing in one place guarantees that a message is transformed in
exactly the same way whether it comes from the CLI, the web UI or the batch
runner - a common source of silent "training/serving skew" bugs.
"""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlparse, urlsplit

URL_PATTERN = re.compile(
    r"(?:https?://|ftp://|www\.)[^\s<>\"']+|(?:\b[\w-]+\.(?:com|net|org|biz|info|co|xyz|"
    r"ly|tk|ml|ga|cf|gq|sh|top|click|link|life|live|online|shop|store|uk|us|in|de|ru|"
    r"cn|vip|win|loan|work|account|support|secure|update|verify|pdf)\b(?:/\S*)?)",
    re.IGNORECASE,
)
PHONE_PATTERN = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
REPEATED_PUNCT = re.compile(r"([!?.:,;])\1{2,}")
WHITESPACE = re.compile(r"\s+")

#: URL shorteners frequently used to hide the final destination of a link.
URL_SHORTENERS = frozenset(
    {
        "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
        "adf.ly", "shorturl.at", "cutt.ly", "rb.gy", "rebrand.ly", "bit.do",
        "t.ly", "s.id", "shorte.st", "da.gd", "v.gd", "z.pr", "url.short",
        "clicknewslink", "free-mob", "tiny.cc",
    }
)

#: Top level domains that are heavily abused by SMS phishing ("smishing").
RISKY_TLDS = frozenset(
    {".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".click", ".link",
     ".zip", ".work", ".loan", ".vip", ".cam", ".rest"}
)


def normalise_text(text: str) -> str:
    """Return a cleaned version of ``text`` for the vectoriser.

    Steps: Unicode NFKC normalisation, e-mail/URL/number placeholders,
    repeated punctuation collapsing and whitespace squeezing. Case is kept
    because capitalisation is itself a spam signal for the char n-gram
    feature block.

    Args:
        text: Raw user supplied message.

    Returns:
        The normalised message. Empty/blank input yields an empty string.
    """
    if not text:
        return ""
    cleaned = unicodedata.normalize("NFKC", str(text))
    cleaned = EMAIL_PATTERN.sub(" <email> ", cleaned)
    cleaned = URL_PATTERN.sub(" <url> ", cleaned)
    cleaned = PHONE_PATTERN.sub(" <number> ", cleaned)
    cleaned = REPEATED_PUNCT.sub(r"\1", cleaned)
    return WHITESPACE.sub(" ", cleaned).strip()


def extract_urls(text: str) -> list[str]:
    """Return every URL-like token found in ``text`` (lower-cased, de-duplicated)."""
    found: list[str] = []
    for match in URL_PATTERN.finditer(text or ""):
        token = match.group(0).strip().rstrip(").,;!?'\"]")
        if token and token.lower() not in found:
            found.append(token.lower())
    return found


def url_host(url: str) -> str:
    """Best effort host extraction from a URL-ish token ("http://a.b/c" -> "a.b")."""
    candidate = url if "://" in url else f"http://{url}"
    try:
        return (urlparse(candidate).netloc or urlsplit(candidate).netloc).lower()
    except ValueError:  # malformed URL characters
        return ""


def base_domain(url: str) -> str:
    """Return the registrable-ish domain (``login.bank.example.com`` -> ``example.com``)."""
    host = url_host(url)
    if not host or host.replace(".", "").isdigit():
        return host
    parts = host.split(":")[0].split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def looks_like_shortener(url: str) -> bool:
    """True when the URL points at a known link shortener service."""
    host = url_host(url)
    return any(host == s or host.endswith(f".{s}") for s in URL_SHORTENERS)


def contains_emoji(text: str) -> bool:
    """Cheap heuristic emoji detector based on Unicode category ranges."""
    return any(
        "\U0001F000" <= char <= "\U0001FAFF" or "\u2600" <= char <= "\u27BF"
        for char in (text or "")
    )
