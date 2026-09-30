"""Transparent, rule based risk signals used alongside the machine learning model.

Why this exists
---------------
A TF-IDF + logistic regression classifier is accurate but not very legible:
"spam, 97% confidence" does not tell a security analyst *what* to look at.
This module adds a deterministic rule layer that points at concrete evidence
(short link, brand impersonation, credential request, countdown language...)
and produces an independent 0-1 *signal score*.

The final risk score is a documented blend of the two views (see
:func:`app.model.SpamDetector.predict`), so the prototype keeps the accuracy of
the learned model while remaining auditable - and it still works with no
network access and no API keys.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field

from app.preprocessing import (
    URL_PATTERN,
    base_domain,
    contains_emoji,
    extract_urls,
    looks_like_shortener,
    url_host,
)

HAM, SPAM = "ham", "spam"


@dataclass(frozen=True)
class Signal:
    """One piece of evidence found in a message."""

    key: str
    label: str
    weight: float
    detail: str
    category: str = "language"

    def as_dict(self) -> dict[str, str | float]:
        """JSON friendly form used by the CLI, UI and batch CSV export."""
        return asdict(self)


@dataclass
class SignalReport:
    """Aggregated output of :func:`analyse`."""

    score: float = 0.0
    signals: list[Signal] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    riskiest_url: str | None = None

    @property
    def top(self) -> list[Signal]:
        """Signals sorted by descending weight (most important first)."""
        return sorted(self.signals, key=lambda s: s.weight, reverse=True)

    @property
    def labels(self) -> list[str]:
        """Human readable labels, most important first."""
        return [s.label for s in self.top]

    def as_dict(self) -> dict[str, object]:
        """Serialisable representation persisted in ``reports/batch_output.csv``."""
        return {
            "signal_score": round(self.score, 4),
            "signal_labels": "; ".join(self.labels),
            "urls": "; ".join(self.urls),
            "riskiest_url": self.riskiest_url or "",
        }


# --------------------------------------------------------------------------- #
# Lexicons                                                                    #
# --------------------------------------------------------------------------- #

#: Brands that are most frequently impersonated in smishing / BEC campaigns.
IMPERSONATED_BRANDS: dict[str, tuple[str, ...]] = {
    "paypal": ("paypal", "paypa1", "paypaI"),
    "amazon": ("amazon", "arnczon", "armazon", "arnazon"),
    "google": ("google", "g00gle", "googie"),
    "microsoft": ("microsoft", "rniicrosoft", "rnicrosoft"),
    "apple": ("apple id", "applepay", "icloud", "apple"),
    "netflix": ("netflix", "netflx"),
    "dhl": ("dhl", "dpex", "fedex", "ups", "usps", "royal mail", "courier"),
    "bank": ("barclays", "hsbc", "lloyds", "natwest", "santander", "revolut",
            "chase", "wells fargo", "citibank", "bank of america", "your bank"),
    "telco": ("vodafone", "o2 ", "ee mobile", "t-mobile", "three network", "airtel", "jio"),
    "government": ("hmrc", "irs", "tax credit", "dvla", "medicare", "social security"),
}

CREDENTIAL_TERMS = (
    r"verify (?:your )?(?:account|identity|login|details|card)",
    r"confirm (?:your )?(?:identity|account|details|payment)",
    r"(?:log|sign)\s*in (?:to|now|securely)",
    r"one[- ]?time (?:password|code)|\botp\b|\bpin\b|passcode",
    r"update (?:your )?(?:card|payment|billing|bank|kyc|details)",
    r"password|credential|security (?:question|answer)|2fa",
)
PRIZE_TERMS = (
    r"\bwon\b|\bwinner\b|\bwinn(?:ing|er)s?\b|\bprize\b|\blottery\b|\bdraw\b",
    r"\bclaim\b|\bfree (?:gift|entry|sim|flight|voucher|phone|money|£|\$)",
    r"\bvoucher\b|\bgift ?card\b|\bcash ?back\b|\breward\b|\bbonus\b",
    r"congratulations|congrats|\bc1ng\b|\bqualified\b|\bselected\b",
)
URGENCY_TERMS = (
    r"\burgent(?:ly)?\b|\bimmediately\b|\bact now\b|\bright now\b|\basap\b",
    r"expires? (?:today|soon|in)|expiring|expiration|valid (?:only )?for",
    r"last (?:reminder|notice|chance)|final notice|last call",
    r"within (?:the )?(?:next )?\d+ ?(?:hours?|mins?|minutes?|days?)",
    r"suspend|deactivat|restrict|lock(?:ed)?|closed? (?:automatically|soon)|"
    r"will be closed|forfeit|lost? forever|deadline",
)
MONEY_TERMS = (
    r"(?:£|\$|€|₹|usd|gbp|eur)\s?\d|\d+(?:\.\d{1,2})?\s?(?:pounds?|dollars?|euros?|gbp|usd)",
    r"\bpayment\b|\binvoice\b|\boverdue\b|\bunpaid\b|\brefund\b|\btransfer\b",
    r"\bcrypto\b|\bbitcoin\b|\bethereum\b|\bforex\b|\binvestment\b|\btrading\b",
    r"guaranteed (?:returns?|profit)|double your|10x|\bsend (?:\d|us)\b",
)
OPT_OUT_TERMS = (
    r"\bstop\b (?:to unsubscribe|and unsubscribe|further|all)",
    r"text \s*stop|reply stop|to opt ?out|unsubscribe|end? msg|msg&data ?rates",
)

# Compiled once, reused for every prediction.
_CREDENTIAL_RE = re.compile("|".join(CREDENTIAL_TERMS), re.IGNORECASE)
_PRIZE_RE = re.compile("|".join(PRIZE_TERMS), re.IGNORECASE)
_URGENCY_RE = re.compile("|".join(URGENCY_TERMS), re.IGNORECASE)
_MONEY_RE = re.compile("|".join(MONEY_TERMS), re.IGNORECASE)
_OPT_OUT_RE = re.compile("|".join(OPT_OUT_TERMS), re.IGNORECASE)
_IP_URL_RE = re.compile(r"https?://\d{1,3}(?:\.\d{1,3}){3}", re.IGNORECASE)
_SHORTCODE_RE = re.compile(r"\b(?:text|reply|txt|call|dial)\s*[\d\s+().-]{3,}\b", re.IGNORECASE)
_LEET_RE = re.compile(r"\b(?:cl1ck|cllck|clik|whatsa?p?p?|faceb00k|rnicrosoft|paypa1)\b", re.IGNORECASE)

# --------------------------------------------------------------------------- #
# Detectors                                                                   #
# --------------------------------------------------------------------------- #

def _brand_hits(text: str) -> list[str]:
    """Return the impersonation-prone brands mentioned in ``text``."""
    lowered = f" {text.lower()} "
    hits: list[str] = []
    for brand, variants in IMPERSONATED_BRANDS.items():
        if any(variant in lowered for variant in variants):
            hits.append(brand)
    return hits


def _url_signals(text: str) -> tuple[list[Signal], list[str], str | None]:
    """Inspect every link in the message and build link related signals."""
    signals: list[Signal] = []
    urls = extract_urls(text)
    if not urls:
        return signals, urls, None

    signals.append(
        Signal("has_url", "Contains an external link", 0.10,
               f"{len(urls)} link(s): {', '.join(urls[:3])}", "link")
    )
    riskiest, risk_weight = urls[0], 0.10

    for url in urls:
        host = url_host(url) or url
        if looks_like_shortener(url):
            signals.append(Signal("url_shortener", "Link shortener hides the destination",
                                  0.30, f"shortened link -> {host}", "link"))
            if risk_weight < 0.30:
                risk_weight, riskiest = 0.30, url
        if _IP_URL_RE.search(url):
            signals.append(Signal("url_ip_literal", "Link points at a raw IP address",
                                  0.45, f"IP based URL: {host}", "link"))
            risk_weight, riskiest = 0.45, url
        if any(host.endswith(tld) for tld in (".tk", ".ml", ".ga", ".cf", ".gq", ".xyz",
                                              ".top", ".click", ".link", ".zip", ".vip")):
            signals.append(Signal("url_risky_tld", "Link uses an abuse-prone domain suffix",
                                  0.25, f"suspicious TLD in {host}", "link"))
            if risk_weight < 0.25:
                risk_weight, riskiest = 0.25, url
        if not url.startswith("http"):
            signals.append(Signal("url_bare", "Bare domain without scheme (common in smishing)",
                                  0.08, f"bare link: {url}", "link"))
        if re.search(r"[-_@]", host):
            signals.append(Signal("url_obfuscated_host", "Unusual characters in the host name",
                                  0.22, f"host: {host}", "link"))
            if risk_weight < 0.22:
                risk_weight, riskiest = 0.22, url

    return signals, urls, riskiest if risk_weight >= 0.22 else None



def analyse(text: str) -> SignalReport:
    """Run every deterministic detector over ``text``.

    Returns:
        A :class:`SignalReport` whose ``score`` is bounded in ``[0, 1]`` using a
        saturating sum, so several weak hits cannot out-vote one strong hit.
    """
    if not text or not text.strip():
        return SignalReport()

    url_sigs, urls, riskiest = _url_signals(text)
    signals: list[Signal] = list(url_sigs)
    signals += _brand_vs_domain_signal(text, urls)

    plain = URL_PATTERN.sub(" ", text)
    letters = [c for c in plain if c.isalpha()]
    caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0.0

    match = _CREDENTIAL_RE.search(text)
    if match:
        signals.append(Signal("credential_request", "Asks for credentials, OTP or account details",
                              0.45, f"matched: {match.group(0).strip()!r}", "identity"))
    match = _PRIZE_RE.search(text)
    if match:
        signals.append(Signal("prize_lure", "Prize, winner or free gift lure",
                              0.35, f"matched: {match.group(0).strip()!r}", "lure"))
    match = _URGENCY_RE.search(text)
    if match:
        signals.append(Signal("urgency_pressure", "Time pressure / threat of account loss",
                              0.30, f"matched: {match.group(0).strip()!r}", "language"))
    match = _MONEY_RE.search(text)
    if match:
        signals.append(Signal("money_hook", "Money, payment or investment hook",
                              0.28, f"matched: {match.group(0).strip()!r}", "money"))
    match = _OPT_OUT_RE.search(text)
    if match:
        signals.append(Signal("opt_out_marker", "Marketing opt-out / Msg&Data rates wording",
                              0.15, "bulk SMS compliance wording", "structure"))
    match = _SHORTCODE_RE.search(text)
    if match:
        signals.append(Signal("premium_action", "Tells you to text or call a number to act",
                              0.20, "call-to-action pointing at a phone number/shortcode",
                              "lure"))
    if caps_ratio > 0.45 and len(letters) > 12:
        signals.append(Signal("shouting", "Message is mostly CAPITAL LETTERS",
                              0.18, f"{caps_ratio:.0%} of letters are uppercase", "structure"))
    if re.search(r"[!?]{2,}", plain):
        signals.append(Signal("repeated_punctuation", "Repeated punctuation (!!!, ???)",
                              0.10, "multiple repeated punctuation marks", "structure"))
    match = _LEET_RE.search(plain)
    if match:
        signals.append(Signal("leet_obfuscation", "Typosquat-style letter swaps (cl1ck, paypa1)",
                              0.35, f"matched: {match.group(0)!r}", "identity"))
    if contains_emoji(text):
        signals.append(Signal("emoji_bait", "Emoji used as visual bait", 0.08,
                              "emoji characters present", "structure"))

    # Saturating sum: weights add up quickly but the score can never exceed 1.
    total = sum(signal.weight for signal in signals)
    score = 1.0 - math.exp(-1.15 * total) if total else 0.0
    return SignalReport(score=round(min(score, 1.0), 4), signals=signals,
                        urls=urls, riskiest_url=riskiest)


def _brand_is_legit(brand: str, domains: set[str]) -> bool:
    """True when ``domains`` contains the real site of ``brand`` (label match).

    A substring test is not enough: ``paypal-secure.tk`` *contains* "paypal" yet is
    a classic phishing host, while ``paypal.com`` and ``www.paypal.com`` are fine.
    Matching whole DNS labels before the public suffix gets both cases right.
    """
    tokens = {brand}
    for variant in IMPERSONATED_BRANDS.get(brand, ()):
        tokens.update(part for part in variant.split() if part)
    return any(
        domain == token or domain.startswith(f"{token}.")
        for domain in domains if domain
        for token in tokens
    )


def _brand_vs_domain_signal(text: str, urls: list[str]) -> list[Signal]:
    """Flag "brand named in text but brand domain not in the link" mismatches."""
    brands = _brand_hits(text)
    if not brands or not urls:
        return []
    domains = {base_domain(u) for u in urls}
    mismatches = [b for b in brands if not _brand_is_legit(b, domains)]
    if not mismatches:
        return []
    return [
        Signal("brand_mismatch", "Mentions a well known brand but links elsewhere",
               0.40, f"mentions {', '.join(mismatches)} but links to "
                     f"{', '.join(sorted(d for d in domains if d)) or 'no known domain'}",
               "identity")
    ]

