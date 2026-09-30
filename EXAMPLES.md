# Example runs

Every transcript below was produced on this machine with the committed model
(`python -m app.cli train`, 2026-09-29) — reproduce them with the commands shown.

## 1. Environment / diagnostics — `python -m app.cli check`

```json
{
  "app": "SpamGuard AI v1.0.0",
  "python": "3.13.5",
  "settings": {
    "ai_provider": "none",
    "ai_status": "disabled (offline heuristics)",
    "ai_key": "(not set)",
    "model_file": "sms_spam_pipeline.joblib",
    "spam_threshold": "0.50",
    "review_threshold": "0.30"
  },
  "dataset": { "total": 5574, "ham": 4827, "spam": 747, "spam_pct": 13.4 },
  "model": {
    "accuracy": 0.9946, "precision": 1.0, "recall": 0.9597,
    "f1": 0.9795, "roc_auc": 0.9958, "pr_auc": 0.9888
  },
  "smoke_test": { "verdict": "spam", "risk_score": 0.6265 }
}
```

## 2. Training — `python -m app.cli train`

```
Training complete
  dataset          : 5,574 msgs (4,827 ham / 747 spam = 13.4% spam)
  split            : 4,459 train / 1,115 test (test_size=0.2)
  algorithm        : Tfidf(word 1-2 + char 3-5) -> LogisticRegression(balanced) (C=3.0)
  CV F1 (3-fold)   : 0.956 (+/- 0.001)
  test accuracy    : 0.995
  test precision   : 1.000
  test recall      : 0.960
  test F1          : 0.980
  ROC-AUC / PR-AUC : 0.996 / 0.989
  confusion matrix : [[966, 0], [6, 143]]
  trained in       : 8.99 s
```

## 3. Single prediction — `python -m app.cli predict "…"`

```
[SPAM] verdict: SPAM
  risk score      : 0.785
  model P(spam)   : 0.788
  rule signals    : 0.776
  confidence      : medium
  latency         : 9.8 ms
  links found     : http://192.168.1.10/secure-login
  riskiest link   : http://192.168.1.10/secure-login
  why:
    - Link points at a raw IP address - IP based URL: 192.168.1.10
    - Asks for credentials, OTP or account details - matched: 'Verify your identity'
    - Time pressure / threat of account loss - matched: 'URGENT'
    - Contains an external link - 1 link(s): http://192.168.1.10/secure-login
  advice          : Do not click or reply. Delete the message, and if you already shared details, contact your bank/security team immediately.
```

## 4. JSON contract — `python -m app.cli predict "…" --json`

```json
{
  "text": "Congratulations! You have WON a 500 gift voucher...",
  "verdict": "spam",
  "risk_score": 0.887,
  "spam_probability": 0.999,
  "signal_score": 0.437,
  "confidence": "medium",
  "reasons": ["Prize, winner or free gift lure - matched: 'Congratulations'", "..."],
  "signals": [ { "key": "prize_lure", "label": "Prize, winner or free gift lure",
                 "weight": 0.35, "detail": "matched: 'Congratulations'",
                 "category": "lure" } ],
  "urls": [], "riskiest_url": null,
  "advice": "Do not click or reply...",
  "model_version": "1.0.0", "latency_ms": 3.1
}
```

## 5. Batch — `python -m app.cli batch --input data/sample_messages.csv`

```
Scored 12 message(s) -> data/sample_messages_scored.csv
  spam=7  review=0  ham=5
```

The output CSV keeps the input columns and appends `verdict`, `risk_score`,
`spam_probability`, `signal_score`, `confidence`, `reasons`.

## 6. Scorecard — `python -m app.cli demo`

| Input (abridged) | Expected | Verdict | Risk | Top evidence |
|---|---|---|---:|---|
| `URGENT: We detected a suspicious login … 192.168.1.10/secure-login` | spam | 🚨 SPAM | 0.785 | raw IP, credential request, urgency |
| `Congratulations! You have WON a 500 gift voucher …` | spam | 🚨 SPAM | 0.887 | prize lure, opt-out wording |
| `Hi team, can we move standup to 10:30 tomorrow?` | ham | ✅ HAM | 0.116 | no indicators |
| `Your parcel … bit.ly/dlv-redir` | spam | 🚨 SPAM | 0.549 | shortener, bare domain, deadline |
| `Lunch at 1 pm today? …` | ham | ✅ HAM | 0.061 | no indicators |
| `Dear customer your online banking has been locked … verify-login.tk` | spam | 🚨 SPAM | 0.86+ | credential + risky TLD |
| `WINNER! … 1000000 USD … 48 hours … 09069775139` | spam | 🚨 SPAM | 0.923 | prize, deadline, money |
| `Meeting notes attached …` | ham | ✅ HAM | 0.139 | no indicators |
| `Final reminder: your invoice 2291 … billing-update.xyz` | spam | 🚨 SPAM | 0.898 | money hook, risky TLD |
| `Your Apple ID … apple-id-verify.acount-support.top` | spam | 🚨 SPAM | 0.846 | brand mismatch, credential |

All 12/12 classified as expected.

## 7. Optional LLM explanation — `python -m app.cli predict "…" --explain`

With `AI_PROVIDER=none` (default) the explanation is generated offline:

```
ai explanation (local):
  Offline explanation generated from 4 rule-based indicator(s): Link points at a
  raw IP address, Asks for credentials, OTP or account details, ... The classifier
  also scored this message 79% likely to be spam. Do not click or reply...
```

Set `AI_PROVIDER=openai` (or `gemini`) plus the matching key in `.env` to have a
90-word narrative instead; on any network/auth error the CLI logs a warning and
falls back to the same offline text.
