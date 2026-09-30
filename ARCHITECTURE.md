# Architecture

Two cooperating layers, one per interface concern: **learn the pattern**, then
**explain the verdict with evidence a human can audit**.

## Data flow

1. **Ingestion** — `app/dataset.py` loads `data/processed/sms_spam.csv`
   (schema `label,label_code,text`) with strict validation and actionable
   error messages (`python scripts/download_data.py` fixes a missing file).
2. **Normalisation** — `app/preprocessing.py` rewrites URLs, e-mails and phone
   numbers into placeholders and squeezes punctuation/whitespace. It is shared
   by *training and inference*, which eliminates train/serve skew.
3. **Learned layer** — `app/model.py`
   `FeatureUnion(TfidfVectorizer(word 1-2), TfidfVectorizer(char_wb 3-5))`
   → `LogisticRegression(class_weight="balanced")`.
   Word n-grams capture topical vocabulary; char n-grams capture obfuscations
   (`cl1ck`, `paypa1`, `baŋk`) that word tokens miss.
   `GridSearchCV` (3-fold stratified, scoring `f1`) picks `C ∈ {0.3, 1.0, 3.0}`.
4. **Rule layer** — `app/signals.py` runs ~25 hand-written detectors grouped as
   `link / identity / lure / language / money / structure`, each returning a
   `Signal(key, label, weight, detail, category)`. Weights saturate through
   `score = 1 − exp(−1.15 · Σw)` so no number of weak signals outvotes one
   strong indicator.
5. **Blend & decision** — `risk = 0.8·P(spam) + 0.2·signal_score`
   with `spam_threshold = 0.50` and `review_threshold = 0.30` (env-tunable).
   Three verdicts, never a silent pass-through of a low-confidence call.
6. **Explanation** — `app/explainer.py` renders the top rule signals offline by
   default; with `AI_PROVIDER=openai|gemini` it asks a public API for a ≤90-word
   narrative over plain HTTPS (`requests`, no vendor SDK) and falls back to the
   offline text on any error. Keys live only in the environment.

## Interfaces

| Interface | Entry point | Notes |
|---|---|---|
| Web | `app/streamlit_app.py` | `@st.cache_resource` keeps one `SpamDetector` per session |
| CLI | `app/cli.py` | `check · train · predict · batch · demo`, all with `--json` |
| Batch | `app/cli.py batch` | CSV in → CSV out with verdict/score/reason columns |
| Programmatic | `SpamDetector.load().predict(text)` | returns a fully populated `Prediction` dataclass |

## Persistence & reproducibility

- One joblib artifact (`models/sms_spam_pipeline.joblib`) stores the fitted
  pipeline **plus** metrics, version and thresholds, so the UI sidebar and
  `check` can report what is actually loaded.
- `random_state=42` everywhere → identical split, metrics and figures.
- `app/reporting.py` writes `reports/metrics.json` (JSON-safe: the private
  `_raw` score arrays are stripped) and three PNGs via the headless `Agg`
  backend.

## Design decisions worth defending

- **Logistic regression over transformers**: interpretable coefficients, tiny
  artifact (≈600 KB), trains in seconds, no GPU — ideal for a reviewable
  prototype.
- **Blend instead of stacking**: keeps the explanation path independent of the
  model, so signals still speak when the model is retrained or unavailable.
- **Three verdicts**: the ambiguous middle band forces a human into the loop
  instead of pretending every message is black or white.
