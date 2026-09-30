# Data card — SMS/e-mail spam corpus

| Field | Value |
|---|---|
| Source | UCI Machine Learning Repository — [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) |
| File | `data/processed/sms_spam.csv` |
| Size | 5,574 messages (≈460 KB) |
| Classes | 4,827 `ham` (86.6%) · 747 `spam` (13.4%) |
| Columns | `label` (`ham`/`spam`), `label_code` (0/1), `text` |
| Average length | 80 characters |
| Class imbalance | 6.46 : 1 (handled by `class_weight="balanced"`) |
| Language / channel | English, SMS-style (2011–2012) |
| Licence | CC BY 4.0 (attribution to UCI required) |
| Freshness | Content is historic — see README → Limitations |

## How it was produced

```bash
python scripts/download_data.py     # downloads the UCI archive, extracts,
                                    # normalises to CSV, verifies counts and
                                    # regenerates this data card
```

The script is standard-library only, prints a checksum-style summary, and fails
loudly if the row counts ever deviate from 5,574 / 4,827 / 747.

## Regenerated splits

There are no persisted split files: the stratified 80/20 split is computed
inside `app.model.train_and_evaluate` with `random_state=42`, so every machine
reproduces the same hold-out set (and the same `reports/metrics.json`).

## Sample inputs

`data/sample_messages.csv` (12 curated messages, columns `id,text,expected`) is
the human-curated demo set used by `python -m app.cli demo`, the Streamlit
example dropdown and the documentation scorecard. It is **not** part of training
data.
