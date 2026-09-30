# SWYNEX Task 2 — Model or API Integration

## Objective

Build a small prototype that integrates a model, library or public AI API for a defined problem and provide example inputs and outputs.

## Prototype

**SpamGuard AI** detects spam/phishing risk in SMS and e-mail text.

### Integrated components

- Scikit-learn TF-IDF + Logistic Regression classifier
- Deterministic security-rule layer for explainability
- Optional OpenAI public API integration
- Optional Google Gemini public API integration
- Streamlit user interface
- CLI and batch interfaces

## Input

A plain-text SMS or e-mail message.

## Output

- Verdict: `spam`, `review` or `ham`
- Risk score
- Model spam probability
- Confidence
- Detected security indicators
- Recommended action
- Optional AI-generated explanation

## Example

**Input**

```text
URGENT: We detected a suspicious login. Your account will be closed in 24 hours. Verify your identity now http://192.168.1.10/secure-login
```

**Output**

```text
verdict: SPAM
risk score: 0.785
model P(spam): 0.788
confidence: medium
```

The detailed output and additional examples are available in `docs/EXAMPLES.md` and `examples/`.

## Integration Flow

```text
Input
  ↓
Preprocessing
  ↓
TF-IDF + Logistic Regression
  ↓
Security rule signals
  ↓
Risk score + verdict
  ↓
Optional OpenAI/Gemini explanation
  ↓
Streamlit / CLI output
```

## API Key Handling

No secret key is committed. The repository includes only `.env.example`. Real credentials must be supplied locally through environment variables.
