# PII Masking Pipeline

A two-approach PII detection system that identifies and masks **person names** (PER) and **email addresses** (EMAIL) from text, comparing a fine-tuned transformer model against a zero-shot large language model.

| Approach | Model | Overall F1 | Precision | Recall |
|---|---|---|---|---|
| Fine-tuned BERT | `bert-base-cased` | **0.9902** | 0.9909 | 0.9894 |
| Zero-shot LLM | `llama-3.1-8b-instant` (Groq) | 0.9322 | 0.9524 | 0.9129 |

---

## Project Structure

```
pii_masking/
├── 01_data_prep.py          # Data loading, email injection, augmentation
├── 02_train.py              # BERT fine-tuning
├── 03_evaluate.py           # BERT evaluation on test set
├── 04_llm_inference.py      # Zero-shot LLaMA evaluation via Groq API
├── 05_error_analysis.py     # FP/FN analysis + improvement suggestions
├── evaluate_external_dataset.py  # Evaluate on an external dataset
├── metrics_utils.py         # Shared metrics helpers
├── requirements.txt
├── data/
│   ├── data.json            # Raw training set  (28,516 records)
│   ├── test_data.json       # Raw test set       (3,650 records)
│   ├── train_augmented.json # Augmented training set
│   └── test_augmented.json  # Augmented test set
├── outputs/
│   └── bert_ner/
│       └── best_model/      # Saved tokenizer + model weights
└── results/
    ├── bert_metrics.json
    ├── bert_predictions.json
    ├── llm_metrics.json
    ├── error_analysis.json
    ├── external_metrics.json
    └── external_predictions.json
```

---

## Setup

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
```

**Dependencies:** `transformers>=4.40`, `torch>=2.0`, `datasets`, `evaluate`, `seqeval`, `scikit-learn`, `faker`, `groq`

---

## Dataset Format

Each record in `data.json` / `test_data.json`:

```json
{
  "tokens":   ["Brad", "Wilk", "is", "a", "drummer"],
  "ner_tags": ["B-PER", "I-PER", "O", "O", "O"],
  "lang":     "en",
  "sequence": "Brad Wilk is a drummer"
}
```

### Label Schema

| Tag | Meaning |
|---|---|
| `O` | Not an entity |
| `B-PER` | First token of a person name |
| `I-PER` | Continuation of a person name |
| `B-EMAIL` | Email address token |
| `I-EMAIL` | Reserved (not used) |

---

## Pipeline

Run the scripts in order:

```bash
# Phase 1 — Data prep: load datasets and inject synthetic email entities (18% of PER sentences)
python 01_data_prep.py

# Phase 2 — Fine-tune bert-base-cased on augmented training set (~20-25 min GPU / ~2hr CPU)
python 02_train.py

# Phase 3 — Evaluate fine-tuned BERT on the held-out test set
python 03_evaluate.py

# Phase 4 — Zero-shot LLM evaluation via Groq API (requires API key)
# Windows
$env:GROQ_API_KEY="your_key_here"
# Linux/macOS
export GROQ_API_KEY=your_key_here
python 04_llm_inference.py

# Phase 5 — Error analysis: false positive/negative examples + improvement suggestions
python 05_error_analysis.py

# Optional — Evaluate on an external dataset
python evaluate_external_dataset.py
```

---

## Results

### BERT (fine-tuned `bert-base-cased`)

| Entity | Precision | Recall | F1 |
|---|---|---|---|
| PER | 0.9903 | 0.9887 | 0.9895 |
| EMAIL | 1.0000 | 1.0000 | 1.0000 |
| **Overall** | **0.9909** | **0.9894** | **0.9902** |

### LLaMA (zero-shot `llama-3.1-8b-instant` via Groq, 180 samples)

| Entity | Precision | Recall | F1 |
|---|---|---|---|
| PER | 0.9487 | 0.9044 | 0.9261 |
| EMAIL | 0.9706 | 1.0000 | 0.9851 |
| **Overall** | **0.9524** | **0.9129** | **0.9322** |

### Key Observations

- The fine-tuned BERT model significantly outperforms zero-shot LLaMA on PER detection (+6.3 F1 points).
- Both models achieve near-perfect EMAIL detection; BERT reaches 100% F1.
- LLM limitations include hallucinated entity boundaries, multi-token name truncation, and JSON format failures requiring regex fallback.

---

## Groq API Setup

1. Sign up at [console.groq.com](https://console.groq.com)
2. Generate an API key
3. Set the `GROQ_API_KEY` environment variable before running `04_llm_inference.py`

The free tier supports ~20–30 requests/minute; retry logic is built into the script.