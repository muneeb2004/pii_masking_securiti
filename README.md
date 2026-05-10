# PII Masking Pipeline

Detects and masks **person names** (PER) and **email addresses** (EMAIL) using:
1. Fine-tuned `bert-base-cased` transformer (NER)
2. Zero-shot `LLaMA 3.2-1B` via Groq API

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Place raw dataset files in `data/`:
```
data/
  data.json        ← training set (28,516 records)
  test_data.json   ← test set     (3,650 records)
```

Each record format:
```json
{
  "tokens":   ["Brad", "Wilk", "is", "a", "drummer"],
  "ner_tags": ["B-PER", "I-PER", "O", "O", "O"],
  "lang":     "en",
  "sequence": "Brad Wilk is a drummer"
}
```

## Execution (run in order)

```bash
# Phase 1 — Data prep + synthetic email injection
python 01_data_prep.py

# Phase 2 — Fine-tune BERT  (~20-25 min on GPU, ~2hr on CPU)
python 02_train.py

# Phase 3 — Evaluate BERT on test set
python 03_evaluate.py

# Phase 4 — Zero-shot LLM evaluation (requires Groq API key)
export GROQ_API_KEY=your_key_here
python 04_llm_inference.py

# Phase 5 — Error analysis + improvement suggestions
python 05_error_analysis.py
```

## Outputs

```
outputs/
  bert_ner/
    best_model/    ← saved tokenizer + model weights

results/
  bert_metrics.json     ← per-entity metrics (BERT)
  bert_predictions.json ← raw token-level predictions
  llm_metrics.json      ← per-entity metrics (LLM) + challenges
  error_analysis.json   ← FP/FN examples + suggestions
```

## Label Schema

| Tag      | Meaning                        |
|----------|-------------------------------|
| O        | Not an entity                  |
| B-PER    | First token of a person name  |
| I-PER    | Continuation of a person name |
| B-EMAIL  | Email address                  |
| I-EMAIL  | (reserved, not used)          |

## Groq API (free)

Sign up at https://console.groq.com → generate API key → set `GROQ_API_KEY`.
Model used: `llama-3.2-1b-preview` (matches task spec, free tier).