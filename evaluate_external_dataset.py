"""Independent external-dataset evaluation for the PII NER model.

Loads the WikiANN English (pan-x) test split via HuggingFace Datasets, maps
its label scheme onto the project's five-tag set (B-PER / I-PER / B-EMAIL /
I-EMAIL / O), injects synthetic email entities using the same utility as
Phase 1, runs the fine-tuned BERT checkpoint, and produces the same metrics
table that Phase 3 generates on the WikiNeural test set.

Run end-to-end with::

    python evaluate_external_dataset.py

Results are written to ``results/external_metrics.json`` and
``results/external_predictions.json``.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset as hf_load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

# Reuse email-injection helpers from Phase 1 without re-running its main().
# 01_data_prep.py cannot be imported with a standard ``import`` statement
# because its name starts with a digit; importlib is the correct approach.
# The module-level side effects (Faker seeding) are harmless.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "data_prep", Path(__file__).parent / "01_data_prep.py"
)
_data_prep = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_data_prep)  # type: ignore[union-attr]
augment_with_emails = _data_prep.augment_with_emails

from metrics_utils import compute_entity_metrics, print_metrics_table

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED: int = 42
MODEL_DIR: Path = Path("outputs") / "bert_ner" / "best_model"
RESULTS_DIR: Path = Path("results")

# Label set must be identical to training.
LABEL_LIST: list[str] = ["O", "B-PER", "I-PER", "B-EMAIL", "I-EMAIL"]
LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL: dict[int, str] = {idx: label for label, idx in LABEL2ID.items()}

MAX_LENGTH: int = 128
BATCH_SIZE: int = 32

# Fraction of PER-containing sentences that receive a synthetic email.
INJECT_RATE: float = 0.18

# WikiANN (pan-x) English tag index → project tag string.
# Integer indices come from the ClassLabel feature of the ``wikiann`` dataset.
# Feature order: O B-PER I-PER B-ORG I-ORG B-LOC I-LOC
_WIKIANN_ID_TO_PROJECT_TAG: dict[int, str] = {
    0: "O",       # O
    1: "B-PER",   # B-PER
    2: "I-PER",   # I-PER
    3: "O",       # B-ORG → collapse
    4: "O",       # I-ORG → collapse
    5: "O",       # B-LOC → collapse
    6: "O",       # I-LOC → collapse
}


# ---------------------------------------------------------------------------
# WikiANN English loading and label conversion
# ---------------------------------------------------------------------------


def load_wikiann_test() -> list[dict]:
    """Download (or use cached) WikiANN English and return the test split.

    WikiANN (pan-x) is a Parquet-based dataset — no loading script is required.
    Each returned record has the shape expected by the rest of the pipeline::

        {"tokens": [...], "ner_tags": [...], "lang": "en"}

    WikiANN tags other than PER are collapsed to ``"O"``.

    Returns:
        List of record dicts with string BIO tags.
    """
    logger.info("Loading WikiANN English test split from HuggingFace Datasets...")
    dataset = hf_load_dataset("wikiann", "en", split="test")
    logger.info("Downloaded %d sentences.", len(dataset))

    records: list[dict] = []
    for example in dataset:
        tokens: list[str] = example["tokens"]
        wikiann_ids: list[int] = example["ner_tags"]
        project_tags = [_WIKIANN_ID_TO_PROJECT_TAG[tag_id] for tag_id in wikiann_ids]
        records.append(
            {
                "tokens": tokens,
                "ner_tags": project_tags,
                "lang": "en",
            }
        )

    per_sentences = sum(
        1 for r in records if any(t in ("B-PER", "I-PER") for t in r["ner_tags"])
    )
    logger.info(
        "Label conversion complete — %d records, %d contain PER entities.",
        len(records),
        per_sentences,
    )
    return records


# ---------------------------------------------------------------------------
# Tokenisation (mirrors Phase 3 / training alignment)
# ---------------------------------------------------------------------------


def tokenize_and_align_labels(
    records: list[dict],
    tokenizer: Any,
) -> list[dict[str, list[int]]]:
    """Tokenize records and align BIO integer labels to subword tokens.

    Continuation subwords and special tokens receive label ``-100`` so they
    are excluded from evaluation, matching the behaviour in Phase 3.

    Args:
        records: Records with ``tokens`` and ``ner_tags`` (string) keys.
        tokenizer: A HuggingFace fast tokenizer.

    Returns:
        List of feature dicts with ``input_ids``, ``attention_mask``, and
        ``labels``.
    """
    processed: list[dict[str, list[int]]] = []

    for record in records:
        word_tokens = record["tokens"]
        integer_tags = [LABEL2ID[tag] for tag in record["ner_tags"]]

        encoding = tokenizer(
            word_tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

        word_ids = encoding.word_ids()
        aligned_labels: list[int] = []
        previous_word_id: int | None = None

        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != previous_word_id:
                aligned_labels.append(integer_tags[word_id])
            else:
                aligned_labels.append(-100)
            previous_word_id = word_id

        processed.append(
            {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels": aligned_labels,
            }
        )

    return processed


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class NERDataset(Dataset):
    """PyTorch Dataset wrapping pre-tokenized NER feature dicts.

    Args:
        features: Feature dicts produced by :func:`tokenize_and_align_labels`.
    """

    def __init__(self, features: list[dict[str, list[int]]]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: torch.tensor(value) for key, value in self.features[index].items()}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference(
    model: AutoModelForTokenClassification,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[list[str]], list[list[str]]]:
    """Run batched inference and decode predictions to BIO label sequences.

    Positions labelled ``-100`` (special tokens, subword continuations) are
    excluded from both the prediction and the ground-truth sequences.

    Args:
        model: Fine-tuned token classification model.
        dataloader: DataLoader over the external test dataset.
        device: Torch device for inference.

    Returns:
        Tuple of ``(all_predictions, all_true_labels)`` — lists of BIO string
        sequences, one per input sentence.
    """
    model.eval()
    all_predictions: list[list[str]] = []
    all_true_labels: list[list[str]] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            true_labels = batch["labels"]

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predictions = np.argmax(logits.cpu().numpy(), axis=2)

            for prediction_seq, label_seq in zip(predictions, true_labels.numpy()):
                filtered_preds: list[str] = []
                filtered_labels: list[str] = []
                for prediction, label in zip(prediction_seq, label_seq):
                    if label != -100:
                        filtered_preds.append(ID2LABEL[prediction])
                        filtered_labels.append(ID2LABEL[label])
                all_predictions.append(filtered_preds)
                all_true_labels.append(filtered_labels)

    return all_predictions, all_true_labels


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Execute the full independent external-dataset evaluation pipeline."""
    random.seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("INDEPENDENT EXTERNAL DATASET EVALUATION (WikiANN English)")
    logger.info("=" * 60)
    logger.info("Device: %s", device)

    # -- 1. Load and convert WikiANN English test split ----------------------
    records = load_wikiann_test()

    # -- 2. Inject synthetic emails ------------------------------------------
    logger.info("Injecting synthetic email entities (rate=%.0f%%)...", INJECT_RATE * 100)
    augmented_records = augment_with_emails(records, INJECT_RATE, "wikiann-en-test")
    email_count = sum(
        1 for r in augmented_records if any(t == "B-EMAIL" for t in r["ner_tags"])
    )
    logger.info("%d sentences now contain EMAIL entities.", email_count)

    # -- 3. Load model and tokenizer -----------------------------------------
    logger.info("Loading model from %s ...", MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForTokenClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    logger.info("Model loaded.")

    # -- 4. Tokenise and build DataLoader ------------------------------------
    logger.info("Tokenizing %d records...", len(augmented_records))
    features = tokenize_and_align_labels(augmented_records, tokenizer)
    dataset = NERDataset(features)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collator)

    # -- 5. Inference --------------------------------------------------------
    logger.info("Running inference...")
    all_predictions, all_true_labels = run_inference(model, dataloader, device)
    logger.info("Processed %d sequences.", len(all_predictions))

    # -- 6. Metrics ----------------------------------------------------------
    logger.info("Computing metrics...")
    metrics = compute_entity_metrics(all_true_labels, all_predictions)

    logger.info("\n--- External Dataset Metrics (WikiANN English test) ---")
    print_metrics_table(metrics)

    # -- 7. Save results -----------------------------------------------------
    metrics_path = RESULTS_DIR / "external_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    predictions_path = RESULTS_DIR / "external_predictions.json"
    payload = [
        {"true": true_seq, "pred": pred_seq}
        for true_seq, pred_seq in zip(all_true_labels, all_predictions)
    ]
    with predictions_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    logger.info("Raw predictions saved to %s", predictions_path)

    logger.info("External evaluation complete.")


if __name__ == "__main__":
    main()
