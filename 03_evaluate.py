"""BERT evaluation pipeline on the held-out test set.

Loads the best fine-tuned checkpoint, runs token-classification inference
on the augmented test set, computes per-entity-type metrics, and persists
results alongside raw predictions for downstream error analysis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

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

DATA_DIR: Path = Path("data")
MODEL_DIR: Path = Path("outputs") / "bert_ner" / "best_model"
RESULTS_DIR: Path = Path("results")
TEST_FILE: Path = DATA_DIR / "test_augmented.json"

LABEL_LIST: list[str] = ["O", "B-PER", "I-PER", "B-EMAIL", "I-EMAIL"]
LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL: dict[int, str] = {idx: label for label, idx in LABEL2ID.items()}

MAX_LENGTH: int = 128
BATCH_SIZE: int = 32


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of record dicts.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of parsed JSON objects.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# Tokenisation (mirrors training alignment)
# ---------------------------------------------------------------------------


def tokenize_and_align_labels(
    records: list[dict],
    tokenizer: Any,
) -> list[dict[str, list[int]]]:
    """Tokenize records and align BIO integer labels to subword tokens.

    Continuation subwords and special tokens receive label ``-100`` to be
    excluded from evaluation metrics.

    Args:
        records: Dataset records with ``tokens`` and ``ner_tags`` keys.
        tokenizer: A Hugging Face fast tokenizer instance.

    Returns:
        List of feature dicts with ``input_ids``, ``attention_mask``,
        and ``labels``.
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
        features: List of feature dicts produced by
            :func:`tokenize_and_align_labels`.
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

    Ignores positions labelled ``-100`` (special tokens, subword continuations).

    Args:
        model: Fine-tuned token classification model.
        dataloader: DataLoader over the test dataset.
        device: Torch device for inference.

    Returns:
        Tuple of ``(all_predictions, all_true_labels)`` where each element
        is a list of BIO string sequences.
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
    """Execute BERT evaluation on the held-out test set."""
    RESULTS_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("PHASE 3 - BERT EVALUATION ON TEST SET")
    logger.info("=" * 60)
    logger.info("Device: %s", device)

    logger.info("Loading test data from %s ...", TEST_FILE)
    test_records = load_jsonl(TEST_FILE)
    logger.info("%d records loaded", len(test_records))

    logger.info("Loading model from %s ...", MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForTokenClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    logger.info("Model loaded.")

    logger.info("Tokenizing test set...")
    test_features = tokenize_and_align_labels(test_records, tokenizer)
    test_dataset = NERDataset(test_features)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collator)

    logger.info("Running inference...")
    all_predictions, all_true_labels = run_inference(model, test_loader, device)
    logger.info("Processed %d sequences.", len(all_predictions))

    logger.info("Computing metrics...")
    metrics = compute_entity_metrics(all_true_labels, all_predictions)
    print_metrics_table(metrics)

    metrics_path = RESULTS_DIR / "bert_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    predictions_path = RESULTS_DIR / "bert_predictions.json"
    payload = [
        {"true": true_seq, "pred": pred_seq}
        for true_seq, pred_seq in zip(all_true_labels, all_predictions)
    ]
    with predictions_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    logger.info("Raw predictions saved to %s", predictions_path)

    logger.info("Phase 3 complete.")


if __name__ == "__main__":
    main()