"""BERT fine-tuning pipeline for PII named-entity recognition.

Fine-tunes ``bert-base-cased`` on the augmented training set produced by
Phase 1.  Supports CPU and CUDA execution with automatic mixed-precision
selection.

Compatibility: transformers >= 4.46, accelerate >= 1.1.0, Python 3.10+.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import evaluate
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

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
MODEL_NAME: str = "bert-base-cased"

DATA_DIR: Path = Path("data")
OUTPUT_DIR: Path = Path("outputs") / "bert_ner"
AUG_TRAIN_FILE: Path = DATA_DIR / "train_augmented.json"

LABEL_LIST: list[str] = ["O", "B-PER", "I-PER", "B-EMAIL", "I-EMAIL"]
LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL: dict[int, str] = {idx: label for label, idx in LABEL2ID.items()}

MAX_LENGTH: int = 128
VALIDATION_SPLIT: float = 0.15
BATCH_SIZE: int = 16
NUM_EPOCHS: int = 3
LEARNING_RATE: float = 2e-5
WEIGHT_DECAY: float = 0.01
EVAL_BATCH_SIZE: int = 32
WARMUP_FRACTION: float = 0.1
LOGGING_STEPS: int = 100
EARLY_STOPPING_PATIENCE: int = 2

USE_FP16: bool = torch.cuda.is_available()

_seqeval = evaluate.load("seqeval")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of record dicts.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of parsed JSON objects, one per non-empty line.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# Tokenisation and label alignment
# ---------------------------------------------------------------------------


def tokenize_and_align_labels(
    records: list[dict],
    tokenizer: Any,
) -> list[dict[str, list[int]]]:
    """Tokenize word sequences and align BIO labels to subword tokens.

    Special tokens and continuation subwords receive label ``-100`` so they
    are ignored by the cross-entropy loss.

    Args:
        records: Dataset records with ``tokens`` and ``ner_tags`` keys.
        tokenizer: A Hugging Face fast tokenizer instance.

    Returns:
        List of feature dicts with ``input_ids``, ``attention_mask``,
        and ``labels`` (integer-encoded).
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
    """PyTorch Dataset wrapping pre-tokenized NER features.

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
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(eval_pred: tuple) -> dict[str, float]:
    """Compute sequence-level precision, recall, F1, and accuracy via seqeval.

    Filters ``-100`` padding labels before evaluation.

    Args:
        eval_pred: Tuple of ``(logits, labels)`` as provided by the
            :class:`~transformers.Trainer`.

    Returns:
        Dict with ``precision``, ``recall``, ``f1``, and ``accuracy`` keys.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=2)

    true_predictions: list[list[str]] = []
    true_labels: list[list[str]] = []

    for prediction_seq, label_seq in zip(predictions, labels):
        filtered_preds: list[str] = []
        filtered_labels: list[str] = []
        for prediction, label in zip(prediction_seq, label_seq):
            if label != -100:
                filtered_preds.append(ID2LABEL[prediction])
                filtered_labels.append(ID2LABEL[label])
        true_predictions.append(filtered_preds)
        true_labels.append(filtered_labels)

    results = _seqeval.compute(
        predictions=true_predictions,
        references=true_labels,
        zero_division=0,
    )
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_training_args(total_steps: int) -> TrainingArguments:
    """Construct :class:`~transformers.TrainingArguments` for the run.

    Args:
        total_steps: Total number of optimizer steps (used to derive warmup).

    Returns:
        Configured :class:`~transformers.TrainingArguments` instance.
    """
    warmup_steps = int(WARMUP_FRACTION * total_steps)
    return TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=LOGGING_STEPS,
        seed=SEED,
        report_to="none",
        fp16=USE_FP16,
    )


def main() -> None:
    """Execute the BERT fine-tuning pipeline."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 60)
    logger.info("PHASE 2 - BERT FINE-TUNING")
    logger.info("=" * 60)
    logger.info("Device: %s | fp16: %s", device_name, USE_FP16)

    logger.info("Loading %s ...", AUG_TRAIN_FILE)
    records = load_jsonl(AUG_TRAIN_FILE)
    logger.info("%d records loaded", len(records))

    entity_presence = [
        int(any(tag != "O" for tag in record["ner_tags"])) for record in records
    ]
    train_records, val_records = train_test_split(
        records,
        test_size=VALIDATION_SPLIT,
        random_state=SEED,
        stratify=entity_presence,
    )
    logger.info("Train: %d | Val: %d", len(train_records), len(val_records))

    logger.info("Loading tokenizer: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    logger.info("Tokenizing and aligning labels...")
    train_features = tokenize_and_align_labels(train_records, tokenizer)
    val_features = tokenize_and_align_labels(val_records, tokenizer)
    train_dataset = NERDataset(train_features)
    val_dataset = NERDataset(val_features)
    logger.info(
        "Train: %d | Val: %d tokenized examples",
        len(train_dataset),
        len(val_dataset),
    )

    logger.info("Loading model: %s", MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    total_steps = (len(train_dataset) // BATCH_SIZE) * NUM_EPOCHS
    training_args = build_training_args(total_steps)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info(
        "Training complete - runtime: %.1fs, loss: %.4f",
        train_result.metrics["train_runtime"],
        train_result.metrics["train_loss"],
    )

    best_model_dir = OUTPUT_DIR / "best_model"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    logger.info("Best model saved to %s", best_model_dir)

    val_metrics = trainer.evaluate()
    logger.info("Validation metrics (best checkpoint):")
    for key, value in val_metrics.items():
        if "runtime" not in key:
            logger.info(
                "  %s: %s", key, f"{value:.4f}" if isinstance(value, float) else value
            )

    logger.info("Phase 2 complete.")


if __name__ == "__main__":
    main()