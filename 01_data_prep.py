"""Data preparation pipeline for PII NER training.

Loads raw train/test JSON datasets, augments them with synthetic email
entities anchored to person-name spans, validates token/tag alignment,
and writes augmented datasets to disk as JSONL.
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter
from pathlib import Path

from faker import Faker

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

DATA_DIR: Path = Path("data")
TRAIN_FILE: Path = DATA_DIR / "data.json"
TEST_FILE: Path = DATA_DIR / "test_data.json"
AUG_TRAIN_FILE: Path = DATA_DIR / "train_augmented.json"
AUG_TEST_FILE: Path = DATA_DIR / "test_augmented.json"

# Fraction of PER-containing sentences that receive a synthetic email injection.
TRAIN_INJECT_RATE: float = 0.18
TEST_INJECT_RATE: float = 0.18

VALID_TAGS: frozenset[str] = frozenset({"O", "B-PER", "I-PER", "B-EMAIL", "I-EMAIL"})

EMAIL_DOMAINS: tuple[str, ...] = (
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "protonmail.com",
    "icloud.com",
    "company.org",
    "work.net",
    "mail.com",
    "live.com",
)

_faker = Faker()
Faker.seed(SEED)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_dataset(path: Path) -> list[dict]:
    """Load a dataset from either JSON array or JSONL format.

    Args:
        path: Filesystem path to the dataset file.

    Returns:
        List of record dicts, each with ``tokens`` and ``ner_tags`` keys.

    Raises:
        FileNotFoundError: If *path* does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
    """
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def save_jsonl(records: list[dict], path: Path) -> None:
    """Serialize *records* to a JSONL file, one record per line.

    Args:
        records: Sequence of serializable dicts.
        path: Destination file path. Parent directories must exist.
    """
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def compute_tag_distribution(records: list[dict]) -> Counter:
    """Count occurrences of each NER tag across all records.

    Args:
        records: Dataset records with ``ner_tags`` lists.

    Returns:
        A :class:`~collections.Counter` mapping tag string to token count.
    """
    distribution: Counter = Counter()
    for record in records:
        distribution.update(record["ner_tags"])
    return distribution


def log_distribution(label: str, records: list[dict]) -> None:
    """Log a tag-frequency summary for the given dataset split.

    Args:
        label: Human-readable split name (e.g. ``"Train (original)"`` ).
        records: Dataset records to summarise.
    """
    distribution = compute_tag_distribution(records)
    total_tokens = sum(distribution.values())
    logger.info("%s — %d records, %d tokens", label, len(records), total_tokens)
    for tag in ("O", "B-PER", "I-PER", "B-EMAIL", "I-EMAIL"):
        logger.info("  %-12s: %7d", tag, distribution.get(tag, 0))


# ---------------------------------------------------------------------------
# Email injection
# ---------------------------------------------------------------------------


def extract_per_spans(tags: list[str]) -> list[tuple[int, int]]:
    """Return the (start, end) inclusive index spans of all PER entities.

    Args:
        tags: BIO tag sequence for a single record.

    Returns:
        List of ``(start_index, end_index)`` tuples (both inclusive).
    """
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(tags):
        if tags[index] == "B-PER":
            end = index + 1
            while end < len(tags) and tags[end] == "I-PER":
                end += 1
            spans.append((index, end - 1))
            index = end
        else:
            index += 1
    return spans


def name_tokens_to_email(name_tokens: list[str]) -> str:
    """Derive a plausible email address from person-name tokens.

    Applies one of several address formats chosen uniformly at random.
    Falls back to a Faker-generated address when no valid alphabetic
    characters can be extracted from *name_tokens*.

    Args:
        name_tokens: Word tokens comprising a person's name.

    Returns:
        A syntactically valid email address string.
    """
    parts = [re.sub(r"[^a-zA-Z]", "", token).lower() for token in name_tokens]
    parts = [part for part in parts if part]
    domain = random.choice(EMAIL_DOMAINS)

    if not parts:
        return _faker.email()

    variant = random.randint(0, 5)
    has_multiple = len(parts) >= 2

    if variant == 0 and has_multiple:
        return f"{parts[0]}.{parts[-1]}@{domain}"
    if variant == 1 and has_multiple:
        return f"{parts[0][0]}.{parts[-1]}@{domain}"
    if variant == 2 and has_multiple:
        return f"{parts[-1]}.{parts[0]}@{domain}"
    if variant == 3:
        return f"{parts[0]}{random.randint(1, 99)}@{domain}"
    if variant == 4 and has_multiple:
        return f"{''.join(part[0] for part in parts)}@{domain}"
    return f"{''.join(parts)}@{domain}"


def inject_email_into_record(record: dict) -> dict:
    """Insert one synthetic email address after a randomly chosen PER span.

    The email is surrounded by parentheses: ``PersonName ( email@domain ) ...``.
    The original record is never mutated.

    Args:
        record: A dataset record with ``tokens``, ``ner_tags``, and ``lang`` keys.

    Returns:
        A new record dict with the email inserted, including an updated
        ``sequence`` field. Returns *record* unchanged if no PER entity exists.
    """
    tokens = list(record["tokens"])
    tags = list(record["ner_tags"])

    spans = extract_per_spans(tags)
    if not spans:
        return record

    start, end = random.choice(spans)
    email_address = name_tokens_to_email(tokens[start : end + 1])
    insert_position = end + 1

    new_tokens = tokens[:insert_position] + ["(", email_address, ")"] + tokens[insert_position:]
    new_tags = tags[:insert_position] + ["O", "B-EMAIL", "O"] + tags[insert_position:]

    return {
        "tokens": new_tokens,
        "ner_tags": new_tags,
        "lang": record["lang"],
        "sequence": " ".join(new_tokens),
    }


def augment_with_emails(
    records: list[dict],
    inject_rate: float,
    split_label: str,
) -> list[dict]:
    """Inject synthetic email entities into a fraction of PER-containing records.

    Args:
        records: Source dataset records.
        inject_rate: Probability of injection for each PER-containing record.
        split_label: Name used in log output (e.g. ``"train"`` ).

    Returns:
        Augmented dataset with the same length as *records*.
    """
    augmented: list[dict] = []
    injected_count = 0

    for record in records:
        has_per_entity = any(tag in ("B-PER", "I-PER") for tag in record["ner_tags"])

        if has_per_entity and random.random() < inject_rate:
            candidate = inject_email_into_record(record)
            if any(tag == "B-EMAIL" for tag in candidate["ner_tags"]):
                augmented.append(candidate)
                injected_count += 1
                continue

        augmented.append(record)

    injection_pct = injected_count / len(records) * 100
    logger.info(
        "[%s] Email injected into %d records (%.1f%%)",
        split_label,
        injected_count,
        injection_pct,
    )
    return augmented


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_records(records: list[dict], split_label: str) -> None:
    """Assert structural correctness of all records in a dataset split.

    Args:
        records: Dataset records to validate.
        split_label: Name used in error messages and logs.

    Raises:
        ValueError: If any record has a token/tag length mismatch or contains
            unrecognised NER tags.
    """
    errors: list[str] = []

    for index, record in enumerate(records):
        if len(record["tokens"]) != len(record["ner_tags"]):
            errors.append(
                f"Record {index}: token count ({len(record['tokens'])}) "
                f"!= tag count ({len(record['ner_tags'])})"
            )
        unknown_tags = set(record["ner_tags"]) - VALID_TAGS
        if unknown_tags:
            errors.append(f"Record {index}: unrecognised tags {unknown_tags}")

    if errors:
        for error in errors[:10]:
            logger.error(error)
        raise ValueError(
            f"Validation failed for '{split_label}' — "
            f"{len(errors)} error(s) detected. See logs for details."
        )

    logger.info("[%s] Validation passed — %d records OK", split_label, len(records))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Execute the full data preparation pipeline."""
    random.seed(SEED)

    logger.info("=" * 60)
    logger.info("PHASE 1 — DATA PREPARATION")
    logger.info("=" * 60)

    logger.info("Loading raw datasets...")
    train_records = load_dataset(TRAIN_FILE)
    test_records = load_dataset(TEST_FILE)
    logger.info("Train: %d records | Test: %d records", len(train_records), len(test_records))

    log_distribution("Train (original)", train_records)
    log_distribution("Test  (original)", test_records)

    logger.info("Injecting synthetic email entities...")
    aug_train = augment_with_emails(train_records, TRAIN_INJECT_RATE, "train")
    aug_test = augment_with_emails(test_records, TEST_INJECT_RATE, "test")

    log_distribution("Train (augmented)", aug_train)
    log_distribution("Test  (augmented)", aug_test)

    logger.info("Validating augmented datasets...")
    validate_records(aug_train, "train")
    validate_records(aug_test, "test")

    sample = next(r for r in aug_train if "B-EMAIL" in r["ner_tags"])
    logger.info("Sample injected record: %s", sample["sequence"][:120])
    for token, tag in zip(sample["tokens"], sample["ner_tags"]):
        if tag != "O":
            logger.info("  %-30s  %s", token, tag)

    save_jsonl(aug_train, AUG_TRAIN_FILE)
    save_jsonl(aug_test, AUG_TEST_FILE)
    logger.info("Saved augmented datasets to %s and %s", AUG_TRAIN_FILE, AUG_TEST_FILE)
    logger.info("Phase 1 complete.")


if __name__ == "__main__":
    main()