"""Zero-shot PII detection using LLaMA 3.1-8B-Instant via the Groq API.

Evaluates a large language model on the same test sample used by the BERT
baseline to enable a controlled comparison.  No fine-tuning is applied;
detection relies entirely on prompt engineering and JSON-mode output.

Results are saved to ``results/llm_metrics.json``.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from pathlib import Path

from groq import Groq

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

DATA_DIR: Path = Path("data")
RESULTS_DIR: Path = Path("results")
TEST_FILE: Path = DATA_DIR / "test_augmented.json"

# llama-3.2-1b-preview was decommissioned by Groq; 8b-instant is the recommended replacement.
GROQ_MODEL: str = "llama-3.1-8b-instant"
SAMPLE_SIZE: int = 300
RETRY_DELAY_SECONDS: int = 2
MAX_RETRIES: int = 3
ENTITY_SAMPLE_FRACTION: float = 0.6

VALID_ENTITY_TYPES: frozenset[str] = frozenset({"PER", "EMAIL"})

_SYSTEM_PROMPT: str = """\
You are a PII detection system. Your only job is to identify personal names \
and email addresses in text.

Output ONLY a valid JSON object - no explanation, no markdown, no extra text.

Format:
{
  "entities": [
    {"text": "<entity text>", "type": "PER"},
    {"text": "<entity text>", "type": "EMAIL"}
  ]
}

Rules:
- type must be exactly "PER" or "EMAIL"
- Include every person name and every email address found
- If none found, return {"entities": []}
- Do not include organizations, locations, or other entities"""

LLM_CHALLENGES: dict[str, str] = {
    "format_failures": (
        "LLaMA 3.2-1B occasionally generates malformed JSON "
        "requiring fallback regex extraction."
    ),
    "hallucinations": (
        "Model may invent entity boundaries for common nouns "
        "resembling names (e.g., 'Machine' in band names)."
    ),
    "boundary_errors": (
        "Multi-token names frequently truncated or merged; "
        "email localpart/domain sometimes split incorrectly."
    ),
    "no_context": (
        "Without fine-tuning, model lacks domain-specific cues "
        "and relies purely on surface patterns."
    ),
    "rate_limits": (
        "Free tier rate limits require retry logic and reduce "
        "throughput to ~20-30 requests/minute."
    ),
}


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
# Prompting
# ---------------------------------------------------------------------------


def build_user_prompt(sentence: str) -> str:
    """Format the per-sentence user turn for entity extraction.

    Args:
        sentence: Raw text sentence to analyse.

    Returns:
        Prompt string ready to send as the ``user`` role message.
    """
    return f'Identify all PER (person names) and EMAIL entities in this text:\n\n"{sentence}"'


def call_groq_api(client: Groq, sentence: str) -> list[dict[str, str]]:
    """Query the Groq API for PII entities in *sentence*.

    Implements linear-backoff retry on rate-limit errors.  Falls back to
    regex-based JSON extraction on parse failures and returns an empty list
    when all retries are exhausted.

    Args:
        client: Authenticated :class:`groq.Groq` client instance.
        sentence: Input text to analyse.

    Returns:
        List of entity dicts, each with ``text`` and ``type`` keys.
    """
    raw_content = ""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(sentence)},
                ],
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw_content = response.choices[0].message.content.strip()
            parsed = json.loads(raw_content)
            return parsed.get("entities", [])

        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group()).get("entities", [])
                except json.JSONDecodeError:
                    pass
            return []

        except Exception as exc:
            if "rate" in str(exc).lower() and attempt < MAX_RETRIES - 1:
                wait_seconds = RETRY_DELAY_SECONDS * (attempt + 1)
                logger.warning("Rate limit hit - retrying in %ds ...", wait_seconds)
                time.sleep(wait_seconds)
                continue
            logger.warning("API error (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, exc)
            return []

    return []


# ---------------------------------------------------------------------------
# Label alignment
# ---------------------------------------------------------------------------


def align_entities_to_bio_labels(
    tokens: list[str],
    entities: list[dict[str, str]],
) -> list[str]:
    """Map extracted entity spans back to per-token BIO labels.

    Uses a case-insensitive sliding-window search to locate each entity's
    token span.  Only the first occurrence is tagged.

    Args:
        tokens: Word tokens of the source sentence.
        entities: Entity dicts with ``text`` and ``type`` keys.

    Returns:
        BIO label list aligned to *tokens*, with ``"O"`` for non-entity tokens.
    """
    labels = ["O"] * len(tokens)

    for entity in entities:
        entity_type = entity.get("type", "").upper()
        if entity_type not in VALID_ENTITY_TYPES:
            continue

        begin_tag = f"B-{entity_type}"
        inside_tag = f"I-{entity_type}"
        entity_tokens = entity.get("text", "").split()
        if not entity_tokens:
            continue

        span_length = len(entity_tokens)
        lower_entity = [token.lower() for token in entity_tokens]

        for start_index in range(len(tokens) - span_length + 1):
            window = [token.lower() for token in tokens[start_index : start_index + span_length]]
            if window == lower_entity:
                labels[start_index] = begin_tag
                for offset in range(1, span_length):
                    labels[start_index + offset] = inside_tag
                break

    return labels


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def build_stratified_sample(
    records: list[dict],
    sample_size: int,
    entity_fraction: float,
) -> list[dict]:
    """Draw a stratified sample balanced between entity and non-entity records.

    Args:
        records: Full test dataset.
        sample_size: Target number of records to sample.
        entity_fraction: Fraction of *sample_size* to fill with entity records.

    Returns:
        Shuffled list of sampled records.
    """
    entity_records = [r for r in records if any(tag != "O" for tag in r["ner_tags"])]
    empty_records = [r for r in records if all(tag == "O" for tag in r["ner_tags"])]

    n_entity = min(int(sample_size * entity_fraction), len(entity_records))
    n_empty = min(sample_size - n_entity, len(empty_records))
    sample = random.sample(entity_records, n_entity) + random.sample(empty_records, n_empty)
    random.shuffle(sample)
    return sample


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Execute zero-shot LLM PII detection and evaluation."""
    random.seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("PHASE 4 - ZERO-SHOT LLM EVALUATION (%s)", GROQ_MODEL)
    logger.info("=" * 60)

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        api_key = input("\nEnter your Groq API key: ").strip()
    client = Groq(api_key=api_key)
    logger.info("Model: %s", GROQ_MODEL)

    logger.info("Loading test data from %s ...", TEST_FILE)
    test_records = load_jsonl(TEST_FILE)

    sample = build_stratified_sample(test_records, SAMPLE_SIZE, ENTITY_SAMPLE_FRACTION)
    entity_count = sum(1 for r in sample if any(tag != "O" for tag in r["ner_tags"]))
    logger.info(
        "Sampled %d records (%d with entities, %d without)",
        len(sample),
        entity_count,
        len(sample) - entity_count,
    )

    logger.info("Running zero-shot inference...")
    true_sequences: list[list[str]] = []
    pred_sequences: list[list[str]] = []
    missed_entity_count = 0

    for index, record in enumerate(sample):
        if index % 50 == 0:
            logger.info("[%d/%d] ...", index, len(sample))

        true_labels = record["ner_tags"]
        entities = call_groq_api(client, record["sequence"])
        pred_labels = align_entities_to_bio_labels(record["tokens"], entities)

        alignment_length = min(len(true_labels), len(pred_labels))
        true_sequences.append(true_labels[:alignment_length])
        pred_sequences.append(pred_labels[:alignment_length])

        if not entities and any(tag != "O" for tag in true_labels):
            missed_entity_count += 1

    logger.info(
        "Done. Records with entities where LLM returned nothing: %d/%d",
        missed_entity_count,
        len(sample),
    )

    logger.info("Computing metrics...")
    metrics = compute_entity_metrics(true_sequences, pred_sequences)
    print_metrics_table(metrics)

    output = {
        "model": GROQ_MODEL,
        "sample_size": len(sample),
        "metrics": metrics,
        "challenges": LLM_CHALLENGES,
    }
    output_path = RESULTS_DIR / "llm_metrics.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    logger.info("Results saved to %s", output_path)

    logger.info("Phase 4 complete.")


if __name__ == "__main__":
    main()