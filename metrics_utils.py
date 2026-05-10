"""Shared token-level NER metrics utilities used across evaluation phases.

Provides entity-type-aware confusion matrix accumulation and a standard
console metrics table, eliminating duplication between Phases 3 and 4.
"""

from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)

ENTITY_TYPES: tuple[str, ...] = ("PER", "EMAIL")


def _entity_class(tag: str) -> str:
    """Strip the BIO prefix and return the entity class.

    Args:
        tag: A BIO NER tag such as ``"B-PER"``, ``"I-EMAIL"``, or ``"O"``.

    Returns:
        The entity class string (e.g. ``"PER"``, ``"EMAIL"``) or ``"O"``.
    """
    return "O" if tag == "O" else tag.split("-", 1)[1]


def compute_entity_metrics(
    true_sequences: Sequence[Sequence[str]],
    pred_sequences: Sequence[Sequence[str]],
) -> dict[str, dict[str, float | int]]:
    """Compute token-level precision, recall, F1, FPR, FNR, and accuracy.

    Collapses B-/I- prefixes into entity classes for per-class counting.
    OVERALL counts are computed independently of per-type counts to avoid
    double-counting tokens that span class boundaries.

    Args:
        true_sequences: Ground-truth BIO label sequences.
        pred_sequences: Predicted BIO label sequences.

    Returns:
        Nested dict keyed by entity type (``"PER"``, ``"EMAIL"``, ``"OVERALL"``),
        each containing ``TP``, ``FP``, ``FN``, ``TN``, ``Precision``, ``Recall``,
        ``F1``, ``FPR``, ``FNR``, and ``Accuracy``.
    """
    stats: dict[str, dict[str, int]] = {
        entity_type: {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        for entity_type in ENTITY_TYPES
    }
    stats["OVERALL"] = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

    for true_seq, pred_seq in zip(true_sequences, pred_sequences):
        for true_tag, pred_tag in zip(true_seq, pred_seq):
            true_class = _entity_class(true_tag)
            pred_class = _entity_class(pred_tag)

            for entity_type in ENTITY_TYPES:
                is_true_class = true_class == entity_type
                is_pred_class = pred_class == entity_type

                if is_true_class and is_pred_class:
                    stats[entity_type]["TP"] += 1
                elif not is_true_class and is_pred_class:
                    stats[entity_type]["FP"] += 1
                elif is_true_class and not is_pred_class:
                    stats[entity_type]["FN"] += 1
                else:
                    stats[entity_type]["TN"] += 1

            # OVERALL: computed outside the per-type loop to avoid double-counting.
            if true_class == "O" and pred_class == "O":
                stats["OVERALL"]["TN"] += 1
            elif true_class != "O" and pred_class != "O" and true_class == pred_class:
                stats["OVERALL"]["TP"] += 1
            elif true_class == "O" and pred_class != "O":
                stats["OVERALL"]["FP"] += 1
            elif true_class != "O" and pred_class == "O":
                stats["OVERALL"]["FN"] += 1

    results: dict[str, dict[str, float | int]] = {}

    for entity_type, counts in stats.items():
        tp = counts["TP"]
        fp = counts["FP"]
        fn = counts["FN"]
        tn = counts["TN"]
        total = tp + fp + fn + tn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0

        results[entity_type] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "FPR": round(fpr, 4),
            "FNR": round(fnr, 4),
            "Accuracy": round(accuracy, 4),
        }

    return results


def print_metrics_table(metrics: dict[str, dict[str, float | int]]) -> None:
    """Print a formatted metrics table to stdout.

    Args:
        metrics: Output from :func:`compute_entity_metrics`.
    """
    header = (
        f"{'Entity':10s} {'Prec':>7} {'Recall':>7} {'F1':>7} "
        f"{'FPR':>7} {'FNR':>7} {'Acc':>7}"
    )
    print("\n" + header)
    print("-" * len(header))
    for entity_type in ("PER", "EMAIL", "OVERALL"):
        m = metrics[entity_type]
        print(
            f"{entity_type:10s} "
            f"{m['Precision']:7.4f} "
            f"{m['Recall']:7.4f} "
            f"{m['F1']:7.4f} "
            f"{m['FPR']:7.4f} "
            f"{m['FNR']:7.4f} "
            f"{m['Accuracy']:7.4f}"
        )
