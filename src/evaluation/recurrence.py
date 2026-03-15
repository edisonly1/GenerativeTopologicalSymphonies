"""Self-similarity and recurrence analyses."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tokenization import PieceExample


@dataclass(slots=True)
class RecurrenceMetrics:
    """Phrase-level recurrence summary for one piece."""

    phrase_count: int
    recurrent_phrase_ratio: float
    mean_max_similarity: float
    max_similarity: float


def _pitch_class_histogram(blocks) -> list[float]:
    """Build a duration-weighted pitch-class histogram for a phrase."""
    histogram = [0.0] * 12
    for block in blocks:
        histogram[block.pitch % 12] += float(block.duration + 1)
    total = sum(histogram)
    if total <= 0.0:
        return histogram
    return [value / total for value in histogram]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity between two histograms."""
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _group_phrase_blocks(example: PieceExample) -> list[list]:
    """Collect event blocks by phrase index."""
    phrase_map: dict[int, list] = {}
    for block in example.event_blocks:
        phrase_map.setdefault(block.phrase_index, []).append(block)
    return [phrase_map[index] for index in sorted(phrase_map)]


def score_recurrence(example: PieceExample, *, threshold: float = 0.85) -> RecurrenceMetrics:
    """Measure how often phrases resemble an earlier phrase."""
    phrase_blocks = _group_phrase_blocks(example)
    phrase_count = len(phrase_blocks)
    if phrase_count <= 1:
        return RecurrenceMetrics(
            phrase_count=phrase_count,
            recurrent_phrase_ratio=0.0,
            mean_max_similarity=0.0,
            max_similarity=0.0,
        )

    histograms = [_pitch_class_histogram(blocks) for blocks in phrase_blocks]
    max_similarities: list[float] = []
    recurrent_count = 0
    for phrase_index, histogram in enumerate(histograms):
        previous_histograms = histograms[:phrase_index]
        if not previous_histograms:
            max_similarities.append(0.0)
            continue
        best_similarity = max(_cosine_similarity(histogram, previous) for previous in previous_histograms)
        max_similarities.append(best_similarity)
        if best_similarity >= threshold:
            recurrent_count += 1

    compared_phrases = max(1, phrase_count - 1)
    return RecurrenceMetrics(
        phrase_count=phrase_count,
        recurrent_phrase_ratio=recurrent_count / compared_phrases,
        mean_max_similarity=sum(max_similarities[1:]) / compared_phrases,
        max_similarity=max(max_similarities[1:], default=0.0),
    )
