"""Tonal-divergence and pitch-class analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tokenization import PieceExample


@dataclass(slots=True)
class TonalMetrics:
    """Pitch-class statistics for one symbolic piece."""

    pitch_class_entropy: float
    tonal_center_strength: float
    pitch_class_divergence: float | None
    dominant_pitch_class: int


def _pitch_class_histogram(example: PieceExample) -> list[float]:
    """Build a duration-weighted pitch-class histogram."""
    histogram = [0.0] * 12
    for block in example.event_blocks:
        histogram[block.pitch % 12] += float(block.duration + 1)
    total = sum(histogram)
    if total <= 0.0:
        return histogram
    return [value / total for value in histogram]


def _entropy(probabilities: list[float]) -> float:
    """Compute Shannon entropy in bits."""
    return -sum(value * math.log2(value) for value in probabilities if value > 0.0)


def _jensen_shannon(left: list[float], right: list[float]) -> float:
    """Compute Jensen-Shannon divergence between two distributions."""
    midpoint = [(a + b) * 0.5 for a, b in zip(left, right, strict=True)]
    return 0.5 * (_kl_divergence(left, midpoint) + _kl_divergence(right, midpoint))


def _kl_divergence(left: list[float], right: list[float]) -> float:
    """Compute KL divergence with zero-safe handling."""
    divergence = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        if left_value <= 0.0 or right_value <= 0.0:
            continue
        divergence += left_value * math.log2(left_value / right_value)
    return divergence


def score_tonal_alignment(
    example: PieceExample,
    *,
    reference_example: PieceExample | None = None,
) -> TonalMetrics:
    """Score tonal concentration and optional divergence to a reference piece."""
    histogram = _pitch_class_histogram(example)
    divergence = None
    if reference_example is not None:
        reference_histogram = _pitch_class_histogram(reference_example)
        divergence = _jensen_shannon(histogram, reference_histogram)
    dominant_pitch_class = max(range(12), key=histogram.__getitem__) if histogram else 0
    return TonalMetrics(
        pitch_class_entropy=_entropy(histogram),
        tonal_center_strength=max(histogram, default=0.0),
        pitch_class_divergence=divergence,
        dominant_pitch_class=dominant_pitch_class,
    )


score_tonal_metrics = score_tonal_alignment

