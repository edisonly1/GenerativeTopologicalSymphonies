"""Local fluency metrics such as jump distance and corpus transition perplexity."""

from __future__ import annotations

import math
from dataclasses import dataclass

from preprocessing import QuantizedPiece


@dataclass(slots=True)
class FluencyMetrics:
    """Symbolic local-motion summary for one piece."""

    transition_count: int
    mean_pitch_jump: float
    median_pitch_jump: float
    large_jump_rate: float
    stepwise_motion_ratio: float
    repeated_pitch_ratio: float
    transition_perplexity: float | None


def _ordered_pitches(piece: QuantizedPiece) -> list[int]:
    """Return pitches ordered by symbolic onset."""
    ordered_events = sorted(
        piece.note_events,
        key=lambda event: (event.start_step, event.pitch, event.instrument, event.track_index),
    )
    return [event.pitch for event in ordered_events]


def _absolute_intervals(piece: QuantizedPiece) -> list[int]:
    """Return absolute pitch jumps between consecutive note onsets."""
    pitches = _ordered_pitches(piece)
    if len(pitches) <= 1:
        return []
    return [abs(current - previous) for previous, current in zip(pitches, pitches[1:])]


def build_interval_language_model(
    pieces: list[QuantizedPiece],
    *,
    max_interval: int = 24,
) -> dict[int, float]:
    """Fit a Laplace-smoothed interval model from reference pieces."""
    counts = {interval: 1.0 for interval in range(max_interval + 1)}
    counts[max_interval + 1] = 1.0
    for piece in pieces:
        for interval in _absolute_intervals(piece):
            bucket = min(interval, max_interval + 1)
            counts[bucket] += 1.0
    total = sum(counts.values())
    return {
        interval: count / total
        for interval, count in counts.items()
    }


def score_transition_perplexity(
    piece: QuantizedPiece,
    interval_model: dict[int, float],
    *,
    max_interval: int = 24,
) -> float:
    """Score a piece against a human-reference interval model."""
    intervals = _absolute_intervals(piece)
    if not intervals:
        return 1.0
    negative_log_probability = 0.0
    for interval in intervals:
        bucket = min(interval, max_interval + 1)
        probability = max(interval_model.get(bucket, 1e-9), 1e-9)
        negative_log_probability += -math.log(probability)
    return math.exp(negative_log_probability / len(intervals))


def score_fluency(
    piece: QuantizedPiece,
    *,
    interval_model: dict[int, float] | None = None,
    large_jump_threshold: int = 12,
) -> FluencyMetrics:
    """Measure local symbolic motion smoothness."""
    intervals = _absolute_intervals(piece)
    if not intervals:
        return FluencyMetrics(
            transition_count=0,
            mean_pitch_jump=0.0,
            median_pitch_jump=0.0,
            large_jump_rate=0.0,
            stepwise_motion_ratio=0.0,
            repeated_pitch_ratio=0.0,
            transition_perplexity=1.0 if interval_model is not None else None,
        )

    sorted_intervals = sorted(intervals)
    midpoint = len(sorted_intervals) // 2
    median = (
        float(sorted_intervals[midpoint])
        if len(sorted_intervals) % 2 == 1
        else 0.5 * (sorted_intervals[midpoint - 1] + sorted_intervals[midpoint])
    )
    stepwise = sum(1 for interval in intervals if interval <= 2)
    repeated = sum(1 for interval in intervals if interval == 0)
    large_jumps = sum(1 for interval in intervals if interval >= large_jump_threshold)
    perplexity = None
    if interval_model is not None:
        perplexity = score_transition_perplexity(piece, interval_model)
    return FluencyMetrics(
        transition_count=len(intervals),
        mean_pitch_jump=sum(intervals) / len(intervals),
        median_pitch_jump=median,
        large_jump_rate=large_jumps / len(intervals),
        stepwise_motion_ratio=stepwise / len(intervals),
        repeated_pitch_ratio=repeated / len(intervals),
        transition_perplexity=perplexity,
    )
