"""Rule-based playability checks for piano-like symbolic pieces."""

from __future__ import annotations

from dataclasses import dataclass

from preprocessing import QuantizedPiece


@dataclass(slots=True)
class PlayabilityMetrics:
    """Simple piano-oriented playability summary."""

    polyphony_peak: int
    note_density_per_bar: float
    mean_simultaneous_span: float
    max_simultaneous_span: float
    large_span_rate: float
    overlap_violation_rate: float


def _group_by_start_step(piece: QuantizedPiece) -> list[list[int]]:
    """Group note pitches by onset step."""
    groups: dict[int, list[int]] = {}
    for event in piece.note_events:
        groups.setdefault(event.start_step, []).append(event.pitch)
    return [groups[step] for step in sorted(groups)]


def _simultaneous_spans(piece: QuantizedPiece) -> list[int]:
    """Compute pitch spans for simultaneous note clusters."""
    spans: list[int] = []
    for pitches in _group_by_start_step(piece):
        if len(pitches) <= 1:
            continue
        spans.append(max(pitches) - min(pitches))
    return spans


def _count_overlap_violations(piece: QuantizedPiece) -> int:
    """Count same-pitch overlaps that are awkward on piano."""
    active_until: dict[tuple[int, int], int] = {}
    violations = 0
    for event in sorted(piece.note_events, key=lambda item: (item.start_step, item.pitch)):
        key = (event.instrument, event.pitch)
        previous_end = active_until.get(key, -1)
        if event.start_step < previous_end:
            violations += 1
        active_until[key] = max(previous_end, event.start_step + event.duration_steps)
    return violations


def score_playability(
    piece: QuantizedPiece,
    *,
    large_span_threshold: int = 24,
) -> PlayabilityMetrics:
    """Measure simple symbolic constraints related to piano playability."""
    spans = _simultaneous_spans(piece)
    overlap_violations = _count_overlap_violations(piece)
    simultaneous_groups = _group_by_start_step(piece)
    onset_count = max(1, len(simultaneous_groups))
    polyphony_peak = max((len(group) for group in simultaneous_groups), default=0)
    return PlayabilityMetrics(
        polyphony_peak=polyphony_peak,
        note_density_per_bar=len(piece.note_events) / max(piece.total_bars, 1),
        mean_simultaneous_span=(sum(spans) / len(spans)) if spans else 0.0,
        max_simultaneous_span=max(spans, default=0),
        large_span_rate=sum(1 for span in spans if span > large_span_threshold) / max(len(spans), 1),
        overlap_violation_rate=overlap_violations / onset_count,
    )

