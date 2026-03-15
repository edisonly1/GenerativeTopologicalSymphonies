"""Phrase segmentation helpers."""

from __future__ import annotations

from dataclasses import replace

from .schema import QuantizedPiece


def _parse_strategy(strategy: str) -> tuple[str, int]:
    """Resolve the phrase segmentation strategy family and target bar span."""
    if strategy == "single_phrase":
        return strategy, 0
    if strategy.startswith("bars_"):
        return "bars", int(strategy.split("_", maxsplit=1)[1])
    if strategy.startswith("cadence_bars_"):
        return "cadence_bars", int(strategy.rsplit("_", maxsplit=1)[1])
    raise ValueError(
        "Unsupported phrase strategy. Use 'single_phrase', 'bars_<n>', or 'cadence_bars_<n>'."
    )


def _rest_boundary_starts(piece: QuantizedPiece, *, min_gap_bars: float = 0.75) -> list[int]:
    """Detect likely phrase starts after substantial silent gaps."""
    if len(piece.note_events) < 2:
        return []
    min_gap_steps = max(1, int(round(piece.bar_steps * min_gap_bars)))
    ordered_events = sorted(piece.note_events, key=lambda event: (event.start_step, event.pitch))
    starts: set[int] = set()
    previous_end = ordered_events[0].start_step + ordered_events[0].duration_steps
    for event in ordered_events[1:]:
        if event.start_step - previous_end >= min_gap_steps:
            starts.add(event.bar)
        previous_end = max(previous_end, event.start_step + event.duration_steps)
    return sorted(start for start in starts if start > 1)


def _cadence_boundary_starts(piece: QuantizedPiece) -> list[int]:
    """Convert cadence bars into the next phrase start positions."""
    cadence_bars = piece.metadata.get("cadence_bars", [])
    return sorted(
        {
            min(piece.total_bars, int(bar) + 1)
            for bar in cadence_bars
            if 1 <= int(bar) < piece.total_bars
        }
    )


def _adaptive_boundaries(piece: QuantizedPiece, *, max_bars: int) -> list[int]:
    """Choose phrase starts near cadences or rests while capping phrase length."""
    if max_bars <= 0:
        return [1]
    candidate_starts = sorted(
        set(_cadence_boundary_starts(piece)) | set(_rest_boundary_starts(piece))
    )
    boundaries = [1]
    current_start = 1
    while current_start <= piece.total_bars:
        fallback_start = current_start + max_bars
        if fallback_start > piece.total_bars:
            if piece.total_bars not in boundaries:
                boundaries.append(piece.total_bars)
            break
        eligible = [
            start
            for start in candidate_starts
            if current_start + 2 <= start <= fallback_start
        ]
        next_start = eligible[-1] if eligible else fallback_start
        if next_start not in boundaries:
            boundaries.append(next_start)
        current_start = next_start
    return sorted(set(boundaries))


def segment_phrases(piece: QuantizedPiece, strategy: str = "bars_4") -> QuantizedPiece:
    """Annotate phrase boundaries using a simple bar-based policy."""
    strategy_name, bars_per_phrase = _parse_strategy(strategy)
    if bars_per_phrase == 0:
        boundaries = [1]
    elif strategy_name == "cadence_bars":
        boundaries = _adaptive_boundaries(piece, max_bars=bars_per_phrase)
    else:
        boundaries = list(range(1, piece.total_bars + 1, bars_per_phrase))
    return replace(piece, phrase_boundaries=boundaries)
