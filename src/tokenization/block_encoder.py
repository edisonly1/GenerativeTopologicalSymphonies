"""Conversion between normalized symbolic data and grouped event blocks."""

from __future__ import annotations

from bisect import bisect_right

from preprocessing.harmony_extract import GLOBAL_HARMONY_VOCAB
from preprocessing.schema import QuantizedPiece

from .dataset import EventBlock, PieceExample


PHRASE_FLAG_MAP = {
    "mid": 0,
    "start": 1,
    "end": 2,
    "start_end": 3,
}


def _bucket_duration(duration_steps: int, duration_bins: int) -> int:
    """Map a quantized duration to a bounded token bucket."""
    if duration_bins < 1:
        raise ValueError("duration_bins must be >= 1.")
    return min(duration_bins - 1, max(0, duration_steps - 1))


def _bucket_velocity(velocity: int, velocity_bins: int) -> int:
    """Map MIDI velocity into a fixed number of bins."""
    if velocity_bins < 1:
        raise ValueError("velocity_bins must be >= 1.")
    clamped = min(127, max(0, velocity))
    return min(velocity_bins - 1, (clamped * velocity_bins) // 128)


def _bucket_bar_position(position: int, *, bar_steps: int, bar_position_bins: int) -> int:
    """Map an onset within a bar onto a fixed number of relative-position bins."""
    if bar_position_bins < 1:
        raise ValueError("bar_position_bins must be >= 1.")
    if bar_steps <= 1:
        return 0
    clamped = min(max(0, position), bar_steps - 1)
    if bar_position_bins >= bar_steps:
        return clamped
    return min(bar_position_bins - 1, (clamped * bar_position_bins) // bar_steps)


def _phrase_flag(bar: int, boundaries: set[int], total_bars: int) -> int:
    """Resolve the phrase flag for an event starting in the given bar."""
    is_start = bar in boundaries
    is_end = bar == total_bars or (bar + 1) in boundaries
    if is_start and is_end:
        return PHRASE_FLAG_MAP["start_end"]
    if is_start:
        return PHRASE_FLAG_MAP["start"]
    if is_end:
        return PHRASE_FLAG_MAP["end"]
    return PHRASE_FLAG_MAP["mid"]


def _phrase_index(bar: int, boundaries: list[int]) -> int:
    """Map a bar number to its containing phrase index."""
    return max(0, bisect_right(boundaries, bar) - 1)


def encode_piece_to_blocks(
    piece: QuantizedPiece,
    *,
    duration_bins: int = 32,
    velocity_bins: int = 16,
    bar_position_bins: int = 16,
) -> PieceExample:
    """Convert a quantized piece into grouped event-block records."""
    ordered_boundaries = sorted(piece.phrase_boundaries or [1])
    boundaries = set(ordered_boundaries)
    event_blocks = [
        EventBlock(
            pitch=event.pitch,
            duration=_bucket_duration(event.duration_steps, duration_bins),
            velocity=_bucket_velocity(event.velocity, velocity_bins),
            bar=event.bar,
            bar_position=_bucket_bar_position(
                event.position,
                bar_steps=piece.bar_steps,
                bar_position_bins=bar_position_bins,
            ),
            phrase_index=_phrase_index(event.bar, ordered_boundaries),
            instrument=event.instrument,
            harmony=GLOBAL_HARMONY_VOCAB.get(event.harmony or "unknown", 0),
            phrase_flag=_phrase_flag(event.bar, boundaries, piece.total_bars),
        )
        for event in piece.note_events
    ]
    return PieceExample(
        piece_id=piece.piece_id,
        event_blocks=event_blocks,
        phrase_boundaries=ordered_boundaries,
        metadata={
            "time_signature": piece.time_signature,
            "tempo_bpm": piece.tempo_bpm,
            "bar_steps": piece.bar_steps,
            "steps_per_beat": piece.steps_per_beat,
            "total_bars": piece.total_bars,
            "harmony_map": GLOBAL_HARMONY_VOCAB,
            "phrase_flag_map": PHRASE_FLAG_MAP.copy(),
            "key": piece.key,
            "chords": list(piece.chords),
        },
    )
