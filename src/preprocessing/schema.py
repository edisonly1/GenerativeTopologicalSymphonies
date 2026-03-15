"""Typed schemas for parsed and quantized symbolic music data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TempoChange:
    """Tempo change expressed in both ticks and quarter-note beats."""

    tick: int
    beat: float
    bpm: float


@dataclass(slots=True)
class TimeSignatureChange:
    """Time-signature change expressed in both ticks and quarter-note beats."""

    tick: int
    beat: float
    numerator: int
    denominator: int

    @property
    def beats_per_bar(self) -> float:
        """Return the bar length in quarter-note beats."""
        return self.numerator * (4 / self.denominator)

    @property
    def label(self) -> str:
        """Return a compact string label such as 4/4."""
        return f"{self.numerator}/{self.denominator}"


@dataclass(slots=True)
class NoteEvent:
    """Single note event in a normalized parsed representation."""

    pitch: int
    velocity: int
    start_tick: int
    end_tick: int
    start_beat: float
    duration_beats: float
    instrument: int
    channel: int
    track_index: int
    track_name: str = ""
    is_drum: bool = False


@dataclass(slots=True)
class ParsedPiece:
    """Normalized output of the MIDI parser."""

    piece_id: str
    ticks_per_beat: int
    note_events: list[NoteEvent] = field(default_factory=list)
    tempo_changes: list[TempoChange] = field(default_factory=list)
    time_signature_changes: list[TimeSignatureChange] = field(default_factory=list)
    source_path: str | None = None
    midi_format: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_tempo_bpm(self) -> float:
        """Return the first tempo or the MIDI default."""
        return self.tempo_changes[0].bpm if self.tempo_changes else 120.0

    @property
    def primary_time_signature(self) -> TimeSignatureChange:
        """Return the first time signature or a 4/4 default."""
        if self.time_signature_changes:
            return self.time_signature_changes[0]
        return TimeSignatureChange(tick=0, beat=0.0, numerator=4, denominator=4)


@dataclass(slots=True)
class QuantizedEvent:
    """Quantized note event on a discrete rhythmic grid."""

    pitch: int
    velocity: int
    instrument: int
    channel: int
    start_step: int
    duration_steps: int
    bar: int
    position: int
    track_index: int
    track_name: str = ""
    is_drum: bool = False
    harmony: str = "unknown"


@dataclass(slots=True)
class QuantizedPiece:
    """Quantized representation used by tokenization and training."""

    piece_id: str
    resolution: str
    steps_per_beat: int
    bar_steps: int
    time_signature: str
    tempo_bpm: float
    note_events: list[QuantizedEvent] = field(default_factory=list)
    phrase_boundaries: list[int] = field(default_factory=list)
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    key: str = "unknown"
    chords: list[str] = field(default_factory=list)

    @property
    def total_steps(self) -> int:
        """Return the inclusive length of the piece on the quantized grid."""
        if not self.note_events:
            return self.bar_steps
        return max(event.start_step + event.duration_steps for event in self.note_events)

    @property
    def total_bars(self) -> int:
        """Return the number of bars implied by the quantized content."""
        steps = self.total_steps
        return max(1, (steps + self.bar_steps - 1) // self.bar_steps)
