"""Dataset schemas for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EventBlock:
    """Single grouped symbolic event block."""

    pitch: int
    duration: int
    velocity: int
    bar: int
    bar_position: int
    phrase_index: int
    instrument: int
    harmony: int
    phrase_flag: int


@dataclass(slots=True)
class PieceExample:
    """Container for a tokenized musical piece."""

    piece_id: str
    event_blocks: list[EventBlock] = field(default_factory=list)
    phrase_boundaries: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return the number of grouped event blocks in the example."""
        return len(self.event_blocks)
