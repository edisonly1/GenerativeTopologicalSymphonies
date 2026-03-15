"""Serialization helpers for processed symbolic dataset artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .schema import QuantizedEvent, QuantizedPiece


def quantized_piece_to_dict(piece: QuantizedPiece) -> dict:
    """Convert a quantized piece dataclass into a JSON-serializable dictionary."""
    return asdict(piece)


def quantized_piece_from_dict(payload: dict) -> QuantizedPiece:
    """Reconstruct a quantized piece from a JSON dictionary."""
    note_events = [QuantizedEvent(**event) for event in payload.get("note_events", [])]
    return QuantizedPiece(
        piece_id=payload["piece_id"],
        resolution=payload["resolution"],
        steps_per_beat=payload["steps_per_beat"],
        bar_steps=payload["bar_steps"],
        time_signature=payload["time_signature"],
        tempo_bpm=payload["tempo_bpm"],
        note_events=note_events,
        phrase_boundaries=payload.get("phrase_boundaries", []),
        source_path=payload.get("source_path"),
        metadata=payload.get("metadata", {}),
        key=payload.get("key", "unknown"),
        chords=payload.get("chords", []),
    )


def write_quantized_piece_json(piece: QuantizedPiece, path: str | Path) -> Path:
    """Write a processed piece artifact to disk as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(quantized_piece_to_dict(piece), indent=2), encoding="utf-8")
    return output_path


def load_quantized_piece_json(path: str | Path) -> QuantizedPiece:
    """Load a processed piece artifact from disk."""
    input_path = Path(path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    return quantized_piece_from_dict(payload)
