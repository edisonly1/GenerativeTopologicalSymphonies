"""Time-grid quantization utilities."""

from __future__ import annotations

import re

from .schema import ParsedPiece, QuantizedEvent, QuantizedPiece


RESOLUTION_TO_STEPS_PER_BEAT = {
    "quarter": 1,
    "eighth": 2,
    "triplet_eighth": 3,
    "sixteenth": 4,
    "triplet_sixteenth": 6,
    "thirty_second": 8,
}


def resolve_steps_per_beat(resolution: str | int) -> int:
    """Translate a named resolution into integer steps per beat."""
    if isinstance(resolution, int):
        if resolution < 1:
            raise ValueError("steps_per_beat must be >= 1.")
        return resolution
    if resolution not in RESOLUTION_TO_STEPS_PER_BEAT:
        supported = ", ".join(sorted(RESOLUTION_TO_STEPS_PER_BEAT))
        raise ValueError(f"Unsupported resolution {resolution!r}. Expected one of: {supported}.")
    return RESOLUTION_TO_STEPS_PER_BEAT[resolution]


def _resolve_primary_time_signature(piece: ParsedPiece):
    """Prefer dataset-provided time signatures when available."""
    annotated_label = piece.metadata.get("annotated_primary_time_signature")
    if isinstance(annotated_label, str):
        match = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", annotated_label)
        if match:
            numerator = int(match.group(1))
            denominator = int(match.group(2))
            if numerator > 0 and denominator > 0:
                from .schema import TimeSignatureChange

                return TimeSignatureChange(
                    tick=0,
                    beat=0.0,
                    numerator=numerator,
                    denominator=denominator,
                )
    return piece.primary_time_signature


def quantize_piece(piece: ParsedPiece, resolution: str | int = "sixteenth") -> QuantizedPiece:
    """Snap parsed note events onto a discrete rhythmic grid."""
    steps_per_beat = resolve_steps_per_beat(resolution)
    time_signature = _resolve_primary_time_signature(piece)
    bar_steps = int(round(time_signature.beats_per_bar * steps_per_beat))
    if bar_steps < 1:
        raise ValueError("Computed bar_steps must be positive.")

    note_events: list[QuantizedEvent] = []
    for note in piece.note_events:
        start_step = int(round(note.start_beat * steps_per_beat))
        end_step = max(
            start_step + 1,
            int(round((note.start_beat + note.duration_beats) * steps_per_beat)),
        )
        duration_steps = end_step - start_step
        bar = (start_step // bar_steps) + 1
        position = start_step % bar_steps
        note_events.append(
            QuantizedEvent(
                pitch=note.pitch,
                velocity=note.velocity,
                instrument=note.instrument,
                channel=note.channel,
                start_step=start_step,
                duration_steps=duration_steps,
                bar=bar,
                position=position,
                track_index=note.track_index,
                track_name=note.track_name,
                is_drum=note.is_drum,
            )
        )

    note_events.sort(key=lambda event: (event.start_step, event.pitch, event.instrument))

    return QuantizedPiece(
        piece_id=piece.piece_id,
        resolution=str(resolution),
        steps_per_beat=steps_per_beat,
        bar_steps=bar_steps,
        time_signature=time_signature.label,
        tempo_bpm=piece.primary_tempo_bpm,
        note_events=note_events,
        source_path=piece.source_path,
        metadata={
            "ticks_per_beat": piece.ticks_per_beat,
            "midi_format": piece.midi_format,
            "time_signature_changes": len(piece.time_signature_changes),
            "tempo_changes": len(piece.tempo_changes),
        },
    )
