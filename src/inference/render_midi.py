"""Utilities for converting generated tokens back to MIDI."""

from __future__ import annotations

from pathlib import Path

from preprocessing.schema import QuantizedPiece


def _encode_vlq(value: int) -> bytes:
    """Encode an integer as a MIDI variable-length quantity."""
    if value < 0:
        raise ValueError("VLQ values must be non-negative.")
    bytes_out = [value & 0x7F]
    value >>= 7
    while value:
        bytes_out.append(0x80 | (value & 0x7F))
        value >>= 7
    return bytes(reversed(bytes_out))


def _time_signature_meta(time_signature: str) -> bytes:
    """Build a MIDI time-signature meta event payload."""
    numerator_str, denominator_str = time_signature.split("/", maxsplit=1)
    numerator = int(numerator_str)
    denominator = int(denominator_str)
    if denominator <= 0 or denominator & (denominator - 1):
        raise ValueError("MIDI time signatures require a power-of-two denominator.")
    denominator_power = denominator.bit_length() - 1
    return bytes([0xFF, 0x58, 0x04, numerator, denominator_power, 24, 8])


def _tempo_meta(bpm: float) -> bytes:
    """Build a MIDI tempo meta event payload."""
    microseconds_per_quarter = round(60_000_000 / bpm)
    return b"\xFF\x51\x03" + microseconds_per_quarter.to_bytes(3, "big")


def render_piece_to_midi_bytes(piece: QuantizedPiece, *, ticks_per_beat: int = 480) -> bytes:
    """Render a quantized piece into Standard MIDI File bytes."""
    if ticks_per_beat % piece.steps_per_beat != 0:
        raise ValueError("ticks_per_beat must be divisible by piece.steps_per_beat.")
    ticks_per_step = ticks_per_beat // piece.steps_per_beat

    timeline: list[tuple[int, int, bytes]] = [
        (0, 0, _tempo_meta(piece.tempo_bpm)),
        (0, 0, _time_signature_meta(piece.time_signature)),
    ]

    program_events: dict[int, int] = {}
    for event in piece.note_events:
        if not event.is_drum and event.channel not in program_events:
            program_events[event.channel] = min(127, max(0, event.instrument))
    for channel, program in sorted(program_events.items()):
        timeline.append((0, 0, bytes([0xC0 | channel, program])))

    for event in piece.note_events:
        start_tick = event.start_step * ticks_per_step
        end_tick = (event.start_step + event.duration_steps) * ticks_per_step
        channel = event.channel
        timeline.append((start_tick, 2, bytes([0x90 | channel, event.pitch, event.velocity])))
        timeline.append((end_tick, 1, bytes([0x80 | channel, event.pitch, 0])))

    timeline.sort(key=lambda item: (item[0], item[1], item[2]))
    track = bytearray()
    previous_tick = 0
    for tick, _, payload in timeline:
        track.extend(_encode_vlq(tick - previous_tick))
        track.extend(payload)
        previous_tick = tick

    track.extend(_encode_vlq(0))
    track.extend(b"\xFF\x2F\x00")

    header = b"MThd" + (6).to_bytes(4, "big")
    header += (0).to_bytes(2, "big") + (1).to_bytes(2, "big")
    header += ticks_per_beat.to_bytes(2, "big")
    track_chunk = b"MTrk" + len(track).to_bytes(4, "big") + bytes(track)
    return header + track_chunk


def render_piece_to_midi(
    piece: QuantizedPiece,
    path: str | Path,
    *,
    ticks_per_beat: int = 480,
) -> Path:
    """Render a quantized piece to a MIDI file on disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(render_piece_to_midi_bytes(piece, ticks_per_beat=ticks_per_beat))
    return output_path
