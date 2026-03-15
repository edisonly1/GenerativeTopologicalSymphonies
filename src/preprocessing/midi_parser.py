"""Pure-Python MIDI parsing entry points for the Phase 0 data pipeline."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import BinaryIO

from .schema import NoteEvent, ParsedPiece, TempoChange, TimeSignatureChange


def _read_vlq(data: bytes, offset: int) -> tuple[int, int]:
    """Read a MIDI variable-length quantity."""
    value = 0
    while True:
        if offset >= len(data):
            raise ValueError("Unexpected end of data while reading VLQ.")
        byte = data[offset]
        offset += 1
        value = (value << 7) | (byte & 0x7F)
        if byte < 0x80:
            return value, offset


def _read_chunk(stream: BinaryIO) -> tuple[bytes, bytes]:
    """Read a single MIDI chunk and return its identifier and payload."""
    chunk_id = stream.read(4)
    if len(chunk_id) != 4:
        raise ValueError("Unexpected end of file while reading chunk id.")
    length_bytes = stream.read(4)
    if len(length_bytes) != 4:
        raise ValueError("Unexpected end of file while reading chunk length.")
    length = int.from_bytes(length_bytes, "big")
    payload = stream.read(length)
    if len(payload) != length:
        raise ValueError("Unexpected end of file while reading chunk payload.")
    return chunk_id, payload


def _finalize_note(
    *,
    piece_id: str,
    ticks_per_beat: int,
    events: list[NoteEvent],
    pitch: int,
    velocity: int,
    start_tick: int,
    end_tick: int,
    instrument: int,
    channel: int,
    track_index: int,
    track_name: str,
) -> None:
    """Create a normalized note event if the note duration is positive."""
    if end_tick <= start_tick:
        end_tick = start_tick + 1
    del piece_id  # Reserved for future per-note metadata.
    events.append(
        NoteEvent(
            pitch=pitch,
            velocity=velocity,
            start_tick=start_tick,
            end_tick=end_tick,
            start_beat=start_tick / ticks_per_beat,
            duration_beats=(end_tick - start_tick) / ticks_per_beat,
            instrument=128 if channel == 9 else instrument,
            channel=channel,
            track_index=track_index,
            track_name=track_name,
            is_drum=channel == 9,
        )
    )


def parse_midi_bytes(data: bytes, piece_id: str = "memory") -> ParsedPiece:
    """Parse raw Standard MIDI File bytes into a normalized representation."""
    from io import BytesIO

    stream = BytesIO(data)
    chunk_id, header = _read_chunk(stream)
    if chunk_id != b"MThd":
        raise ValueError("Invalid MIDI file: missing MThd header.")
    if len(header) != 6:
        raise ValueError("Invalid MIDI header length.")

    midi_format = int.from_bytes(header[0:2], "big")
    track_count = int.from_bytes(header[2:4], "big")
    division = int.from_bytes(header[4:6], "big")
    if division & 0x8000:
        raise NotImplementedError("SMPTE-timed MIDI files are not supported.")
    ticks_per_beat = division

    note_events: list[NoteEvent] = []
    tempo_changes: list[TempoChange] = []
    time_signatures: list[TimeSignatureChange] = []

    for track_index in range(track_count):
        chunk_id, track_data = _read_chunk(stream)
        if chunk_id != b"MTrk":
            raise ValueError("Invalid MIDI file: expected MTrk chunk.")

        offset = 0
        absolute_tick = 0
        running_status: int | None = None
        track_name = f"track_{track_index}"
        programs = [0] * 16
        active_notes: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)

        while offset < len(track_data):
            delta, offset = _read_vlq(track_data, offset)
            absolute_tick += delta
            if offset >= len(track_data):
                break

            status = track_data[offset]
            if status >= 0x80:
                offset += 1
            elif running_status is not None:
                status = running_status
            else:
                raise ValueError("Running status encountered before any status byte.")

            if status == 0xFF:
                running_status = None
                if offset >= len(track_data):
                    raise ValueError("Malformed MIDI meta event.")
                meta_type = track_data[offset]
                offset += 1
                length, offset = _read_vlq(track_data, offset)
                payload = track_data[offset : offset + length]
                offset += length

                if meta_type == 0x03:
                    track_name = payload.decode("utf-8", errors="replace").strip() or track_name
                elif meta_type == 0x51 and len(payload) == 3:
                    microseconds_per_quarter = int.from_bytes(payload, "big")
                    bpm = 60_000_000 / microseconds_per_quarter
                    tempo_changes.append(
                        TempoChange(
                            tick=absolute_tick,
                            beat=absolute_tick / ticks_per_beat,
                            bpm=bpm,
                        )
                    )
                elif meta_type == 0x58 and len(payload) >= 2:
                    numerator = payload[0]
                    denominator = 2 ** payload[1]
                    time_signatures.append(
                        TimeSignatureChange(
                            tick=absolute_tick,
                            beat=absolute_tick / ticks_per_beat,
                            numerator=numerator,
                            denominator=denominator,
                        )
                    )
                elif meta_type == 0x2F:
                    break
                continue

            if status in (0xF0, 0xF7):
                running_status = None
                length, offset = _read_vlq(track_data, offset)
                offset += length
                continue

            running_status = status
            event_type = status >> 4
            channel = status & 0x0F

            if event_type == 0xC:
                programs[channel] = track_data[offset]
                offset += 1
                continue
            if event_type == 0xD:
                offset += 1
                continue

            if offset + 2 > len(track_data):
                raise ValueError("Malformed MIDI channel event.")
            data_1 = track_data[offset]
            data_2 = track_data[offset + 1]
            offset += 2

            if event_type == 0x9 and data_2 > 0:
                active_notes[(channel, data_1)].append((absolute_tick, data_2, programs[channel]))
                continue

            if event_type == 0x8 or (event_type == 0x9 and data_2 == 0):
                stack = active_notes.get((channel, data_1))
                if not stack:
                    continue
                start_tick, velocity, instrument = stack.pop()
                _finalize_note(
                    piece_id=piece_id,
                    ticks_per_beat=ticks_per_beat,
                    events=note_events,
                    pitch=data_1,
                    velocity=velocity,
                    start_tick=start_tick,
                    end_tick=absolute_tick,
                    instrument=instrument,
                    channel=channel,
                    track_index=track_index,
                    track_name=track_name,
                )

        for (channel, pitch), starts in active_notes.items():
            for start_tick, velocity, instrument in starts:
                _finalize_note(
                    piece_id=piece_id,
                    ticks_per_beat=ticks_per_beat,
                    events=note_events,
                    pitch=pitch,
                    velocity=velocity,
                    start_tick=start_tick,
                    end_tick=absolute_tick if absolute_tick > start_tick else start_tick + 1,
                    instrument=instrument,
                    channel=channel,
                    track_index=track_index,
                    track_name=track_name,
                )

    note_events.sort(key=lambda event: (event.start_tick, event.pitch, event.instrument))
    tempo_changes.sort(key=lambda event: event.tick)
    time_signatures.sort(key=lambda event: event.tick)

    if not tempo_changes:
        tempo_changes.append(TempoChange(tick=0, beat=0.0, bpm=120.0))
    if not time_signatures:
        time_signatures.append(TimeSignatureChange(tick=0, beat=0.0, numerator=4, denominator=4))

    return ParsedPiece(
        piece_id=piece_id,
        ticks_per_beat=ticks_per_beat,
        note_events=note_events,
        tempo_changes=tempo_changes,
        time_signature_changes=time_signatures,
        midi_format=midi_format,
    )


def parse_midi_file(path: str | Path) -> ParsedPiece:
    """Parse a MIDI file into a normalized intermediate representation."""
    midi_path = Path(path)
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    piece = parse_midi_bytes(midi_path.read_bytes(), piece_id=midi_path.stem)
    piece.source_path = str(midi_path)
    return piece
