"""Generate a tiny synthetic MIDI dataset for local pipeline testing."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from inference.render_midi import render_piece_to_midi

from .schema import QuantizedEvent, QuantizedPiece

DEFAULT_PATTERNS = (
    [60, 62, 64, 65, 67, 69, 71, 72],
    [60, 64, 67, 72, 67, 64, 60, 55],
    [60, 60, 67, 67, 69, 69, 67, 65],
    [72, 71, 69, 67, 65, 64, 62, 60],
)


def _build_toy_piece(
    piece_id: str,
    pitches: list[int],
    *,
    tempo_bpm: float,
    instrument: int,
    channel: int,
) -> QuantizedPiece:
    """Create a simple quantized melody suitable for pipeline smoke tests."""
    note_events: list[QuantizedEvent] = []
    for index, pitch in enumerate(pitches):
        start_step = index * 4
        duration_steps = 4 if index < len(pitches) - 1 else 8
        note_events.append(
            QuantizedEvent(
                pitch=pitch,
                velocity=96,
                instrument=instrument,
                channel=channel,
                start_step=start_step,
                duration_steps=duration_steps,
                bar=(start_step // 16) + 1,
                position=start_step % 16,
                track_index=0,
            )
        )
    piece = QuantizedPiece(
        piece_id=piece_id,
        resolution="sixteenth",
        steps_per_beat=4,
        bar_steps=16,
        time_signature="4/4",
        tempo_bpm=tempo_bpm,
        note_events=note_events,
        phrase_boundaries=[1],
    )
    piece.source_path = f"{piece_id}.mid"
    return piece


def generate_toy_dataset(
    output_dir: str | Path,
    *,
    num_pieces: int = 6,
    seed: int = 42,
) -> list[Path]:
    """Write a small set of synthetic MIDI files to the raw-data directory."""
    if num_pieces < 1:
        raise ValueError("num_pieces must be >= 1.")

    rng = random.Random(seed)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    written_paths: list[Path] = []
    for index in range(num_pieces):
        base_pattern = list(DEFAULT_PATTERNS[index % len(DEFAULT_PATTERNS)])
        transpose = rng.choice((-5, -2, 0, 2, 5))
        pitches = [min(84, max(48, pitch + transpose)) for pitch in base_pattern]
        tempo_bpm = rng.choice((88.0, 96.0, 104.0, 112.0))
        piece_id = f"toy_{index:03d}"
        piece = _build_toy_piece(
            piece_id,
            pitches,
            tempo_bpm=tempo_bpm,
            instrument=0,
            channel=0,
        )
        midi_path = render_piece_to_midi(piece, output_root / f"{piece_id}.mid")
        written_paths.append(midi_path)
    return written_paths


def parse_args() -> argparse.Namespace:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--num-pieces", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Run the toy-data generator CLI and print a short summary."""
    args = parse_args()
    paths = generate_toy_dataset(
        output_dir=args.output_dir,
        num_pieces=args.num_pieces,
        seed=args.seed,
    )
    print(f"Wrote {len(paths)} toy MIDI file(s) into {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
