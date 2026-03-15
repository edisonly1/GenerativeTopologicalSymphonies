"""Build a matched reference set from processed pieces and a generation manifest."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from preprocessing import QuantizedPiece, load_quantized_piece_json, write_quantized_piece_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-event", type=int, default=0)
    parser.add_argument("--event-count", type=int, default=None)
    return parser.parse_args()


def slice_quantized_piece(
    piece: QuantizedPiece,
    *,
    start_event: int = 0,
    event_count: int | None = None,
) -> QuantizedPiece:
    """Slice a quantized piece by event index and rebase the segment."""
    if start_event < 0:
        raise ValueError("start_event must be non-negative.")
    end_event = len(piece.note_events) if event_count is None else start_event + event_count
    selected_events = piece.note_events[start_event:end_event]
    if not selected_events:
        raise ValueError(f"Requested slice is empty for piece {piece.piece_id}.")

    first_event = selected_events[0]
    start_step_offset = first_event.start_step
    start_bar_offset = first_event.bar - 1
    rebased_events = [
        replace(
            event,
            start_step=event.start_step - start_step_offset,
            bar=event.bar - start_bar_offset,
        )
        for event in selected_events
    ]
    last_bar = rebased_events[-1].bar
    rebased_boundaries = sorted(
        {
            max(1, boundary - start_bar_offset)
            for boundary in piece.phrase_boundaries
            if start_bar_offset < boundary <= start_bar_offset + last_bar
        }
    )
    if not rebased_boundaries or rebased_boundaries[0] != 1:
        rebased_boundaries.insert(0, 1)

    metadata = dict(piece.metadata)
    metadata.update(
        {
            "source_piece_id": piece.piece_id,
            "slice_start_event": start_event,
            "slice_event_count": len(rebased_events),
        }
    )
    return QuantizedPiece(
        piece_id=piece.piece_id,
        resolution=piece.resolution,
        steps_per_beat=piece.steps_per_beat,
        bar_steps=piece.bar_steps,
        time_signature=piece.time_signature,
        tempo_bpm=piece.tempo_bpm,
        note_events=rebased_events,
        phrase_boundaries=rebased_boundaries,
        source_path=piece.source_path,
        metadata=metadata,
        key=piece.key,
        chords=piece.chords,
    )


def build_matched_reference_set(
    manifest_path: str | Path,
    *,
    processed_dir: str | Path,
    output_dir: str | Path,
    start_event: int = 0,
    default_event_count: int | None = None,
) -> dict[str, Any]:
    """Create a reference set aligned to generated sample lengths."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    processed_root = Path(processed_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    items = []
    for item in manifest.get("items", []):
        piece_id = item["piece_id"]
        event_count = item.get("generated_event_count", default_event_count)
        if event_count is None:
            raise ValueError(
                f"Manifest item {piece_id} is missing generated_event_count and no default was provided."
            )
        source_path = processed_root / f"{piece_id}.json"
        reference_piece = slice_quantized_piece(
            load_quantized_piece_json(source_path),
            start_event=start_event,
            event_count=event_count,
        )
        piece_output_dir = output_root / piece_id
        piece_output_dir.mkdir(parents=True, exist_ok=True)
        output_json = write_quantized_piece_json(reference_piece, piece_output_dir / "piece.json")
        items.append(
            {
                "piece_id": piece_id,
                "source_json": str(source_path),
                "output_json": str(output_json),
                "event_count": len(reference_piece.note_events),
                "start_event": start_event,
            }
        )

    output_manifest = {
        "source_manifest": str(manifest_path),
        "piece_count": len(items),
        "items": items,
    }
    manifest_output_path = output_root / "manifest.json"
    manifest_output_path.write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "piece_count": len(items),
        "manifest_path": str(manifest_output_path),
        "items": items,
    }


def main() -> None:
    args = parse_args()
    result = build_matched_reference_set(
        args.manifest,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        start_event=args.start_event,
        default_event_count=args.event_count,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
