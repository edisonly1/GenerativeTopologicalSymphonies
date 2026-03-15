"""Post-process generated symbolic pieces to reduce obvious playability artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from inference.render_midi import render_piece_to_midi
from preprocessing import QuantizedEvent, QuantizedPiece, load_quantized_piece_json, write_quantized_piece_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preserve-prefix-events", type=int, default=64)
    parser.add_argument("--max-notes-per-onset", type=int, default=6)
    parser.add_argument("--max-simultaneous-span", type=int, default=24)
    parser.add_argument("--trim-same-pitch-overlaps", action="store_true")
    return parser.parse_args()


def _discover_piece_paths(input_dir: Path) -> list[Path]:
    ignored_names = {"manifest.json", "summary.json", "metrics.jsonl"}
    return sorted(
        path
        for path in input_dir.rglob("*.json")
        if path.name not in ignored_names
    )


def _event_end(event: QuantizedEvent) -> int:
    return event.start_step + event.duration_steps


def _rebase_event_timing(event: QuantizedEvent, *, start_step: int, bar_steps: int) -> QuantizedEvent:
    """Return a copy whose time fields are consistent with the new start step."""
    return replace(
        event,
        start_step=start_step,
        bar=(start_step // bar_steps) + 1,
        position=start_step % bar_steps,
    )


def _trim_same_pitch_overlaps(
    piece: QuantizedPiece,
    *,
    preserve_prefix_events: int,
) -> list[tuple[QuantizedEvent, bool]]:
    """Trim or delay same-pitch overlaps while respecting an immutable prompt prefix."""
    processed: list[tuple[QuantizedEvent, bool]] = []
    active_by_key: dict[tuple[int, int], int] = {}

    ordered_events = [
        (index, event)
        for index, event in enumerate(
            sorted(piece.note_events, key=lambda item: (item.start_step, item.pitch, item.instrument, item.track_index))
        )
    ]
    for original_index, raw_event in ordered_events:
        mutable = original_index >= preserve_prefix_events
        event = raw_event
        key = (event.instrument, event.pitch)
        previous_index = active_by_key.get(key)
        if previous_index is not None:
            previous_event, previous_mutable = processed[previous_index]
            previous_end = _event_end(previous_event)
            if event.start_step < previous_end:
                if previous_mutable:
                    trimmed_duration = max(1, event.start_step - previous_event.start_step)
                    previous_event = replace(previous_event, duration_steps=trimmed_duration)
                    processed[previous_index] = (previous_event, previous_mutable)
                    previous_end = _event_end(previous_event)
                elif mutable:
                    event = _rebase_event_timing(
                        event,
                        start_step=previous_end,
                        bar_steps=piece.bar_steps,
                    )
        processed.append((event, mutable))
        active_by_key[key] = len(processed) - 1
    return processed


def _select_dense_cluster(
    events: list[QuantizedEvent],
    *,
    limit: int,
    max_span: int,
) -> list[QuantizedEvent]:
    """Keep the densest pitch cluster subject to note-count/span limits."""
    if not events or limit <= 0:
        return []
    ordered = sorted(events, key=lambda item: (item.pitch, item.instrument, item.track_index))
    best_cluster = [ordered[0]]
    for left in range(len(ordered)):
        for right in range(left, len(ordered)):
            cluster = ordered[left : right + 1]
            if len(cluster) > limit:
                break
            span = cluster[-1].pitch - cluster[0].pitch
            if span > max_span:
                break
            if len(cluster) > len(best_cluster):
                best_cluster = cluster
            elif len(cluster) == len(best_cluster):
                best_span = best_cluster[-1].pitch - best_cluster[0].pitch
                if span < best_span:
                    best_cluster = cluster
    if len(best_cluster) >= min(limit, len(ordered)):
        return best_cluster

    # If nothing fits the span limit at full size, keep the tightest window up to `limit`.
    best_cluster = ordered[: min(limit, len(ordered))]
    best_span = best_cluster[-1].pitch - best_cluster[0].pitch
    for left in range(len(ordered)):
        right = min(len(ordered), left + limit)
        cluster = ordered[left:right]
        if not cluster:
            continue
        span = cluster[-1].pitch - cluster[0].pitch
        if span < best_span:
            best_span = span
            best_cluster = cluster
    return best_cluster


def _cleanup_onset_group(
    group: list[tuple[QuantizedEvent, bool]],
    *,
    max_notes_per_onset: int,
    max_simultaneous_span: int,
) -> list[tuple[QuantizedEvent, bool]]:
    """Reduce pathological simultaneities while keeping immutable prompt notes."""
    if len(group) <= 1:
        return group

    deduped: dict[tuple[int, int], tuple[QuantizedEvent, bool]] = {}
    for event, mutable in group:
        key = (event.instrument, event.pitch)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = (event, mutable)
            continue
        existing_event, existing_mutable = existing
        if not existing_mutable and mutable:
            continue
        if existing_mutable and not mutable:
            deduped[key] = (event, mutable)
            continue
        if event.duration_steps > existing_event.duration_steps:
            deduped[key] = (event, mutable)

    reduced = list(deduped.values())
    fixed = [item for item in reduced if not item[1]]
    mutable = [item for item in reduced if item[1]]
    if len(fixed) >= max_notes_per_onset:
        return sorted(fixed, key=lambda item: (item[0].pitch, item[0].instrument, item[0].track_index))

    remaining = max_notes_per_onset - len(fixed)
    selected_mutable_events = _select_dense_cluster(
        [event for event, _ in mutable],
        limit=remaining,
        max_span=max_simultaneous_span,
    )
    selected_mutable_keys = {
        (event.instrument, event.pitch, event.track_index, event.duration_steps)
        for event in selected_mutable_events
    }
    selected_mutable: list[tuple[QuantizedEvent, bool]] = []
    for item in mutable:
        event = item[0]
        key = (event.instrument, event.pitch, event.track_index, event.duration_steps)
        if key in selected_mutable_keys:
            selected_mutable.append(item)
            selected_mutable_keys.remove(key)

    combined = fixed + selected_mutable
    combined.sort(key=lambda item: (item[0].pitch, item[0].instrument, item[0].track_index))
    if len(combined) <= 1:
        return combined
    if combined[-1][0].pitch - combined[0][0].pitch <= max_simultaneous_span:
        return combined

    fixed_events = [event for event, _ in fixed]
    mutable_events = [event for event, _ in selected_mutable]
    if not mutable_events:
        return combined
    allowed_mutable = _select_dense_cluster(
        mutable_events,
        limit=max(0, max_notes_per_onset - len(fixed_events)),
        max_span=max_simultaneous_span,
    )
    allowed_keys = {
        (event.instrument, event.pitch, event.track_index, event.duration_steps)
        for event in allowed_mutable
    }
    trimmed = fixed.copy()
    for item in selected_mutable:
        event = item[0]
        key = (event.instrument, event.pitch, event.track_index, event.duration_steps)
        if key in allowed_keys:
            trimmed.append(item)
            allowed_keys.remove(key)
    trimmed.sort(key=lambda item: (item[0].pitch, item[0].instrument, item[0].track_index))
    return trimmed


def cleanup_piece(
    piece: QuantizedPiece,
    *,
    preserve_prefix_events: int = 64,
    max_notes_per_onset: int = 6,
    max_simultaneous_span: int = 24,
    trim_same_pitch_overlaps: bool = True,
) -> QuantizedPiece:
    """Apply conservative cleanups that improve piano-like symbolic plausibility."""
    processed = [
        (event, index >= preserve_prefix_events)
        for index, event in enumerate(piece.note_events)
    ]
    if trim_same_pitch_overlaps:
        processed = _trim_same_pitch_overlaps(
            piece,
            preserve_prefix_events=preserve_prefix_events,
        )

    cleaned: list[tuple[QuantizedEvent, bool]] = []
    sorted_processed = sorted(
        processed,
        key=lambda item: (item[0].start_step, item[0].pitch, item[0].instrument, item[0].track_index),
    )
    index = 0
    while index < len(sorted_processed):
        start_step = sorted_processed[index][0].start_step
        group: list[tuple[QuantizedEvent, bool]] = []
        while index < len(sorted_processed) and sorted_processed[index][0].start_step == start_step:
            group.append(sorted_processed[index])
            index += 1
        cleaned.extend(
            _cleanup_onset_group(
                group,
                max_notes_per_onset=max_notes_per_onset,
                max_simultaneous_span=max_simultaneous_span,
            )
        )

    note_events = [
        event
        for event, _ in sorted(
            cleaned,
            key=lambda item: (item[0].start_step, item[0].pitch, item[0].instrument, item[0].track_index),
        )
    ]
    metadata = dict(piece.metadata)
    metadata.update(
        {
            "cleaned": True,
            "cleanup_preserve_prefix_events": preserve_prefix_events,
            "cleanup_max_notes_per_onset": max_notes_per_onset,
            "cleanup_max_simultaneous_span": max_simultaneous_span,
            "cleanup_trim_same_pitch_overlaps": trim_same_pitch_overlaps,
        }
    )
    return replace(piece, note_events=note_events, metadata=metadata)


def cleanup_directory(
    input_dir: str | Path,
    *,
    output_dir: str | Path,
    preserve_prefix_events: int = 64,
    max_notes_per_onset: int = 6,
    max_simultaneous_span: int = 24,
    trim_same_pitch_overlaps: bool = True,
) -> dict[str, object]:
    """Apply cleanup_piece to every generated sample in a directory tree."""
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, object]] = []
    for piece_path in _discover_piece_paths(input_root):
        draft_piece = load_quantized_piece_json(piece_path)
        cleaned_piece = cleanup_piece(
            draft_piece,
            preserve_prefix_events=preserve_prefix_events,
            max_notes_per_onset=max_notes_per_onset,
            max_simultaneous_span=max_simultaneous_span,
            trim_same_pitch_overlaps=trim_same_pitch_overlaps,
        )
        relative_key = str(piece_path.relative_to(input_root).parent)
        piece_output_dir = output_root / relative_key
        piece_output_dir.mkdir(parents=True, exist_ok=True)
        output_json = write_quantized_piece_json(cleaned_piece, piece_output_dir / "piece.json")
        output_midi = render_piece_to_midi(cleaned_piece, piece_output_dir / "piece.mid")
        items.append(
            {
                "piece_id": relative_key,
                "source_json": str(piece_path),
                "output_json": str(output_json),
                "output_midi": str(output_midi),
                "generated_event_count": len(cleaned_piece.note_events),
            }
        )

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "piece_count": len(items),
                "items": items,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "piece_count": len(items),
        "manifest_path": str(manifest_path),
        "items": items,
    }


def main() -> None:
    args = parse_args()
    result = cleanup_directory(
        args.input_dir,
        output_dir=args.output_dir,
        preserve_prefix_events=args.preserve_prefix_events,
        max_notes_per_onset=args.max_notes_per_onset,
        max_simultaneous_span=args.max_simultaneous_span,
        trim_same_pitch_overlaps=args.trim_same_pitch_overlaps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
