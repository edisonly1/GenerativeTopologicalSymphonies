"""Helpers for working with the ASAP aligned-score dataset."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ASAP_REQUIRED_METADATA_COLUMNS = {
    "composer",
    "title",
    "folder",
    "xml_score",
    "midi_score",
    "midi_performance",
}


@dataclass(slots=True)
class AsapSourceEntry:
    """Resolved ASAP source file plus optional score/performance annotations."""

    midi_path: Path
    source_relative_path: str
    piece_group: str
    metadata: dict[str, Any]
    quality_score: tuple[int, int, int]


def _normalize_relative_path(raw_root: Path, raw_value: str | None) -> Path | None:
    """Resolve an ASAP metadata path relative to the dataset root."""
    if not raw_value:
        return None
    normalized = raw_value.replace("\\", "/").strip()
    if not normalized:
        return None
    candidate = Path(normalized)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        return None
    resolved = raw_root / candidate
    if resolved.exists():
        return resolved
    return None


def _sanitize_group_component(value: str) -> str:
    """Convert free text into a stable grouping token."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return cleaned or "unknown"


def _piece_group_key(composer: str, title: str) -> str:
    """Create a leakage-safe grouping key for ASAP unique pieces."""
    return f"{_sanitize_group_component(composer)}::{_sanitize_group_component(title)}"


def _find_metadata_path(raw_root: Path) -> Path | None:
    """Locate the ASAP metadata CSV inside a raw dataset directory."""
    candidate = raw_root / "metadata.csv"
    return candidate if candidate.exists() else None


def _find_annotations_path(raw_root: Path) -> Path | None:
    """Locate the ASAP combined JSON annotations file."""
    candidate = raw_root / "asap_annotations.json"
    return candidate if candidate.exists() else None


def load_asap_metadata(raw_dir: str | Path) -> list[dict[str, str]] | None:
    """Load the ASAP metadata table when present."""
    raw_root = Path(raw_dir)
    metadata_path = _find_metadata_path(raw_root)
    if metadata_path is None:
        return None
    with metadata_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not ASAP_REQUIRED_METADATA_COLUMNS.issubset(reader.fieldnames):
            return None
        return [dict(row) for row in reader]


def load_asap_annotations(raw_dir: str | Path) -> dict[str, Any]:
    """Load the ASAP JSON annotations file when present."""
    raw_root = Path(raw_dir)
    annotations_path = _find_annotations_path(raw_root)
    if annotations_path is None:
        return {}
    return json.loads(annotations_path.read_text(encoding="utf-8"))


def detect_asap_dataset(raw_dir: str | Path) -> bool:
    """Return True when a directory looks like the ASAP dataset root."""
    return load_asap_metadata(raw_dir) is not None


def _sorted_annotation_items(payload: dict[str, Any] | list[Any] | None) -> list[tuple[str, Any]]:
    """Return annotation entries sorted by time key where possible."""
    if isinstance(payload, dict):
        def _sort_key(item: tuple[str, Any]) -> tuple[int, float | str]:
            raw_key = item[0]
            try:
                return (0, float(raw_key))
            except (TypeError, ValueError):
                return (1, raw_key)

        return sorted(payload.items(), key=_sort_key)
    if isinstance(payload, list):
        return [(str(index), value) for index, value in enumerate(payload)]
    return []


def _extract_primary_time_signature(payload: dict[str, Any] | list[Any] | None) -> str | None:
    """Extract the first time-signature label from ASAP annotations."""
    for _, value in _sorted_annotation_items(payload):
        if isinstance(value, (list, tuple)) and value:
            label = value[0]
            if isinstance(label, str) and "/" in label:
                return label.strip()
        if isinstance(value, str) and "/" in value:
            return value.strip()
        if isinstance(value, dict):
            label = value.get("time_signature_string")
            if isinstance(label, str) and "/" in label:
                return label.strip()
    return None


def _extract_primary_key_signature(payload: dict[str, Any] | list[Any] | None) -> tuple[int | None, int | None]:
    """Extract tonic and accidental-count hints from ASAP key-signature annotations."""
    for _, value in _sorted_annotation_items(payload):
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return int(value[0]), int(value[1])
            except (TypeError, ValueError):
                continue
        if isinstance(value, dict):
            tonic = value.get("key_signature_number")
            accidentals = value.get("number_of_sharps")
            if tonic is None or accidentals is None:
                continue
            try:
                return int(tonic), int(accidentals)
            except (TypeError, ValueError):
                continue
    return None, None


def _annotation_quality_score(annotation_entry: dict[str, Any], *, scope_prefix: str) -> tuple[int, int, int]:
    """Prefer entries with aligned score/performance and richer annotations."""
    aligned = 1 if annotation_entry.get("score_and_performance_aligned") else 0
    has_time_signature = 1 if _extract_primary_time_signature(annotation_entry.get(f"{scope_prefix}_time_signatures")) else 0
    has_key_signature = 1 if _extract_primary_key_signature(annotation_entry.get(f"{scope_prefix}_key_signatures"))[0] is not None else 0
    return (aligned, has_time_signature, has_key_signature)


def _build_entry_metadata(
    *,
    row: dict[str, str],
    annotation_entry: dict[str, Any] | None,
    scope_prefix: str,
    source_mode: str,
) -> dict[str, Any]:
    """Build per-piece metadata from ASAP CSV and JSON annotations."""
    composer = row.get("composer", "").strip() or "unknown"
    title = row.get("title", "").strip() or "unknown"
    piece_group = _piece_group_key(composer, title)
    metadata: dict[str, Any] = {
        "dataset_kind": "asap",
        "dataset_source_mode": source_mode,
        "asap_piece_group": piece_group,
        "composer": composer,
        "title": title,
        "folder": row.get("folder", "").strip(),
        "xml_score": row.get("xml_score", "").strip(),
        "midi_score": row.get("midi_score", "").strip(),
        "midi_performance": row.get("midi_performance", "").strip(),
    }
    if annotation_entry is None:
        return metadata

    primary_time_signature = _extract_primary_time_signature(
        annotation_entry.get(f"{scope_prefix}_time_signatures")
    )
    if primary_time_signature is not None:
        metadata["annotated_primary_time_signature"] = primary_time_signature

    key_tonic, key_accidentals = _extract_primary_key_signature(
        annotation_entry.get(f"{scope_prefix}_key_signatures")
    )
    if key_tonic is not None:
        metadata["annotated_key_signature_tonic"] = key_tonic
        metadata["annotated_key_signature_accidentals"] = key_accidentals

    beats = annotation_entry.get(f"{scope_prefix}_beats")
    downbeats = annotation_entry.get(f"{scope_prefix}_downbeats")
    if isinstance(beats, list):
        metadata["annotated_beat_count"] = len(beats)
    if isinstance(downbeats, list):
        metadata["annotated_downbeat_count"] = len(downbeats)
    downbeat_map = annotation_entry.get("downbeats_score_map")
    if isinstance(downbeat_map, list):
        metadata["annotated_downbeats_score_map"] = list(downbeat_map)
    metadata["score_and_performance_aligned"] = bool(
        annotation_entry.get("score_and_performance_aligned", False)
    )
    return metadata


def build_asap_source_entries(
    raw_dir: str | Path,
    *,
    source_mode: str = "score",
    include_annotations: bool = False,
) -> list[AsapSourceEntry]:
    """Resolve ASAP score or performance MIDI files into unique source entries."""
    if source_mode not in {"score", "performance"}:
        raise ValueError("ASAP source_mode must be 'score' or 'performance'.")

    raw_root = Path(raw_dir)
    metadata_rows = load_asap_metadata(raw_root)
    if metadata_rows is None:
        raise ValueError(f"No ASAP metadata.csv found in {raw_root}")
    annotations = load_asap_annotations(raw_root) if include_annotations else {}

    chosen_entries: dict[str, AsapSourceEntry] = {}
    path_column = "midi_score" if source_mode == "score" else "midi_performance"
    annotation_scope = "midi_score" if source_mode == "score" else "performance"

    for row in metadata_rows:
        midi_path = _normalize_relative_path(raw_root, row.get(path_column))
        if midi_path is None:
            continue
        source_relative_path = midi_path.relative_to(raw_root).as_posix()
        performance_key = row.get("midi_performance", "").replace("\\", "/").strip()
        annotation_entry = annotations.get(performance_key) if performance_key else None
        quality = (
            _annotation_quality_score(annotation_entry, scope_prefix=annotation_scope)
            if annotation_entry is not None
            else (0, 0, 0)
        )
        metadata = _build_entry_metadata(
            row=row,
            annotation_entry=annotation_entry,
            scope_prefix=annotation_scope,
            source_mode=source_mode,
        )
        piece_group = metadata["asap_piece_group"]
        candidate = AsapSourceEntry(
            midi_path=midi_path,
            source_relative_path=source_relative_path,
            piece_group=piece_group,
            metadata=metadata,
            quality_score=quality,
        )
        current = chosen_entries.get(source_relative_path)
        if current is None or candidate.quality_score > current.quality_score:
            chosen_entries[source_relative_path] = candidate

    return sorted(chosen_entries.values(), key=lambda entry: entry.source_relative_path)
