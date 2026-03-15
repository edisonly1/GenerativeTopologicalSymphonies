"""Dataset-preparation CLI for raw MIDI to processed JSON artifacts and splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .asap import build_asap_source_entries, detect_asap_dataset
from .harmony_extract import annotate_quantized_piece_harmony
from .midi_parser import parse_midi_file
from .phrase_segment import segment_phrases
from .quantize import quantize_piece
from .schema import QuantizedPiece
from .serialization import write_quantized_piece_json

MIDI_SUFFIXES = {".mid", ".midi"}


@dataclass(slots=True)
class PreparedArtifact:
    """Summary of a single processed piece artifact."""

    piece_id: str
    source_path: str
    processed_path: str
    note_count: int
    total_bars: int


@dataclass(slots=True)
class DatasetPreparationResult:
    """Summary of the dataset-preparation run."""

    artifacts: list[PreparedArtifact] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)
    split_assignments: dict[str, list[str]] = field(default_factory=dict)
    split_strategy: str = "random"


def detect_dataset_kind(raw_dir: str | Path) -> str:
    """Infer the dataset family from the raw directory contents."""
    raw_root = Path(raw_dir)
    if detect_asap_dataset(raw_root):
        return "asap"
    if detect_maestro_official_splits(raw_root) is not None:
        return "maestro"
    return "generic"


def sanitize_piece_id(relative_path: Path) -> str:
    """Build a deterministic piece identifier from a raw-file relative path."""
    parts = []
    for part in relative_path.with_suffix("").parts:
        cleaned = re.sub(r"[^A-Za-z0-9]+", "_", part).strip("_").lower()
        if cleaned:
            parts.append(cleaned)
    return "__".join(parts) or "piece"


def discover_midi_files(raw_dir: str | Path, recursive: bool = True) -> list[Path]:
    """Return MIDI files from the raw dataset directory."""
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {root}")
    globber = root.rglob if recursive else root.glob
    return sorted(path for path in globber("*") if path.is_file() and path.suffix.lower() in MIDI_SUFFIXES)


def _normalize_split_name(split_name: str) -> str:
    """Map dataset-specific split names into train, val, and test."""
    normalized = split_name.strip().lower()
    if normalized == "validation":
        return "val"
    if normalized in {"train", "val", "test"}:
        return normalized
    raise ValueError(f"Unsupported split name: {split_name!r}")


def detect_maestro_official_splits(
    raw_dir: str | Path,
    *,
    available_piece_ids: set[str] | None = None,
) -> dict[str, list[str]] | None:
    """Detect and load MAESTRO official splits from the local metadata CSV."""
    raw_root = Path(raw_dir)
    csv_files = sorted(raw_root.rglob("maestro-v*.csv"))
    if not csv_files:
        return None

    split_assignments: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    metadata_path = csv_files[0]
    with metadata_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"midi_filename", "split"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            return None
        for row in reader:
            relative_path = (metadata_path.parent / row["midi_filename"]).relative_to(raw_root)
            piece_id = sanitize_piece_id(relative_path)
            if available_piece_ids is not None and piece_id not in available_piece_ids:
                continue
            split_name = _normalize_split_name(row["split"])
            split_assignments[split_name].append(piece_id)

    if available_piece_ids is not None:
        assigned = set().union(*split_assignments.values())
        if assigned != available_piece_ids:
            missing = sorted(available_piece_ids - assigned)
            raise ValueError(
                "Official MAESTRO split metadata did not cover all prepared pieces. "
                f"Missing ids: {missing[:5]}"
            )

    return split_assignments


def _largest_remainder_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    """Convert split ratios into integer counts while preserving the total."""
    raw_counts = {name: ratio * total for name, ratio in ratios.items()}
    counts = {name: int(value) for name, value in raw_counts.items()}
    remaining = total - sum(counts.values())
    remainders = sorted(
        ((raw_counts[name] - counts[name], name) for name in ratios),
        reverse=True,
    )
    for _, split_name in remainders[:remaining]:
        counts[split_name] += 1
    return counts


def assign_splits(
    piece_ids: list[str],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Assign piece ids to reproducible train, validation, and test splits."""
    if not 0 < train_ratio <= 1:
        raise ValueError("train_ratio must be in the range (0, 1].")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in the range [0, 1).")
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0:
        raise ValueError("train_ratio + val_ratio must be <= 1.")

    shuffled = list(piece_ids)
    random.Random(seed).shuffle(shuffled)
    counts = _largest_remainder_counts(
        len(shuffled),
        {"train": train_ratio, "val": val_ratio, "test": test_ratio},
    )

    train_end = counts["train"]
    val_end = train_end + counts["val"]
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def assign_grouped_splits(
    piece_ids: list[str],
    *,
    piece_groups: dict[str, str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Assign splits while keeping grouped pieces together."""
    grouped: dict[str, list[str]] = {}
    for piece_id in piece_ids:
        group_key = piece_groups.get(piece_id, piece_id)
        grouped.setdefault(group_key, []).append(piece_id)

    group_keys = sorted(grouped)
    group_assignments = assign_splits(
        group_keys,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    split_assignments = {"train": [], "val": [], "test": []}
    for split_name, assigned_groups in group_assignments.items():
        for group_key in assigned_groups:
            split_assignments[split_name].extend(sorted(grouped[group_key]))
    return split_assignments


def _write_split_file(path: Path, split_name: str, piece_ids: list[str]) -> None:
    """Write one split manifest."""
    payload = {
        "split": split_name,
        "count": len(piece_ids),
        "piece_ids": piece_ids,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_manifest(
    path: Path,
    *,
    raw_dir: Path,
    processed_dir: Path,
    dataset_kind: str,
    dataset_source_mode: str,
    resolution: str | int,
    phrase_strategy: str,
    annotate_harmony: bool,
    use_dataset_annotations: bool,
    artifacts: list[PreparedArtifact],
    failures: list[dict[str, str]],
    split_strategy: str,
) -> None:
    """Write a top-level processing manifest."""
    payload = {
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "dataset_kind": dataset_kind,
        "dataset_source_mode": dataset_source_mode,
        "resolution": resolution,
        "phrase_strategy": phrase_strategy,
        "annotate_harmony": annotate_harmony,
        "use_dataset_annotations": use_dataset_annotations,
        "piece_count": len(artifacts),
        "failure_count": len(failures),
        "split_strategy": split_strategy,
        "artifacts": [asdict(artifact) for artifact in artifacts],
        "failures": failures,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_dataset(
    *,
    raw_dir: str | Path,
    processed_dir: str | Path,
    splits_dir: str | Path,
    resolution: str | int = "sixteenth",
    phrase_strategy: str = "bars_4",
    annotate_harmony: bool = False,
    dataset_kind: str = "auto",
    dataset_source_mode: str = "performance",
    use_dataset_annotations: bool = False,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    recursive: bool = True,
) -> DatasetPreparationResult:
    """Process a raw MIDI directory into JSON artifacts and train/val/test splits."""
    raw_root = Path(raw_dir)
    processed_root = Path(processed_dir)
    splits_root = Path(splits_dir)
    processed_root.mkdir(parents=True, exist_ok=True)
    splits_root.mkdir(parents=True, exist_ok=True)

    resolved_dataset_kind = dataset_kind
    if resolved_dataset_kind == "auto":
        resolved_dataset_kind = detect_dataset_kind(raw_root)
    if resolved_dataset_kind not in {"generic", "maestro", "asap"}:
        raise ValueError("dataset_kind must be one of: auto, generic, maestro, asap.")

    source_metadata_by_path: dict[str, dict[str, object]] = {}
    piece_groups_by_path: dict[str, str] = {}
    if resolved_dataset_kind == "asap":
        asap_entries = build_asap_source_entries(
            raw_root,
            source_mode=dataset_source_mode,
            include_annotations=use_dataset_annotations,
        )
        midi_files = [entry.midi_path for entry in asap_entries]
        source_metadata_by_path = {
            entry.source_relative_path: dict(entry.metadata)
            for entry in asap_entries
        }
        piece_groups_by_path = {
            entry.source_relative_path: entry.piece_group
            for entry in asap_entries
        }
    else:
        midi_files = discover_midi_files(raw_root, recursive=recursive)
    if not midi_files:
        raise ValueError(f"No MIDI files found in {raw_root}")

    artifacts: list[PreparedArtifact] = []
    failures: list[dict[str, str]] = []
    piece_groups_by_id: dict[str, str] = {}

    for midi_path in midi_files:
        relative_path = midi_path.relative_to(raw_root)
        piece_id = sanitize_piece_id(relative_path)
        try:
            parsed = parse_midi_file(midi_path)
            parsed.piece_id = piece_id
            parsed.metadata.update(source_metadata_by_path.get(relative_path.as_posix(), {}))
            quantized = quantize_piece(parsed, resolution=resolution)
            if annotate_harmony:
                quantized = annotate_quantized_piece_harmony(quantized)
            quantized = segment_phrases(quantized, strategy=phrase_strategy)
            quantized.source_path = relative_path.as_posix()
            quantized.metadata.update(source_metadata_by_path.get(relative_path.as_posix(), {}))
            artifact_path = processed_root / f"{piece_id}.json"
            write_quantized_piece_json(quantized, artifact_path)
            artifacts.append(
                PreparedArtifact(
                    piece_id=piece_id,
                    source_path=relative_path.as_posix(),
                    processed_path=artifact_path.name,
                    note_count=len(quantized.note_events),
                    total_bars=quantized.total_bars,
                )
            )
            if relative_path.as_posix() in piece_groups_by_path:
                piece_groups_by_id[piece_id] = piece_groups_by_path[relative_path.as_posix()]
        except Exception as exc:  # noqa: BLE001
            failures.append({"source_path": relative_path.as_posix(), "error": str(exc)})

    if not artifacts:
        raise RuntimeError("Dataset preparation failed for every MIDI file.")

    available_piece_ids = {artifact.piece_id for artifact in artifacts}
    if resolved_dataset_kind == "maestro":
        official_splits = detect_maestro_official_splits(
            raw_root,
            available_piece_ids=available_piece_ids,
        )
    else:
        official_splits = None

    if official_splits is not None:
        split_assignments = official_splits
        split_strategy = "official_maestro"
    elif resolved_dataset_kind == "asap" and piece_groups_by_id:
        split_assignments = assign_grouped_splits(
            sorted(available_piece_ids),
            piece_groups=piece_groups_by_id,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        split_strategy = "grouped_asap_piece"
    else:
        split_assignments = assign_splits(
            sorted(available_piece_ids),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        split_strategy = "random"
    for split_name, piece_ids in split_assignments.items():
        _write_split_file(splits_root / f"{split_name}.json", split_name, piece_ids)

    _write_manifest(
        processed_root / "manifest.json",
        raw_dir=raw_root,
        processed_dir=processed_root,
        dataset_kind=resolved_dataset_kind,
        dataset_source_mode=dataset_source_mode,
        resolution=resolution,
        phrase_strategy=phrase_strategy,
        annotate_harmony=annotate_harmony,
        use_dataset_annotations=use_dataset_annotations,
        artifacts=artifacts,
        failures=failures,
        split_strategy=split_strategy,
    )

    return DatasetPreparationResult(
        artifacts=artifacts,
        failures=failures,
        split_assignments=split_assignments,
        split_strategy=split_strategy,
    )


def parse_args() -> argparse.Namespace:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--resolution", default="sixteenth")
    parser.add_argument("--phrase-strategy", default="bars_4")
    parser.add_argument(
        "--dataset-kind",
        default="auto",
        choices=("auto", "generic", "maestro", "asap"),
        help="Select dataset-specific discovery and split logic.",
    )
    parser.add_argument(
        "--dataset-source-mode",
        default="performance",
        choices=("performance", "score"),
        help="For ASAP, choose whether to process performance or score MIDI files.",
    )
    parser.add_argument(
        "--annotate-harmony",
        action="store_true",
        help="Estimate bar-level harmony/key annotations and write them into processed pieces.",
    )
    parser.add_argument(
        "--use-dataset-annotations",
        action="store_true",
        help="For supported datasets such as ASAP, inject official time/key metadata into processed pieces.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only scan the top level of the raw dataset directory.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the dataset-preparation CLI and print a short summary."""
    args = parse_args()
    result = prepare_dataset(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        splits_dir=args.splits_dir,
        resolution=args.resolution,
        phrase_strategy=args.phrase_strategy,
        annotate_harmony=args.annotate_harmony,
        dataset_kind=args.dataset_kind,
        dataset_source_mode=args.dataset_source_mode,
        use_dataset_annotations=args.use_dataset_annotations,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        recursive=not args.non_recursive,
    )
    print(
        "Prepared "
        f"{len(result.artifacts)} piece(s); "
        f"train={len(result.split_assignments['train'])}, "
        f"val={len(result.split_assignments['val'])}, "
        f"test={len(result.split_assignments['test'])}, "
        f"failures={len(result.failures)}"
    )


if __name__ == "__main__":
    main()
