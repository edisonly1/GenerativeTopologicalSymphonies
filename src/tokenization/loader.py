"""Load processed dataset artifacts into tokenized training examples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from preprocessing.serialization import load_quantized_piece_json

from .block_encoder import encode_piece_to_blocks
from .dataset import PieceExample


@dataclass(slots=True)
class DatasetSummary:
    """Compact summary of a loaded dataset split."""

    split: str
    piece_count: int
    event_block_count: int
    min_length: int
    max_length: int
    mean_length: float


def load_split_piece_ids(splits_dir: str | Path, split: str = "train") -> list[str]:
    """Load the piece ids assigned to a split."""
    return _read_split_manifest(Path(splits_dir) / f"{split}.json")


def load_piece_example(
    path: str | Path,
    *,
    duration_bins: int = 32,
    velocity_bins: int = 16,
    bar_position_bins: int = 16,
) -> PieceExample:
    """Load one processed JSON artifact and convert it into grouped event blocks."""
    quantized_piece = load_quantized_piece_json(path)
    return encode_piece_to_blocks(
        quantized_piece,
        duration_bins=duration_bins,
        velocity_bins=velocity_bins,
        bar_position_bins=bar_position_bins,
    )


def _read_split_manifest(path: Path) -> list[str]:
    """Read the piece ids for a dataset split."""
    if not path.exists():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("piece_ids", [])


def load_processed_dataset(
    *,
    processed_dir: str | Path,
    splits_dir: str | Path,
    split: str = "train",
    duration_bins: int = 32,
    velocity_bins: int = 16,
    bar_position_bins: int = 16,
    limit: int | None = None,
) -> list[PieceExample]:
    """Load one processed split into tokenized training examples."""
    processed_root = Path(processed_dir)
    split_ids = load_split_piece_ids(splits_dir, split=split)
    if limit is not None:
        split_ids = split_ids[:limit]
    return [
        load_piece_example(
            processed_root / f"{piece_id}.json",
            duration_bins=duration_bins,
            velocity_bins=velocity_bins,
            bar_position_bins=bar_position_bins,
        )
        for piece_id in split_ids
    ]


def summarize_examples(examples: list[PieceExample], *, split: str) -> DatasetSummary:
    """Summarize a list of tokenized examples."""
    lengths = [len(example) for example in examples]
    if not lengths:
        return DatasetSummary(
            split=split,
            piece_count=0,
            event_block_count=0,
            min_length=0,
            max_length=0,
            mean_length=0.0,
        )
    return DatasetSummary(
        split=split,
        piece_count=len(examples),
        event_block_count=sum(lengths),
        min_length=min(lengths),
        max_length=max(lengths),
        mean_length=mean(lengths),
    )


def example_to_feature_lists(example: PieceExample) -> dict[str, list[int] | str | dict[str, Any]]:
    """Convert a piece example into column-oriented feature lists."""
    return {
        "piece_id": example.piece_id,
        "pitch": [block.pitch for block in example.event_blocks],
        "duration": [block.duration for block in example.event_blocks],
        "velocity": [block.velocity for block in example.event_blocks],
        "bar": [block.bar for block in example.event_blocks],
        "bar_position": [block.bar_position for block in example.event_blocks],
        "phrase_index": [block.phrase_index for block in example.event_blocks],
        "instrument": [block.instrument for block in example.event_blocks],
        "harmony": [block.harmony for block in example.event_blocks],
        "phrase_flag": [block.phrase_flag for block in example.event_blocks],
        "metadata": example.metadata,
    }
