"""PyTorch dataset and collation utilities for autoregressive token training."""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from preprocessing.harmony_extract import (
    GLOBAL_HARMONY_ID_TO_LABEL,
    GLOBAL_HARMONY_VOCAB,
    transpose_chord_label,
    transpose_key_label,
)
from tokenization import example_to_feature_lists, load_piece_example, load_split_piece_ids
from tokenization.dataset import EventBlock, PieceExample
from training.conductor_targets import CONDUCTOR_TARGET_NAMES, derive_phrase_control_targets

FEATURE_NAMES = (
    "pitch",
    "duration",
    "velocity",
    "bar_position",
    "instrument",
    "harmony",
    "phrase_flag",
)


@dataclass(slots=True)
class AutoregressiveBatch:
    """Batch container for next-step grouped-token prediction."""

    piece_ids: list[str]
    inputs: dict[str, Tensor]
    targets: dict[str, Tensor]
    attention_mask: Tensor
    lengths: Tensor
    phrase_boundaries: list[list[int]]
    metadata: list[dict[str, Any]]
    window_ranges: list[tuple[int, int]]
    phrase_ids: Tensor
    phrase_mask: Tensor
    phrase_complete_mask: Tensor
    conductor_targets: dict[str, Tensor]
    phrase_spans: list[list[tuple[int, int]]] = field(default_factory=list)


class AutoregressiveTokenDataset(Dataset[dict[str, Any]]):
    """Lazily load processed pieces and convert them into autoregressive examples."""

    def __init__(
        self,
        *,
        processed_dir: str | Path,
        splits_dir: str | Path,
        split: str = "train",
        duration_bins: int = 32,
        velocity_bins: int = 16,
        bar_position_bins: int = 16,
        limit: int | None = None,
        cache_examples: bool = False,
        transpose_semitones: list[int] | None = None,
        transpose_probability: float = 0.0,
        transpose_min_pitch: int = 0,
        transpose_max_pitch: int = 127,
    ) -> None:
        self.processed_root = Path(processed_dir)
        self.duration_bins = duration_bins
        self.velocity_bins = velocity_bins
        self.bar_position_bins = bar_position_bins
        self.cache_examples = cache_examples
        self.transpose_semitones = list(transpose_semitones or [])
        self.transpose_probability = float(transpose_probability)
        self.transpose_min_pitch = int(transpose_min_pitch)
        self.transpose_max_pitch = int(transpose_max_pitch)
        self.piece_ids = load_split_piece_ids(splits_dir, split=split)
        if limit is not None:
            self.piece_ids = self.piece_ids[:limit]
        self._cache: dict[str, dict[str, Any]] = {}

    def __len__(self) -> int:
        """Return the number of pieces in the split."""
        return len(self.piece_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one tokenized piece and convert it into shifted training features."""
        piece_id = self.piece_ids[index]
        augmentation_active = bool(self.transpose_semitones) and self.transpose_probability > 0.0
        if self.cache_examples and not augmentation_active and piece_id in self._cache:
            return self._cache[piece_id]

        example = load_piece_example(
            self.processed_root / f"{piece_id}.json",
            duration_bins=self.duration_bins,
            velocity_bins=self.velocity_bins,
            bar_position_bins=self.bar_position_bins,
        )
        if augmentation_active and random.random() < self.transpose_probability:
            semitones = _select_transposition(
                example,
                self.transpose_semitones,
                min_pitch=self.transpose_min_pitch,
                max_pitch=self.transpose_max_pitch,
            )
            if semitones != 0:
                example = _transpose_piece_example(example, semitones)
        sample = piece_example_to_autoregressive_sample(example)
        if self.cache_examples and not augmentation_active:
            self._cache[piece_id] = sample
        return sample


def _select_transposition(
    example: PieceExample,
    semitone_choices: list[int],
    *,
    min_pitch: int,
    max_pitch: int,
) -> int:
    """Choose a valid transposition that keeps all notes inside the requested range."""
    if not semitone_choices or not example.event_blocks:
        return 0
    lowest_pitch = min(block.pitch for block in example.event_blocks)
    highest_pitch = max(block.pitch for block in example.event_blocks)
    valid_choices = [
        semitones
        for semitones in semitone_choices
        if min_pitch <= lowest_pitch + semitones and highest_pitch + semitones <= max_pitch
    ]
    if not valid_choices:
        return 0
    return random.choice(valid_choices)


def _transpose_harmony_id(harmony_id: int, semitones: int) -> int:
    """Transpose a harmony token id through the shared global harmony vocabulary."""
    label = GLOBAL_HARMONY_ID_TO_LABEL.get(harmony_id, "unknown")
    transposed_label = transpose_chord_label(label, semitones)
    return GLOBAL_HARMONY_VOCAB.get(transposed_label, 0)


def _transpose_piece_example(example: PieceExample, semitones: int) -> PieceExample:
    """Transpose pitches and harmony labels together for train-time augmentation."""
    transposed_blocks = [
        replace(
            block,
            pitch=block.pitch + semitones,
            harmony=_transpose_harmony_id(block.harmony, semitones),
        )
        for block in example.event_blocks
    ]
    metadata = dict(example.metadata)
    if "key" in metadata:
        metadata["key"] = transpose_key_label(str(metadata["key"]), semitones)
    if "chords" in metadata:
        metadata["chords"] = [
            transpose_chord_label(str(chord), semitones)
            for chord in metadata["chords"]
        ]
    return replace(
        example,
        event_blocks=transposed_blocks,
        metadata=metadata,
    )


def piece_example_to_autoregressive_sample(example: Any) -> dict[str, Any]:
    """Convert a tokenized piece into shifted next-step training tensors."""
    feature_lists = example_to_feature_lists(example)
    phrase_targets = derive_phrase_control_targets(example)
    feature_tensors = {
        feature: torch.tensor(feature_lists[feature], dtype=torch.long) + 1
        for feature in FEATURE_NAMES
    }
    sequence_length = max(0, len(example) - 1)
    inputs = {
        feature: values[:-1] if sequence_length > 0 else values.new_empty((0,))
        for feature, values in feature_tensors.items()
    }
    targets = {
        feature: values[1:] if sequence_length > 0 else values.new_empty((0,))
        for feature, values in feature_tensors.items()
    }
    phrase_ids = torch.tensor(phrase_targets.phrase_ids, dtype=torch.long)
    conductor_targets = {
        name: torch.tensor(values, dtype=torch.long)
        for name, values in phrase_targets.targets.items()
    }
    return {
        "piece_id": example.piece_id,
        "inputs": inputs,
        "targets": targets,
        "sequence_length": sequence_length,
        "phrase_boundaries": list(example.phrase_boundaries),
        "metadata": dict(example.metadata),
        "window_range": (0, sequence_length),
        "phrase_ids": phrase_ids,
        "phrase_ranges": list(phrase_targets.phrase_ranges),
        "phrase_complete": [True] * len(phrase_targets.phrase_ranges),
        "conductor_targets": conductor_targets,
    }


class WindowedAutoregressiveTokenDataset(Dataset[dict[str, Any]]):
    """Slice full autoregressive pieces into shorter overlapping training windows."""

    def __init__(
        self,
        *,
        processed_dir: str | Path,
        splits_dir: str | Path,
        split: str = "train",
        duration_bins: int = 32,
        velocity_bins: int = 16,
        bar_position_bins: int = 16,
        sequence_window: int = 512,
        sequence_hop: int | None = None,
        min_sequence_length: int = 64,
        phrase_aligned_windows: bool = False,
        min_complete_phrases: int = 0,
        min_distinct_phrases: int = 0,
        complete_phrase_weight: float = 0.0,
        distinct_phrase_weight: float = 0.0,
        recurrence_boost: float = 0.0,
        cadence_boost: float = 0.0,
        transpose_semitones: list[int] | None = None,
        transpose_probability: float = 0.0,
        transpose_min_pitch: int = 0,
        transpose_max_pitch: int = 127,
        limit_pieces: int | None = None,
        cache_examples: bool = False,
    ) -> None:
        self.piece_dataset = AutoregressiveTokenDataset(
            processed_dir=processed_dir,
            splits_dir=splits_dir,
            split=split,
            duration_bins=duration_bins,
            velocity_bins=velocity_bins,
            bar_position_bins=bar_position_bins,
            limit=limit_pieces,
            cache_examples=cache_examples,
            transpose_semitones=transpose_semitones,
            transpose_probability=transpose_probability,
            transpose_min_pitch=transpose_min_pitch,
            transpose_max_pitch=transpose_max_pitch,
        )
        self.sequence_window = sequence_window
        self.sequence_hop = sequence_hop or sequence_window
        self.min_sequence_length = min_sequence_length
        self.phrase_aligned_windows = phrase_aligned_windows
        self.min_complete_phrases = max(0, min_complete_phrases)
        self.min_distinct_phrases = max(0, min_distinct_phrases)
        self.complete_phrase_weight = float(complete_phrase_weight)
        self.distinct_phrase_weight = float(distinct_phrase_weight)
        self.recurrence_boost = float(recurrence_boost)
        self.cadence_boost = float(cadence_boost)
        self.window_index, self.window_weights = self._build_window_index()

    def _window_qualifies(self, sample: dict[str, Any], start: int, end: int) -> bool:
        """Return whether a candidate window carries enough complete phrase structure."""
        if end - start < self.min_sequence_length:
            return False
        if self.min_complete_phrases == 0 and self.min_distinct_phrases == 0:
            return True
        complete_phrases = 0
        overlapping_phrases = 0
        for phrase_start, phrase_end in sample["phrase_ranges"]:
            if phrase_end <= start or phrase_start >= end:
                continue
            overlapping_phrases += 1
            if phrase_start >= start and phrase_end <= end:
                complete_phrases += 1
        return (
            complete_phrases >= self.min_complete_phrases
            and overlapping_phrases >= self.min_distinct_phrases
        )

    def _window_weight(self, sample: dict[str, Any], start: int, end: int) -> float:
        """Score a window so structurally rich phrases can be sampled more often."""
        weight = 1.0
        complete_phrase_ids: list[int] = []
        overlapping_phrases = 0
        for phrase_id, (phrase_start, phrase_end) in enumerate(sample["phrase_ranges"]):
            if phrase_end <= start or phrase_start >= end:
                continue
            overlapping_phrases += 1
            if phrase_start >= start and phrase_end <= end:
                complete_phrase_ids.append(phrase_id)
        if self.complete_phrase_weight > 0.0:
            weight += self.complete_phrase_weight * len(complete_phrase_ids)
        if self.distinct_phrase_weight > 0.0:
            weight += self.distinct_phrase_weight * overlapping_phrases
        for phrase_id in complete_phrase_ids:
            if (
                self.recurrence_boost > 0.0
                and int(sample["conductor_targets"]["recurrence"][phrase_id].item()) > 0
            ):
                weight += self.recurrence_boost
            if (
                self.cadence_boost > 0.0
                and int(sample["conductor_targets"]["cadence"][phrase_id].item()) > 0
            ):
                weight += self.cadence_boost
        return max(weight, 1e-6)

    def _build_window_index(self) -> tuple[list[tuple[int, int, int]], list[float]]:
        """Create a list of piece-index/window-range tuples and matching sample weights."""
        index: list[tuple[int, int, int]] = []
        weights: list[float] = []
        for piece_index in range(len(self.piece_dataset)):
            sample = self.piece_dataset[piece_index]
            sequence_length = sample["sequence_length"]
            if sequence_length < self.min_sequence_length:
                continue
            if sequence_length <= self.sequence_window:
                if self._window_qualifies(sample, 0, sequence_length):
                    index.append((piece_index, 0, sequence_length))
                    weights.append(self._window_weight(sample, 0, sequence_length))
                continue

            if self.phrase_aligned_windows and sample["phrase_ranges"]:
                starts = sorted(
                    {
                        max(0, min(sequence_length - self.sequence_window, phrase_start))
                        for phrase_start, _ in sample["phrase_ranges"]
                    }
                )
            else:
                starts = list(
                    range(0, max(1, sequence_length - self.sequence_window + 1), self.sequence_hop)
                )
                last_start = max(0, sequence_length - self.sequence_window)
                if starts[-1] != last_start:
                    starts.append(last_start)
            for start in starts:
                end = min(sequence_length, start + self.sequence_window)
                if self._window_qualifies(sample, start, end):
                    index.append((piece_index, start, end))
                    weights.append(self._window_weight(sample, start, end))
        return index, weights

    def __len__(self) -> int:
        """Return the number of training windows in the split."""
        return len(self.window_index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one windowed autoregressive training sample."""
        piece_index, start, end = self.window_index[index]
        sample = self.piece_dataset[piece_index]
        window_phrase_ids = sample["phrase_ids"][start:end]
        local_phrase_order: list[int] = []
        local_phrase_map: dict[int, int] = {}
        for phrase_id in window_phrase_ids.tolist():
            if phrase_id not in local_phrase_map:
                local_phrase_map[phrase_id] = len(local_phrase_order)
                local_phrase_order.append(phrase_id)
        remapped_phrase_ids = torch.tensor(
            [local_phrase_map[phrase_id] for phrase_id in window_phrase_ids.tolist()],
            dtype=torch.long,
        )
        phrase_ranges: list[tuple[int, int]] = []
        phrase_complete: list[bool] = []
        conductor_targets: dict[str, Tensor] = {}
        for global_phrase_id in local_phrase_order:
            full_start, full_end = sample["phrase_ranges"][global_phrase_id]
            clipped_start = max(start, full_start)
            clipped_end = min(end, full_end)
            phrase_ranges.append((clipped_start - start, clipped_end - start))
            phrase_complete.append(full_start >= start and full_end <= end)
        for target_name in CONDUCTOR_TARGET_NAMES:
            values = [
                int(sample["conductor_targets"][target_name][global_phrase_id].item())
                for global_phrase_id in local_phrase_order
            ]
            conductor_targets[target_name] = torch.tensor(values, dtype=torch.long)
        return {
            "piece_id": sample["piece_id"],
            "inputs": {
                feature: values[start:end]
                for feature, values in sample["inputs"].items()
            },
            "targets": {
                feature: values[start:end]
                for feature, values in sample["targets"].items()
            },
            "sequence_length": end - start,
            "phrase_boundaries": list(sample["phrase_boundaries"]),
            "metadata": dict(sample["metadata"]),
            "window_range": (start, end),
            "phrase_ids": remapped_phrase_ids,
            "phrase_ranges": phrase_ranges,
            "phrase_complete": phrase_complete,
            "conductor_targets": conductor_targets,
        }


def collate_autoregressive_batch(
    batch: list[dict[str, Any]],
    *,
    input_pad_value: int = 0,
    target_pad_value: int = -100,
) -> AutoregressiveBatch:
    """Pad a list of autoregressive samples into dense batch tensors."""
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    max_length = max(sample["sequence_length"] for sample in batch)
    max_length = max(1, max_length)
    max_phrases = max(1, max(len(sample["phrase_complete"]) for sample in batch))
    batch_size = len(batch)

    inputs = {
        feature: torch.full((batch_size, max_length), input_pad_value, dtype=torch.long)
        for feature in FEATURE_NAMES
    }
    targets = {
        feature: torch.full((batch_size, max_length), target_pad_value, dtype=torch.long)
        for feature in FEATURE_NAMES
    }
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    phrase_ids = torch.full((batch_size, max_length), -1, dtype=torch.long)
    phrase_mask = torch.zeros((batch_size, max_phrases), dtype=torch.bool)
    phrase_complete_mask = torch.zeros((batch_size, max_phrases), dtype=torch.bool)
    conductor_targets = {
        target_name: torch.full((batch_size, max_phrases), target_pad_value, dtype=torch.long)
        for target_name in CONDUCTOR_TARGET_NAMES
    }

    piece_ids: list[str] = []
    phrase_boundaries: list[list[int]] = []
    metadata: list[dict[str, Any]] = []
    window_ranges: list[tuple[int, int]] = []
    phrase_spans: list[list[tuple[int, int]]] = []
    for row, sample in enumerate(batch):
        length = sample["sequence_length"]
        lengths[row] = length
        attention_mask[row, :length] = True
        piece_ids.append(sample["piece_id"])
        phrase_boundaries.append(sample["phrase_boundaries"])
        metadata.append(sample["metadata"])
        window_ranges.append(sample.get("window_range", (0, length)))
        phrase_spans.append(list(sample.get("phrase_ranges", [])))
        if length > 0:
            phrase_ids[row, :length] = sample["phrase_ids"]
        phrase_count = len(sample["phrase_complete"])
        if phrase_count > 0:
            phrase_mask[row, :phrase_count] = True
            complete_mask = torch.tensor(sample["phrase_complete"], dtype=torch.bool)
            phrase_complete_mask[row, :phrase_count] = complete_mask
            for target_name in CONDUCTOR_TARGET_NAMES:
                target_values = sample["conductor_targets"][target_name]
                conductor_targets[target_name][row, :phrase_count] = target_values
                conductor_targets[target_name][row, :phrase_count][~complete_mask] = target_pad_value
        for feature in FEATURE_NAMES:
            inputs[feature][row, :length] = sample["inputs"][feature]
            targets[feature][row, :length] = sample["targets"][feature]

    return AutoregressiveBatch(
        piece_ids=piece_ids,
        inputs=inputs,
        targets=targets,
        attention_mask=attention_mask,
        lengths=lengths,
        phrase_boundaries=phrase_boundaries,
        metadata=metadata,
        window_ranges=window_ranges,
        phrase_ids=phrase_ids,
        phrase_mask=phrase_mask,
        phrase_complete_mask=phrase_complete_mask,
        conductor_targets=conductor_targets,
        phrase_spans=phrase_spans,
    )


def create_autoregressive_dataloader(
    dataset: Dataset[dict[str, Any]],
    *,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    """Create a DataLoader that pads grouped-token sequences batch-wise."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_autoregressive_batch,
    )


def build_weighted_window_sampler(
    dataset: WindowedAutoregressiveTokenDataset,
) -> WeightedRandomSampler | None:
    """Create a replacement sampler when window weights are meaningfully non-uniform."""
    if not dataset.window_weights:
        return None
    min_weight = min(dataset.window_weights)
    max_weight = max(dataset.window_weights)
    if abs(max_weight - min_weight) < 1e-6:
        return None
    return WeightedRandomSampler(
        weights=torch.tensor(dataset.window_weights, dtype=torch.double),
        num_samples=len(dataset.window_weights),
        replacement=True,
    )
