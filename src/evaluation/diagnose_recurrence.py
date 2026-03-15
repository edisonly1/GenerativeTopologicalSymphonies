"""Diagnostics for recurrence targets, phrase counts, and conductor predictions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from preprocessing import load_quantized_piece_json
from tokenization import encode_piece_to_blocks, load_piece_example, load_split_piece_ids
from training.conductor_targets import derive_phrase_control_targets
from training.data import piece_example_to_autoregressive_sample
from training.train_baseline import build_feature_vocab_sizes, load_config, resolve_device
from training.train_conductor import build_conductor_model
from training.train_torus import build_torus_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/conductor_long.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--splits", nargs="*", default=["train", "val"])
    parser.add_argument("--input-dirs", nargs="*", default=[])
    parser.add_argument("--limit-pieces", type=int, default=None)
    parser.add_argument("--duration-bins", type=int, default=None)
    parser.add_argument("--velocity-bins", type=int, default=None)
    parser.add_argument("--bar-position-bins", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _discover_piece_paths(input_dir: Path) -> list[Path]:
    """Find piece JSON files inside a directory tree."""
    ignored_names = {"manifest.json", "summary.json", "metrics.jsonl"}
    return sorted(
        path
        for path in input_dir.rglob("*.json")
        if path.name not in ignored_names
    )


def _load_example_from_json(
    path: Path,
    *,
    duration_bins: int,
    velocity_bins: int,
    bar_position_bins: int,
):
    """Load one tokenized example from a quantized-piece JSON file."""
    quantized_piece = load_quantized_piece_json(path)
    return encode_piece_to_blocks(
        quantized_piece,
        duration_bins=duration_bins,
        velocity_bins=velocity_bins,
        bar_position_bins=bar_position_bins,
    )


def _summarize_examples(examples, *, limit_pieces: int | None = None) -> dict[str, Any]:
    """Summarize phrase counts and recurrence-target prevalence for a set of examples."""
    if limit_pieces is not None:
        examples = examples[:limit_pieces]
    piece_count = len(examples)
    phrase_counts: list[int] = []
    recurrence_positive_total = 0
    recurrence_phrase_total = 0
    multi_phrase_piece_count = 0

    for example in examples:
        phrase_targets = derive_phrase_control_targets(example)
        phrase_count = len(phrase_targets.phrase_ranges)
        phrase_counts.append(phrase_count)
        if phrase_count > 1:
            multi_phrase_piece_count += 1
        recurrence_values = phrase_targets.targets["recurrence"]
        recurrence_positive_total += sum(recurrence_values)
        recurrence_phrase_total += len(recurrence_values)

    if piece_count == 0:
        return {
            "piece_count": 0,
            "mean_phrase_count": 0.0,
            "single_phrase_rate": 0.0,
            "multi_phrase_rate": 0.0,
            "recurrence_positive_rate": 0.0,
            "phrase_count_histogram": {},
        }

    histogram = Counter(phrase_counts)
    return {
        "piece_count": piece_count,
        "mean_phrase_count": sum(phrase_counts) / piece_count,
        "single_phrase_rate": sum(1 for count in phrase_counts if count <= 1) / piece_count,
        "multi_phrase_rate": multi_phrase_piece_count / piece_count,
        "recurrence_positive_rate": (
            recurrence_positive_total / recurrence_phrase_total if recurrence_phrase_total else 0.0
        ),
        "phrase_count_histogram": {str(key): value for key, value in sorted(histogram.items())},
    }


def summarize_split_targets(
    *,
    processed_dir: str | Path,
    splits_dir: str | Path,
    split: str,
    duration_bins: int,
    velocity_bins: int,
    bar_position_bins: int,
    limit_pieces: int | None = None,
) -> dict[str, Any]:
    """Summarize heuristic recurrence labels for one processed split."""
    processed_root = Path(processed_dir)
    piece_ids = load_split_piece_ids(splits_dir, split=split)
    if limit_pieces is not None:
        piece_ids = piece_ids[:limit_pieces]
    examples = [
        load_piece_example(
            processed_root / f"{piece_id}.json",
            duration_bins=duration_bins,
            velocity_bins=velocity_bins,
            bar_position_bins=bar_position_bins,
        )
        for piece_id in piece_ids
    ]
    return _summarize_examples(examples)


def summarize_piece_directory(
    input_dir: str | Path,
    *,
    duration_bins: int,
    velocity_bins: int,
    bar_position_bins: int,
    limit_pieces: int | None = None,
) -> dict[str, Any]:
    """Summarize phrase counts and heuristic recurrence labels for a piece directory."""
    piece_paths = _discover_piece_paths(Path(input_dir))
    if limit_pieces is not None:
        piece_paths = piece_paths[:limit_pieces]
    examples = [
        _load_example_from_json(
            path,
            duration_bins=duration_bins,
            velocity_bins=velocity_bins,
            bar_position_bins=bar_position_bins,
        )
        for path in piece_paths
    ]
    summary = _summarize_examples(examples)
    summary["input_dir"] = str(input_dir)
    return summary


def _sample_to_model_inputs(sample: dict[str, Any], device: torch.device) -> dict[str, Any] | None:
    """Convert one autoregressive sample into batch-of-one model inputs."""
    sequence_length = sample["sequence_length"]
    phrase_count = len(sample["phrase_ranges"])
    if sequence_length == 0 or phrase_count == 0:
        return None
    return {
        "inputs": {
            feature: values.unsqueeze(0).to(device)
            for feature, values in sample["inputs"].items()
        },
        "attention_mask": torch.ones((1, sequence_length), dtype=torch.bool, device=device),
        "phrase_ids": sample["phrase_ids"].unsqueeze(0).to(device),
        "phrase_mask": torch.ones((1, phrase_count), dtype=torch.bool, device=device),
        "recurrence_targets": sample["conductor_targets"]["recurrence"],
    }


def summarize_conductor_predictions(
    checkpoint: str | Path,
    *,
    config_path: str | Path,
    input_dirs: list[str],
    duration_bins: int,
    velocity_bins: int,
    bar_position_bins: int,
    device: str = "cpu",
    limit_pieces: int | None = None,
) -> dict[str, Any]:
    """Summarize recurrence-head predictions on one or more piece directories."""
    config = load_config(config_path)
    run_device = resolve_device(device)
    payload = torch.load(checkpoint, map_location=run_device)
    vocab_sizes = build_feature_vocab_sizes(config["tokenization"])
    if config["model"].get("use_torus", False):
        model = build_torus_model(config, vocab_sizes=vocab_sizes)
    else:
        model = build_conductor_model(config, vocab_sizes=vocab_sizes)
    model.load_state_dict(payload["model_state"])
    model.to(run_device)
    model.eval()

    diagnostics: dict[str, Any] = {}
    for input_dir in input_dirs:
        piece_paths = _discover_piece_paths(Path(input_dir))
        if limit_pieces is not None:
            piece_paths = piece_paths[:limit_pieces]
        probability_sum = 0.0
        predicted_positive_count = 0
        target_positive_count = 0
        phrase_total = 0
        evaluated_pieces = 0

        for path in piece_paths:
            example = _load_example_from_json(
                path,
                duration_bins=duration_bins,
                velocity_bins=velocity_bins,
                bar_position_bins=bar_position_bins,
            )
            sample = piece_example_to_autoregressive_sample(example)
            model_inputs = _sample_to_model_inputs(sample, run_device)
            if model_inputs is None:
                continue
            with torch.no_grad():
                output = model(
                    model_inputs["inputs"],
                    model_inputs["attention_mask"],
                    phrase_ids=model_inputs["phrase_ids"],
                    phrase_mask=model_inputs["phrase_mask"],
                )
            recurrence_logits = output.conductor_logits["recurrence"][0]
            recurrence_probs = torch.softmax(recurrence_logits, dim=-1)[:, 1].cpu()
            targets = model_inputs["recurrence_targets"]
            valid_mask = targets != -100
            if not valid_mask.any():
                continue
            valid_probs = recurrence_probs[valid_mask]
            valid_targets = targets[valid_mask]
            probability_sum += float(valid_probs.sum().item())
            predicted_positive_count += int((valid_probs >= 0.5).sum().item())
            target_positive_count += int((valid_targets == 1).sum().item())
            phrase_total += int(valid_mask.sum().item())
            evaluated_pieces += 1

        diagnostics[input_dir] = {
            "evaluated_pieces": evaluated_pieces,
            "phrase_total": phrase_total,
            "mean_recurrence_probability": probability_sum / phrase_total if phrase_total else 0.0,
            "predicted_positive_rate": predicted_positive_count / phrase_total if phrase_total else 0.0,
            "target_positive_rate": target_positive_count / phrase_total if phrase_total else 0.0,
        }
    return diagnostics


def run_recurrence_diagnostics(
    *,
    config_path: str | Path,
    checkpoint: str | Path | None = None,
    processed_dir: str | Path | None = None,
    splits_dir: str | Path | None = None,
    splits: list[str] | None = None,
    input_dirs: list[str] | None = None,
    limit_pieces: int | None = None,
    duration_bins: int | None = None,
    velocity_bins: int | None = None,
    bar_position_bins: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run recurrence diagnostics across dataset splits and piece directories."""
    config = load_config(config_path)
    processed_root = Path(processed_dir or config["data"]["processed_dir"])
    splits_root = Path(splits_dir or config["data"]["splits_dir"])
    active_splits = splits or ["train", "val"]
    duration_bins = duration_bins or config["tokenization"]["duration_bins"]
    velocity_bins = velocity_bins or config["tokenization"]["velocity_bins"]
    bar_position_bins = bar_position_bins or config["tokenization"].get("bar_position_bins", 16)
    active_input_dirs = input_dirs or []

    split_summaries = {
        split: summarize_split_targets(
            processed_dir=processed_root,
            splits_dir=splits_root,
            split=split,
            duration_bins=duration_bins,
            velocity_bins=velocity_bins,
            bar_position_bins=bar_position_bins,
            limit_pieces=limit_pieces,
        )
        for split in active_splits
    }
    piece_set_summaries = {
        input_dir: summarize_piece_directory(
            input_dir,
            duration_bins=duration_bins,
            velocity_bins=velocity_bins,
            bar_position_bins=bar_position_bins,
            limit_pieces=limit_pieces,
        )
        for input_dir in active_input_dirs
    }
    prediction_summaries = {}
    if checkpoint is not None and active_input_dirs:
        prediction_summaries = summarize_conductor_predictions(
            checkpoint,
            config_path=config_path,
            input_dirs=active_input_dirs,
            duration_bins=duration_bins,
            velocity_bins=velocity_bins,
            bar_position_bins=bar_position_bins,
            device=device,
            limit_pieces=limit_pieces,
        )

    return {
        "config_path": str(config_path),
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "split_summaries": split_summaries,
        "piece_set_summaries": piece_set_summaries,
        "prediction_summaries": prediction_summaries,
    }


def main() -> None:
    args = parse_args()
    result = run_recurrence_diagnostics(
        config_path=args.config,
        checkpoint=args.checkpoint,
        processed_dir=args.processed_dir,
        splits_dir=args.splits_dir,
        splits=args.splits,
        input_dirs=args.input_dirs,
        limit_pieces=args.limit_pieces,
        duration_bins=args.duration_bins,
        velocity_bins=args.velocity_bins,
        bar_position_bins=args.bar_position_bins,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
