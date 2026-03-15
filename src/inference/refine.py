"""Apply the denoising refiner to generated symbolic piece directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from inference.generate import _decode_bar_position, _decode_velocity
from inference.render_midi import render_piece_to_midi
from preprocessing import QuantizedEvent, QuantizedPiece, load_quantized_piece_json, write_quantized_piece_json
from tokenization import encode_piece_to_blocks, example_to_feature_lists
from training.data import (
    FEATURE_NAMES,
    collate_autoregressive_batch,
    piece_example_to_autoregressive_sample,
)
from training.train_baseline import build_feature_vocab_sizes, load_config, move_batch_to_device, resolve_device
from training.train_refiner import (
    _clean_feature_targets,
    build_condition_state,
    build_refiner_model,
)
from training.train_torus import build_torus_model, maybe_initialize_from_checkpoint


PHRASE_FLAG_START_VALUES = {1, 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--primary-checkpoint", default=None)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preserve-prefix-events", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _discover_piece_paths(input_dir: Path) -> list[Path]:
    """Find generated piece JSON files inside an input directory."""
    ignored_names = {"manifest.json", "summary.json", "metrics.jsonl"}
    paths = sorted(
        path
        for path in input_dir.rglob("*.json")
        if path.name not in ignored_names
    )
    if not paths:
        raise FileNotFoundError(f"No JSON pieces found in {input_dir}")
    return paths


def _load_refiner_stack(
    checkpoint: str | Path,
    *,
    config_path: str | Path | None,
    primary_checkpoint: str | Path | None,
    device: torch.device,
) -> tuple[dict[str, Any], torch.nn.Module, torch.nn.Module]:
    """Load the primary model and the refiner checkpoint."""
    checkpoint_path = Path(checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = payload.get("config")
    if config is None:
        if config_path is None:
            raise ValueError("Config is required when the refiner checkpoint does not embed one.")
        config = load_config(config_path)

    vocab_sizes = build_feature_vocab_sizes(config["tokenization"])
    primary_model = build_torus_model(config, vocab_sizes=vocab_sizes).to(device)
    maybe_initialize_from_checkpoint(
        primary_model,
        primary_checkpoint or config["training"].get("init_checkpoint"),
        device=device,
    )
    primary_model.eval()
    for parameter in primary_model.parameters():
        parameter.requires_grad_(False)

    refiner_model = build_refiner_model(config, vocab_sizes=vocab_sizes)
    refiner_model.load_state_dict(payload["model_state"])
    refiner_model.to(device)
    refiner_model.eval()
    return config, primary_model, refiner_model


def _decode_feature_predictions(
    logits: dict[str, torch.Tensor],
    fallback_targets: dict[str, torch.Tensor],
) -> dict[str, list[int]]:
    """Convert refiner logits into raw grouped-token values."""
    decoded: dict[str, list[int]] = {}
    for feature in FEATURE_NAMES:
        feature_logits = logits[feature].clone()
        feature_logits[..., 0] = float("-inf")
        predicted_tokens = torch.argmax(feature_logits, dim=-1)
        predicted_tokens = torch.where(
            predicted_tokens > 0,
            predicted_tokens,
            fallback_targets[feature],
        )
        decoded[feature] = [
            max(0, int(value) - 1)
            for value in predicted_tokens[0].tolist()
        ]
    return decoded


def _reconstruct_piece(
    *,
    piece_id: str,
    features: dict[str, list[int]],
    template_piece: QuantizedPiece,
    velocity_bins: int,
    bar_position_bins: int,
) -> QuantizedPiece:
    """Rebuild a quantized piece from grouped-token feature columns."""
    current_bar = 1
    previous_position = None
    phrase_boundaries = [1]
    note_events: list[QuantizedEvent] = []
    for index, pitch in enumerate(features["pitch"]):
        bar_position = _decode_bar_position(
            features["bar_position"][index],
            bar_steps=template_piece.bar_steps,
            bar_position_bins=bar_position_bins,
        )
        if previous_position is not None and bar_position < previous_position:
            current_bar += 1
        previous_position = bar_position
        phrase_flag = features["phrase_flag"][index]
        if index > 0 and phrase_flag in PHRASE_FLAG_START_VALUES and current_bar not in phrase_boundaries:
            phrase_boundaries.append(current_bar)
        note_events.append(
            QuantizedEvent(
                pitch=max(0, min(127, pitch)),
                velocity=_decode_velocity(features["velocity"][index], velocity_bins),
                instrument=max(0, min(127, features["instrument"][index])),
                channel=0,
                start_step=(current_bar - 1) * template_piece.bar_steps + bar_position,
                duration_steps=max(1, features["duration"][index] + 1),
                bar=current_bar,
                position=bar_position,
                track_index=0,
                track_name="refined",
                is_drum=False,
                harmony="unknown",
            )
        )

    metadata = dict(template_piece.metadata)
    metadata.update(
        {
            "refined": True,
            "refined_from": template_piece.piece_id,
        }
    )
    return QuantizedPiece(
        piece_id=piece_id,
        resolution=template_piece.resolution,
        steps_per_beat=template_piece.steps_per_beat,
        bar_steps=template_piece.bar_steps,
        time_signature=template_piece.time_signature,
        tempo_bpm=template_piece.tempo_bpm,
        note_events=note_events,
        phrase_boundaries=sorted(set(phrase_boundaries)),
        source_path=template_piece.source_path,
        metadata=metadata,
        key=template_piece.key,
        chords=list(template_piece.chords),
    )


def refine_piece(
    draft_piece: QuantizedPiece,
    *,
    config: dict[str, Any],
    primary_model,
    refiner_model,
    device: torch.device,
    preserve_prefix_events: int = 0,
) -> QuantizedPiece:
    """Refine one generated draft piece."""
    example = encode_piece_to_blocks(
        draft_piece,
        duration_bins=config["tokenization"]["duration_bins"],
        velocity_bins=config["tokenization"]["velocity_bins"],
        bar_position_bins=config["tokenization"].get("bar_position_bins", 16),
    )
    if len(example.event_blocks) <= 1:
        return draft_piece

    original_features = example_to_feature_lists(example)
    sample = piece_example_to_autoregressive_sample(example)
    batch = collate_autoregressive_batch([sample])
    batch = move_batch_to_device(batch, device)
    clean_targets = _clean_feature_targets(batch.targets)

    with torch.no_grad():
        condition_state = build_condition_state(primary_model, batch)
        logits = refiner_model(
            clean_targets,
            batch.attention_mask,
            condition_state=condition_state,
        )

    predicted_targets = _decode_feature_predictions(logits, clean_targets)
    sequence_length = int(batch.lengths[0].item())
    full_features = {
        feature: [original_features[feature][0]] + predicted_targets[feature][:sequence_length]
        for feature in FEATURE_NAMES
    }
    preserve_count = min(preserve_prefix_events, len(example.event_blocks))
    if preserve_count > 0:
        for feature in FEATURE_NAMES:
            full_features[feature][:preserve_count] = original_features[feature][:preserve_count]
    return _reconstruct_piece(
        piece_id=f"{draft_piece.piece_id}__refined",
        features=full_features,
        template_piece=draft_piece,
        velocity_bins=config["tokenization"]["velocity_bins"],
        bar_position_bins=config["tokenization"].get("bar_position_bins", 16),
    )


def refine_directory(
    checkpoint: str | Path,
    *,
    config_path: str | Path | None = None,
    primary_checkpoint: str | Path | None = None,
    input_dir: str | Path,
    output_dir: str | Path,
    preserve_prefix_events: int = 0,
    device: str = "auto",
) -> dict[str, Any]:
    """Apply the refiner checkpoint to every piece in a generated directory."""
    run_device = resolve_device(device)
    config, primary_model, refiner_model = _load_refiner_stack(
        checkpoint,
        config_path=config_path,
        primary_checkpoint=primary_checkpoint,
        device=run_device,
    )

    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for piece_path in _discover_piece_paths(input_root):
        draft_piece = load_quantized_piece_json(piece_path)
        refined_piece = refine_piece(
            draft_piece,
            config=config,
            primary_model=primary_model,
            refiner_model=refiner_model,
            device=run_device,
            preserve_prefix_events=preserve_prefix_events,
        )
        relative_key = str(piece_path.relative_to(input_root).parent)
        piece_output_dir = output_root / relative_key
        piece_output_dir.mkdir(parents=True, exist_ok=True)
        output_json = write_quantized_piece_json(refined_piece, piece_output_dir / "piece.json")
        output_midi = render_piece_to_midi(refined_piece, piece_output_dir / "piece.mid")
        items.append(
            {
                "piece_id": relative_key,
                "source_json": str(piece_path),
                "output_json": str(output_json),
                "output_midi": str(output_midi),
                "generated_event_count": len(refined_piece.note_events),
            }
        )

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "piece_count": len(items),
                "items": items,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "checkpoint": str(checkpoint),
        "piece_count": len(items),
        "manifest_path": str(manifest_path),
        "items": items,
    }


def main() -> None:
    args = parse_args()
    result = refine_directory(
        args.checkpoint,
        config_path=args.config,
        primary_checkpoint=args.primary_checkpoint,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        preserve_prefix_events=args.preserve_prefix_events,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
