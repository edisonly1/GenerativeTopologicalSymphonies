"""Checkpoint-based symbolic continuation generation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch

from inference.render_midi import render_piece_to_midi
from preprocessing import load_quantized_piece_json, write_quantized_piece_json
from tokenization import example_to_feature_lists, load_piece_example, load_split_piece_ids
from training.train_baseline import build_feature_vocab_sizes, build_model, load_config, resolve_device
from training.train_conductor import build_conductor_model
from training.train_diffusion_unet import build_diffusion_unet_model
from training.train_torus import build_torus_model
from training.train_vae import build_vae_model


PHRASE_FLAG_START_VALUES = {1, 3}
FEATURE_NAMES = (
    "pitch",
    "duration",
    "velocity",
    "bar_position",
    "instrument",
    "harmony",
    "phrase_flag",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--piece-id", default=None)
    parser.add_argument("--limit-pieces", type=int, default=1)
    parser.add_argument("--prompt-events", type=int, default=64)
    parser.add_argument("--generate-events", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _sample_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    """Sample one categorical token from a logit vector."""
    logits = logits.clone()
    logits[0] = float("-inf")
    if temperature <= 0.0:
        temperature = 1.0
    logits = logits / temperature

    if top_k > 0 and top_k < logits.numel():
        threshold = torch.topk(logits, top_k).values[-1]
        logits[logits < threshold] = float("-inf")

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove_mask = cumulative > top_p
        if remove_mask.any():
            remove_mask[1:] = remove_mask[:-1].clone()
            remove_mask[0] = False
            logits[sorted_indices[remove_mask]] = float("-inf")

    probabilities = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probabilities).any() or float(probabilities.sum()) == 0.0:
        return int(torch.argmax(logits).item())
    return int(torch.multinomial(probabilities, num_samples=1).item())


def _build_model_from_config(config: dict[str, Any], checkpoint_payload: dict[str, Any], device: torch.device):
    """Instantiate and load either the baseline or conductor model."""
    vocab_sizes = build_feature_vocab_sizes(config["tokenization"])
    architecture = config["model"].get("architecture", "decoder_transformer")
    if architecture == "vae_decoder":
        model = build_vae_model(config, vocab_sizes=vocab_sizes)
    elif architecture == "diffusion_unet":
        model = build_diffusion_unet_model(config, vocab_sizes=vocab_sizes)
    elif config["model"].get("use_torus", False):
        model = build_torus_model(config, vocab_sizes=vocab_sizes)
    elif config["model"].get("use_conductor", False):
        model = build_conductor_model(config, vocab_sizes=vocab_sizes)
    else:
        model = build_model(config, vocab_sizes=vocab_sizes)
    model.load_state_dict(checkpoint_payload["model_state"])
    model.to(device)
    model.eval()
    return model


def _initial_prompt_features(example, prompt_events: int) -> dict[str, list[int]]:
    """Extract the prompt prefix features from a tokenized example."""
    feature_lists = example_to_feature_lists(example)
    prefix_length = min(max(1, prompt_events), len(example.event_blocks))
    return {
        feature: feature_lists[feature][:prefix_length]
        for feature in FEATURE_NAMES
    }


def _derive_phrase_ids(phrase_flags: list[int]) -> list[int]:
    """Infer phrase ids from a generated phrase-flag stream."""
    phrase_ids: list[int] = []
    current_phrase = 0
    for index, phrase_flag in enumerate(phrase_flags):
        if index > 0 and phrase_flag in PHRASE_FLAG_START_VALUES:
            current_phrase += 1
        phrase_ids.append(current_phrase)
    return phrase_ids


def _prepare_model_inputs(
    features: dict[str, list[int]],
    device: torch.device,
    *,
    offset_tokens: bool = True,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert feature lists into batched tensors for inference."""
    sequence_length = len(features["pitch"])
    if sequence_length == 0:
        raise ValueError("Cannot generate from an empty prompt.")
    offset = 1 if offset_tokens else 0
    inputs = {
        feature: torch.tensor([values], dtype=torch.long, device=device) + offset
        for feature, values in features.items()
    }
    attention_mask = torch.ones((1, sequence_length), dtype=torch.bool, device=device)
    phrase_ids_list = _derive_phrase_ids(features["phrase_flag"])
    phrase_ids = torch.tensor([phrase_ids_list], dtype=torch.long, device=device)
    phrase_count = max(phrase_ids_list) + 1
    phrase_mask = torch.ones((1, phrase_count), dtype=torch.bool, device=device)
    return inputs, attention_mask, phrase_ids, phrase_mask


def _generate_next_feature_values(
    model,
    config: dict[str, Any],
    features: dict[str, list[int]],
    *,
    device: torch.device,
    temperature: float,
    top_k: int,
    top_p: float,
) -> dict[str, int]:
    """Run one autoregressive generation step."""
    inputs, attention_mask, phrase_ids, phrase_mask = _prepare_model_inputs(features, device)
    with torch.no_grad():
        if config["model"].get("use_torus", False) or config["model"].get("use_conductor", False):
            output = model(
                inputs,
                attention_mask,
                phrase_ids=phrase_ids,
                phrase_mask=phrase_mask,
            )
            logits = output.token_logits
        else:
            output = model(inputs, attention_mask)
            logits = output.token_logits if hasattr(output, "token_logits") else output
    return {
        feature: _sample_from_logits(
            logits[feature][0, -1],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        ) - 1
        for feature in FEATURE_NAMES
    }


def _iterative_denoise_feature_values(
    model,
    features: dict[str, list[int]],
    *,
    device: torch.device,
    temperature: float,
    top_k: int,
    top_p: float,
    denoising_steps: int,
    prompt_length: int,
) -> dict[str, list[int]]:
    """Iteratively fill masked continuation positions for denoising-style models."""
    current = {
        feature: list(values)
        for feature, values in features.items()
    }
    for _ in range(max(denoising_steps, 1)):
        inputs, attention_mask, _, _ = _prepare_model_inputs(current, device)
        with torch.no_grad():
            output = model(inputs, attention_mask)
            logits = output.token_logits if hasattr(output, "token_logits") else output
        for position in range(prompt_length, len(current["pitch"])):
            for feature in FEATURE_NAMES:
                current[feature][position] = _sample_from_logits(
                    logits[feature][0, position],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                ) - 1
    return current


def _decode_velocity(bucket: int, velocity_bins: int) -> int:
    """Convert a velocity bucket back to a MIDI velocity."""
    if velocity_bins <= 0:
        return 64
    step = 128 / velocity_bins
    return max(1, min(127, int(round((bucket + 0.5) * step))))


def _decode_bar_position(bucket: int, *, bar_steps: int, bar_position_bins: int) -> int:
    """Convert a relative-position bucket back into an onset step within the bar."""
    if bar_steps <= 1 or bar_position_bins <= 1:
        return 0
    clamped_bucket = max(0, min(bar_position_bins - 1, bucket))
    if bar_position_bins >= bar_steps:
        return min(bar_steps - 1, clamped_bucket)
    step = bar_steps / bar_position_bins
    return max(0, min(bar_steps - 1, int(round((clamped_bucket + 0.5) * step - 0.5))))


def _reconstruct_quantized_piece(
    *,
    piece_id: str,
    generated_features: dict[str, list[int]],
    prompt_piece,
    velocity_bins: int,
    bar_position_bins: int,
):
    """Rebuild a QuantizedPiece from generated grouped event factors."""
    from preprocessing.schema import QuantizedEvent, QuantizedPiece

    bar_steps = prompt_piece.bar_steps
    current_bar = 1
    previous_position = None
    phrase_boundaries = [1]
    note_events: list[QuantizedEvent] = []

    for index, pitch in enumerate(generated_features["pitch"]):
        bar_position = _decode_bar_position(
            generated_features["bar_position"][index],
            bar_steps=bar_steps,
            bar_position_bins=bar_position_bins,
        )
        if previous_position is not None and bar_position < previous_position:
            current_bar += 1
        previous_position = bar_position
        phrase_flag = generated_features["phrase_flag"][index]
        if index > 0 and phrase_flag in PHRASE_FLAG_START_VALUES and current_bar not in phrase_boundaries:
            phrase_boundaries.append(current_bar)

        start_step = (current_bar - 1) * bar_steps + bar_position
        duration_steps = max(1, generated_features["duration"][index] + 1)
        instrument = max(0, min(127, generated_features["instrument"][index]))
        note_events.append(
            QuantizedEvent(
                pitch=max(0, min(127, pitch)),
                velocity=_decode_velocity(
                    generated_features["velocity"][index],
                    velocity_bins=velocity_bins,
                ),
                instrument=instrument,
                channel=0,
                start_step=start_step,
                duration_steps=duration_steps,
                bar=current_bar,
                position=bar_position,
                track_index=0,
                track_name="generated",
                is_drum=False,
                harmony="unknown",
            )
        )

    return QuantizedPiece(
        piece_id=piece_id,
        resolution=prompt_piece.resolution,
        steps_per_beat=prompt_piece.steps_per_beat,
        bar_steps=prompt_piece.bar_steps,
        time_signature=prompt_piece.time_signature,
        tempo_bpm=prompt_piece.tempo_bpm,
        note_events=note_events,
        phrase_boundaries=sorted(set(phrase_boundaries)),
        source_path=None,
        metadata={"prompt_piece_id": prompt_piece.piece_id, "generated": True},
        key=prompt_piece.key,
        chords=[],
    )


def generate_piece_continuation(
    *,
    model,
    config: dict[str, Any],
    prompt_example,
    prompt_piece,
    prompt_events: int,
    generate_events: int,
    device: torch.device,
    temperature: float,
    top_k: int,
    top_p: float,
    output_piece_id: str,
) -> tuple[dict[str, list[int]], Any]:
    """Generate a continuation from a prompt prefix."""
    architecture = config["model"].get("architecture", "decoder_transformer")
    features = _initial_prompt_features(prompt_example, prompt_events)
    if architecture == "diffusion_unet":
        for feature in FEATURE_NAMES:
            features[feature].extend([-1] * generate_events)
        features = _iterative_denoise_feature_values(
            model,
            features,
            device=device,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            denoising_steps=config.get("generation", {}).get("denoising_steps", 6),
            prompt_length=len(features["pitch"]) - generate_events,
        )
    else:
        for _ in range(generate_events):
            next_values = _generate_next_feature_values(
                model,
                config,
                features,
                device=device,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            for feature, value in next_values.items():
                features[feature].append(value)

    generated_piece = _reconstruct_quantized_piece(
        piece_id=output_piece_id,
        generated_features=features,
        prompt_piece=prompt_piece,
        velocity_bins=config["tokenization"]["velocity_bins"],
        bar_position_bins=config["tokenization"].get("bar_position_bins", 16),
    )
    return features, generated_piece


def generate_from_checkpoint(
    checkpoint: str | Path,
    *,
    config_path: str | Path | None = None,
    processed_dir: str | Path | None = None,
    splits_dir: str | Path | None = None,
    split: str = "val",
    piece_id: str | None = None,
    limit_pieces: int = 1,
    prompt_events: int = 64,
    generate_events: int = 128,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    device: str = "auto",
    output_dir: str | Path = "outputs/generated",
    seed: int = 42,
) -> dict[str, Any]:
    """Generate symbolic continuations from one checkpoint."""
    random.seed(seed)
    torch.manual_seed(seed)

    checkpoint_path = Path(checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = payload.get("config")
    if config is None:
        if config_path is None:
            raise ValueError("Config is required when checkpoint does not contain one.")
        config = load_config(config_path)

    processed_root = Path(processed_dir or config["data"]["processed_dir"])
    splits_root = Path(splits_dir or config["data"]["splits_dir"])
    generation_config = config.get("generation", {})
    sampling_temperature = temperature if temperature is not None else generation_config.get("temperature", 1.0)
    sampling_top_k = top_k if top_k is not None else generation_config.get("top_k", 0)
    sampling_top_p = top_p if top_p is not None else generation_config.get("top_p", 1.0)
    run_device = resolve_device(device)
    model = _build_model_from_config(config, payload, run_device)

    if piece_id is not None:
        piece_ids = [piece_id]
    else:
        piece_ids = load_split_piece_ids(splits_root, split=split)[:limit_pieces]
    if not piece_ids:
        raise ValueError("No piece ids selected for generation.")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_items = []
    for selected_piece_id in piece_ids:
        prompt_path = processed_root / f"{selected_piece_id}.json"
        prompt_piece = load_quantized_piece_json(prompt_path)
        prompt_example = load_piece_example(
            prompt_path,
            duration_bins=config["tokenization"]["duration_bins"],
            velocity_bins=config["tokenization"]["velocity_bins"],
            bar_position_bins=config["tokenization"].get("bar_position_bins", 16),
        )
        _, generated_piece = generate_piece_continuation(
            model=model,
            config=config,
            prompt_example=prompt_example,
            prompt_piece=prompt_piece,
            prompt_events=prompt_events,
            generate_events=generate_events,
            device=run_device,
            temperature=sampling_temperature,
            top_k=sampling_top_k,
            top_p=sampling_top_p,
            output_piece_id=f"{selected_piece_id}__sample",
        )
        piece_output_dir = output_root / selected_piece_id
        piece_output_dir.mkdir(parents=True, exist_ok=True)
        json_path = write_quantized_piece_json(generated_piece, piece_output_dir / "piece.json")
        midi_path = render_piece_to_midi(generated_piece, piece_output_dir / "piece.mid")
        manifest_items.append(
            {
                "piece_id": selected_piece_id,
                "output_json": str(json_path),
                "output_midi": str(midi_path),
                "generated_event_count": len(generated_piece.note_events),
            }
        )

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "piece_count": len(manifest_items),
                "items": manifest_items,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "checkpoint": str(checkpoint_path),
        "piece_count": len(manifest_items),
        "manifest_path": str(manifest_path),
        "items": manifest_items,
    }


def main() -> None:
    args = parse_args()
    result = generate_from_checkpoint(
        args.checkpoint,
        config_path=args.config,
        processed_dir=args.processed_dir,
        splits_dir=args.splits_dir,
        split=args.split,
        piece_id=args.piece_id,
        limit_pieces=args.limit_pieces,
        prompt_events=args.prompt_events,
        generate_events=args.generate_events,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
