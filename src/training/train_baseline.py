"""Train or dry-run the baseline grouped-token generator."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn.utils import clip_grad_norm_

from losses.reconstruction import grouped_reconstruction_loss
from models.decoder import BaselineDecoderConfig, BaselineGroupedDecoder
from tokenization import load_split_piece_ids
from training.data import (
    FEATURE_NAMES,
    AutoregressiveBatch,
    WindowedAutoregressiveTokenDataset,
    build_weighted_window_sampler,
    create_autoregressive_dataloader,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit-pieces", type=int, default=None)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve the requested training device."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def build_feature_vocab_sizes(tokenization_config: dict[str, Any]) -> dict[str, int]:
    """Create per-feature vocabulary sizes including the reserved padding id."""
    pitch_cardinality = tokenization_config.get("pitch_vocab_size", 128)
    instrument_cardinality = tokenization_config.get("instrument_bins", 129)
    harmony_cardinality = tokenization_config.get("harmony_bins", 1)
    phrase_flag_cardinality = tokenization_config.get("phrase_flag_bins", 4)
    return {
        "pitch": pitch_cardinality + 1,
        "duration": tokenization_config["duration_bins"] + 1,
        "velocity": tokenization_config["velocity_bins"] + 1,
        "bar_position": tokenization_config["bar_position_bins"] + 1,
        "instrument": instrument_cardinality + 1,
        "harmony": harmony_cardinality + 1,
        "phrase_flag": phrase_flag_cardinality + 1,
    }


def move_batch_to_device(batch: AutoregressiveBatch, device: torch.device) -> AutoregressiveBatch:
    """Move a collated batch to the target device."""
    return AutoregressiveBatch(
        piece_ids=batch.piece_ids,
        inputs={feature: values.to(device) for feature, values in batch.inputs.items()},
        targets={feature: values.to(device) for feature, values in batch.targets.items()},
        attention_mask=batch.attention_mask.to(device),
        lengths=batch.lengths.to(device),
        phrase_boundaries=batch.phrase_boundaries,
        metadata=batch.metadata,
        window_ranges=batch.window_ranges,
        phrase_ids=batch.phrase_ids.to(device),
        phrase_mask=batch.phrase_mask.to(device),
        phrase_complete_mask=batch.phrase_complete_mask.to(device),
        conductor_targets={
            target_name: values.to(device)
            for target_name, values in batch.conductor_targets.items()
        },
        phrase_spans=batch.phrase_spans,
    )


def build_datasets(
    config: dict[str, Any],
    *,
    limit_pieces: int | None = None,
) -> tuple[WindowedAutoregressiveTokenDataset, WindowedAutoregressiveTokenDataset]:
    """Build windowed train and validation datasets from config."""
    tokenization_config = config["tokenization"]
    data_config = config["data"]
    training_config = config["training"]
    common = dict(
        processed_dir=data_config["processed_dir"],
        splits_dir=data_config["splits_dir"],
        duration_bins=tokenization_config["duration_bins"],
        velocity_bins=tokenization_config["velocity_bins"],
        bar_position_bins=tokenization_config.get("bar_position_bins", 16),
        sequence_window=training_config.get("sequence_window", 512),
        sequence_hop=training_config.get("sequence_hop", training_config.get("sequence_window", 512)),
        min_sequence_length=training_config.get("min_sequence_length", 64),
        phrase_aligned_windows=training_config.get("phrase_aligned_windows", False),
        min_complete_phrases=training_config.get("min_complete_phrases", 0),
        min_distinct_phrases=training_config.get("min_distinct_phrases", 0),
        complete_phrase_weight=training_config.get("complete_phrase_weight", 0.0),
        distinct_phrase_weight=training_config.get("distinct_phrase_weight", 0.0),
        recurrence_boost=training_config.get("recurrence_boost", 0.0),
        cadence_boost=training_config.get("cadence_boost", 0.0),
        limit_pieces=limit_pieces,
        cache_examples=training_config.get("cache_examples", False),
    )
    train_common = dict(common)
    train_common.update(
        transpose_semitones=training_config.get("transpose_augmentation_semitones", []),
        transpose_probability=training_config.get("transpose_probability", 0.0),
        transpose_min_pitch=training_config.get("transpose_min_pitch", 0),
        transpose_max_pitch=training_config.get("transpose_max_pitch", 127),
    )
    val_common = dict(common)
    return (
        WindowedAutoregressiveTokenDataset(split="train", **train_common),
        WindowedAutoregressiveTokenDataset(split="val", **val_common),
    )


def build_model(config: dict[str, Any], *, vocab_sizes: dict[str, int]) -> BaselineGroupedDecoder:
    """Instantiate the baseline grouped-token decoder."""
    model_config = config["model"]
    decoder_config = BaselineDecoderConfig(
        d_model=model_config["d_model"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        dropout=model_config["dropout"],
        dim_feedforward=model_config.get("dim_feedforward", model_config["d_model"] * 4),
    )
    return BaselineGroupedDecoder(vocab_sizes=vocab_sizes, config=decoder_config)


def summarize_batch(batch: AutoregressiveBatch) -> dict[str, Any]:
    """Return a compact serializable summary for one batch."""
    return {
        "shape": tuple(batch.inputs["pitch"].shape),
        "lengths": batch.lengths.tolist(),
        "window_ranges": batch.window_ranges,
    }


def evaluate_model(
    model: BaselineGroupedDecoder,
    dataloader,
    *,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """Run validation and return the mean reconstruction loss."""
    model.eval()
    total_loss = 0.0
    batch_count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            logits = model(batch.inputs, batch.attention_mask)
            loss_output = grouped_reconstruction_loss(logits, batch.targets)
            total_loss += float(loss_output.total_loss.item())
            batch_count += 1
    if batch_count == 0:
        raise ValueError("Validation dataloader produced no batches.")
    return total_loss / batch_count


def append_metrics(path: Path, payload: dict[str, Any]) -> None:
    """Append one metrics record as JSONL."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def save_checkpoint(
    path: Path,
    *,
    model: BaselineGroupedDecoder,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val_loss: float,
    config: dict[str, Any],
) -> None:
    """Persist a training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "best_val_loss": best_val_loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def load_compatible_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, Any],
) -> tuple[Any, list[str]]:
    """Load only checkpoint tensors whose names and shapes still match."""
    model_state = model.state_dict()
    filtered_state = {}
    skipped_keys: list[str] = []
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if model_state[key].shape != value.shape:
            skipped_keys.append(key)
            continue
        filtered_state[key] = value
    load_result = model.load_state_dict(filtered_state, strict=False)
    return load_result, skipped_keys


def prepare_run_directory(config_path: Path, config: dict[str, Any]) -> tuple[Path, Path, Path]:
    """Create the run directory and persist a copy of the active config."""
    output_dir = Path(config["training"].get("output_dir", "outputs/runs/baseline"))
    checkpoints_dir = output_dir / "checkpoints"
    metrics_path = output_dir / "metrics.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output_dir / "source_config_path.txt").write_text(str(config_path), encoding="utf-8")
    return output_dir, checkpoints_dir, metrics_path


def run_baseline_training(
    config: dict[str, Any],
    *,
    config_path: str | Path = "configs/baseline.yaml",
    dry_run: bool = False,
    max_steps_override: int | None = None,
    limit_pieces: int | None = None,
) -> dict[str, Any]:
    """Run the baseline training pipeline or a dry-run validation pass."""
    set_seed(config["seed"])
    device = resolve_device(config["training"].get("device", "auto"))
    vocab_sizes = build_feature_vocab_sizes(config["tokenization"])
    train_dataset, val_dataset = build_datasets(config, limit_pieces=limit_pieces)
    if len(train_dataset) == 0:
        raise ValueError("Training dataset produced no windows.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset produced no windows.")

    training_config = config["training"]
    batch_size = training_config["batch_size"]
    train_sampler = build_weighted_window_sampler(train_dataset)
    train_loader = create_autoregressive_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=training_config.get("num_workers", 0),
    )
    val_loader = create_autoregressive_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=training_config.get("num_workers", 0),
    )
    model = build_model(config, vocab_sizes=vocab_sizes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 0.0),
    )

    if dry_run:
        train_batch = move_batch_to_device(next(iter(train_loader)), device)
        val_batch = move_batch_to_device(next(iter(val_loader)), device)
        with torch.no_grad():
            train_loss = grouped_reconstruction_loss(
                model(train_batch.inputs, train_batch.attention_mask),
                train_batch.targets,
            )
            val_loss = grouped_reconstruction_loss(
                model(val_batch.inputs, val_batch.attention_mask),
                val_batch.targets,
            )
        return {
            "mode": "dry_run",
            "device": str(device),
            "train_windows": len(train_dataset),
            "val_windows": len(val_dataset),
            "vocab_sizes": vocab_sizes,
            "train_batch": summarize_batch(train_batch),
            "val_batch": summarize_batch(val_batch),
            "train_loss": float(train_loss.total_loss.item()),
            "val_loss": float(val_loss.total_loss.item()),
        }

    config_path = Path(config_path)
    output_dir, checkpoints_dir, metrics_path = prepare_run_directory(config_path, config)
    max_steps = max_steps_override or training_config["max_steps"]
    validate_every = training_config.get("validate_every", 100)
    checkpoint_every = training_config.get("checkpoint_every", validate_every)
    log_every = training_config.get("log_every", 10)
    gradient_clip_norm = training_config.get("gradient_clip_norm", 1.0)
    eval_batches = training_config.get("eval_batches")
    best_val_loss = float("inf")

    train_iter = iter(train_loader)
    latest_train_loss = 0.0
    for step in range(1, max_steps + 1):
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.inputs, batch.attention_mask)
        loss_output = grouped_reconstruction_loss(logits, batch.targets)
        loss_output.total_loss.backward()
        clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        latest_train_loss = float(loss_output.total_loss.item())

        if step == 1 or step % log_every == 0:
            payload = {
                "step": step,
                "phase": "train",
                "loss": latest_train_loss,
                "batch_shape": tuple(batch.inputs["pitch"].shape),
                "mean_length": float(batch.lengths.float().mean().item()),
            }
            append_metrics(metrics_path, payload)
            print(
                f"[train] step={step} loss={latest_train_loss:.4f} "
                f"batch_shape={payload['batch_shape']} mean_len={payload['mean_length']:.1f}"
            )

        should_validate = step == 1 or step % validate_every == 0 or step == max_steps
        if should_validate:
            val_loss = evaluate_model(model, val_loader, device=device, max_batches=eval_batches)
            append_metrics(metrics_path, {"step": step, "phase": "val", "loss": val_loss})
            print(f"[val] step={step} loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    checkpoints_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    best_val_loss=best_val_loss,
                    config=config,
                )

        if step % checkpoint_every == 0 or step == max_steps:
            save_checkpoint(
                checkpoints_dir / f"step_{step:06d}.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                best_val_loss=best_val_loss,
                config=config,
            )

    save_checkpoint(
        output_dir / "latest.pt",
        model=model,
        optimizer=optimizer,
        step=max_steps,
        best_val_loss=best_val_loss,
        config=config,
    )
    return {
        "mode": "train",
        "device": str(device),
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "max_steps": max_steps,
        "best_val_loss": best_val_loss,
        "latest_train_loss": latest_train_loss,
        "output_dir": str(output_dir),
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    result = run_baseline_training(
        config,
        config_path=config_path,
        dry_run=args.dry_run,
        max_steps_override=args.max_steps,
        limit_pieces=args.limit_pieces,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
