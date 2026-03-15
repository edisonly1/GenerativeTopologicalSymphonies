"""Train or dry-run the conditional denoising refiner stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from losses.refiner import masked_grouped_reconstruction_loss
from models.conductor import broadcast_phrase_states
from models.refiner import ConditionalDenoisingRefiner, RefinerConfig
from training.data import FEATURE_NAMES, create_autoregressive_dataloader
from training.train_baseline import (
    append_metrics,
    build_datasets,
    build_feature_vocab_sizes,
    load_config,
    move_batch_to_device,
    prepare_run_directory,
    resolve_device,
    save_checkpoint,
    set_seed,
    summarize_batch,
)
from training.train_torus import build_torus_model, maybe_initialize_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/refiner.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit-pieces", type=int, default=None)
    return parser.parse_args()


def build_refiner_model(
    config: dict[str, Any],
    *,
    vocab_sizes: dict[str, int],
) -> ConditionalDenoisingRefiner:
    """Instantiate the denoising refiner."""
    model_config = config["model"]
    refiner_config = RefinerConfig(
        d_model=model_config["d_model"],
        num_layers=model_config.get("refiner_layers", model_config["num_layers"]),
        num_heads=model_config.get("refiner_heads", model_config["num_heads"]),
        dropout=model_config.get("refiner_dropout", model_config["dropout"]),
        dim_feedforward=model_config.get(
            "refiner_dim_feedforward",
            model_config.get("dim_feedforward", model_config["d_model"] * 4),
        ),
    )
    return ConditionalDenoisingRefiner(vocab_sizes=vocab_sizes, config=refiner_config)


def freeze_module(module: torch.nn.Module) -> None:
    """Freeze a module for teacher-forced conditioning."""
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _clean_feature_targets(targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert target tensors into refiner-input tokens with pad=0."""
    return {
        feature: torch.where(values >= 0, values, torch.zeros_like(values))
        for feature, values in targets.items()
    }


def corrupt_grouped_inputs(
    clean_targets: dict[str, torch.Tensor],
    attention_mask: torch.Tensor,
    *,
    vocab_sizes: dict[str, int],
    token_mask_prob: float,
    pitch_shift_prob: float,
    duration_shift_prob: float,
    phrase_flag_flip_prob: float,
    bar_position_jitter_prob: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Apply symbolic corruption to refiner inputs and return corruption masks."""
    corrupted = {
        feature: values.clone()
        for feature, values in clean_targets.items()
    }
    corruption_masks = {
        feature: torch.zeros_like(values, dtype=torch.bool)
        for feature, values in clean_targets.items()
    }

    valid_masks = {
        feature: attention_mask & (clean_targets[feature] > 0)
        for feature in FEATURE_NAMES
    }

    for feature in FEATURE_NAMES:
        if token_mask_prob > 0.0:
            mask = (torch.rand_like(clean_targets[feature].float()) < token_mask_prob) & valid_masks[feature]
            corrupted[feature][mask] = 0
            corruption_masks[feature] |= mask

    if pitch_shift_prob > 0.0:
        mask = (torch.rand_like(clean_targets["pitch"].float()) < pitch_shift_prob) & valid_masks["pitch"]
        direction = torch.randint(0, 2, clean_targets["pitch"].shape, device=clean_targets["pitch"].device)
        offset = torch.where(direction == 0, -1, 1)
        shifted = (clean_targets["pitch"] + offset).clamp(min=1, max=vocab_sizes["pitch"] - 1)
        corrupted["pitch"][mask] = shifted[mask]
        corruption_masks["pitch"] |= mask

    if duration_shift_prob > 0.0:
        for feature in ("duration", "velocity"):
            mask = (torch.rand_like(clean_targets[feature].float()) < duration_shift_prob) & valid_masks[feature]
            direction = torch.randint(0, 2, clean_targets[feature].shape, device=clean_targets[feature].device)
            offset = torch.where(direction == 0, -1, 1)
            shifted = (clean_targets[feature] + offset).clamp(min=1, max=vocab_sizes[feature] - 1)
            corrupted[feature][mask] = shifted[mask]
            corruption_masks[feature] |= mask

    if phrase_flag_flip_prob > 0.0:
        mask = (
            torch.rand_like(clean_targets["phrase_flag"].float()) < phrase_flag_flip_prob
        ) & valid_masks["phrase_flag"]
        corrupted["phrase_flag"][mask] = 1
        corruption_masks["phrase_flag"] |= mask

    if bar_position_jitter_prob > 0.0:
        mask = (
            torch.rand_like(clean_targets["bar_position"].float()) < bar_position_jitter_prob
        ) & valid_masks["bar_position"]
        direction = torch.randint(
            0,
            2,
            clean_targets["bar_position"].shape,
            device=clean_targets["bar_position"].device,
        )
        offset = torch.where(direction == 0, -1, 1)
        shifted = (clean_targets["bar_position"] + offset).clamp(
            min=1,
            max=vocab_sizes["bar_position"] - 1,
        )
        corrupted["bar_position"][mask] = shifted[mask]
        corruption_masks["bar_position"] |= mask

    return corrupted, corruption_masks


def build_condition_state(primary_model, batch) -> torch.Tensor:
    """Build frozen phrase-plan and torus conditioning for the refiner."""
    with torch.no_grad():
        primary_output = primary_model(
            batch.inputs,
            batch.attention_mask,
            phrase_ids=batch.phrase_ids,
            phrase_mask=batch.phrase_mask,
        )
        phrase_condition = primary_output.control_state + primary_output.torus_embedding
        return broadcast_phrase_states(
            phrase_condition,
            batch.phrase_ids,
            attention_mask=batch.attention_mask,
        )


def compute_refiner_loss(logits, batch, corruption_masks) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute masked denoising reconstruction loss."""
    loss_output = masked_grouped_reconstruction_loss(logits, batch.targets, corruption_masks)
    total_feature_positions = max(int(batch.attention_mask.sum().item()) * len(FEATURE_NAMES), 1)
    metrics = {
        "loss": float(loss_output.total_loss.item()),
        "corrupted_token_count": float(loss_output.corrupted_token_count),
        "mean_corrupted_per_feature": loss_output.corrupted_token_count / max(len(FEATURE_NAMES), 1),
        "corrupted_fraction": float(loss_output.corrupted_token_count / total_feature_positions),
    }
    return loss_output.total_loss, metrics


def evaluate_model(
    refiner_model: ConditionalDenoisingRefiner,
    primary_model: torch.nn.Module,
    dataloader,
    *,
    device: torch.device,
    vocab_sizes: dict[str, int],
    corruption_config: dict[str, float],
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run validation and return mean refiner metrics."""
    refiner_model.eval()
    totals = {
        "loss": 0.0,
        "corrupted_token_count": 0.0,
        "mean_corrupted_per_feature": 0.0,
        "corrupted_fraction": 0.0,
    }
    batch_count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            clean_targets = _clean_feature_targets(batch.targets)
            corrupted_inputs, corruption_masks = corrupt_grouped_inputs(
                clean_targets,
                batch.attention_mask,
                vocab_sizes=vocab_sizes,
                **corruption_config,
            )
            condition_state = build_condition_state(primary_model, batch)
            logits = refiner_model(
                corrupted_inputs,
                batch.attention_mask,
                condition_state=condition_state,
            )
            _, metrics = compute_refiner_loss(logits, batch, corruption_masks)
            for key in totals:
                totals[key] += metrics[key]
            batch_count += 1
    if batch_count == 0:
        raise ValueError("Validation dataloader produced no batches.")
    return {key: value / batch_count for key, value in totals.items()}


def run_refiner_training(
    config: dict[str, Any],
    *,
    config_path: str | Path = "configs/refiner.yaml",
    dry_run: bool = False,
    max_steps_override: int | None = None,
    limit_pieces: int | None = None,
) -> dict[str, Any]:
    """Run the denoising refiner stage."""
    set_seed(config["seed"])
    training_config = config["training"]
    corruption_config = training_config.get("corruption", {})
    active_corruption = {
        "token_mask_prob": corruption_config.get("token_mask_prob", 0.15),
        "pitch_shift_prob": corruption_config.get("pitch_shift_prob", 0.10),
        "duration_shift_prob": corruption_config.get("duration_shift_prob", 0.10),
        "phrase_flag_flip_prob": corruption_config.get("phrase_flag_flip_prob", 0.10),
        "bar_position_jitter_prob": corruption_config.get("bar_position_jitter_prob", 0.10),
    }

    device = resolve_device(training_config.get("device", "auto"))
    vocab_sizes = build_feature_vocab_sizes(config["tokenization"])
    train_dataset, val_dataset = build_datasets(config, limit_pieces=limit_pieces)
    if len(train_dataset) == 0:
        raise ValueError("Training dataset produced no windows.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset produced no windows.")

    batch_size = training_config["batch_size"]
    train_loader = create_autoregressive_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=training_config.get("num_workers", 0),
    )
    val_loader = create_autoregressive_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=training_config.get("num_workers", 0),
    )

    primary_model = build_torus_model(config, vocab_sizes=vocab_sizes).to(device)
    initialization = maybe_initialize_from_checkpoint(
        primary_model,
        training_config.get("init_checkpoint"),
        device=device,
    )
    freeze_module(primary_model)

    refiner_model = build_refiner_model(config, vocab_sizes=vocab_sizes).to(device)
    optimizer = torch.optim.AdamW(
        refiner_model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 0.0),
    )

    if dry_run:
        batch = move_batch_to_device(next(iter(train_loader)), device)
        val_batch = move_batch_to_device(next(iter(val_loader)), device)
        with torch.no_grad():
            clean_targets = _clean_feature_targets(batch.targets)
            corrupted_inputs, corruption_masks = corrupt_grouped_inputs(
                clean_targets,
                batch.attention_mask,
                vocab_sizes=vocab_sizes,
                **active_corruption,
            )
            condition_state = build_condition_state(primary_model, batch)
            train_logits = refiner_model(
                corrupted_inputs,
                batch.attention_mask,
                condition_state=condition_state,
            )
            _, train_metrics = compute_refiner_loss(train_logits, batch, corruption_masks)

            val_clean_targets = _clean_feature_targets(val_batch.targets)
            val_corrupted_inputs, val_corruption_masks = corrupt_grouped_inputs(
                val_clean_targets,
                val_batch.attention_mask,
                vocab_sizes=vocab_sizes,
                **active_corruption,
            )
            val_condition_state = build_condition_state(primary_model, val_batch)
            val_logits = refiner_model(
                val_corrupted_inputs,
                val_batch.attention_mask,
                condition_state=val_condition_state,
            )
            _, val_metrics = compute_refiner_loss(val_logits, val_batch, val_corruption_masks)
        return {
            "mode": "dry_run",
            "device": str(device),
            "train_windows": len(train_dataset),
            "val_windows": len(val_dataset),
            "vocab_sizes": vocab_sizes,
            "initialization": initialization,
            "train_batch": summarize_batch(batch),
            "val_batch": summarize_batch(val_batch),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
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
    latest_metrics = {
        "loss": 0.0,
        "corrupted_token_count": 0.0,
        "mean_corrupted_per_feature": 0.0,
        "corrupted_fraction": 0.0,
    }
    for step in range(1, max_steps + 1):
        refiner_model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = move_batch_to_device(batch, device)
        clean_targets = _clean_feature_targets(batch.targets)
        corrupted_inputs, corruption_masks = corrupt_grouped_inputs(
            clean_targets,
            batch.attention_mask,
            vocab_sizes=vocab_sizes,
            **active_corruption,
        )
        condition_state = build_condition_state(primary_model, batch)
        optimizer.zero_grad(set_to_none=True)
        logits = refiner_model(
            corrupted_inputs,
            batch.attention_mask,
            condition_state=condition_state,
        )
        total_loss, latest_metrics = compute_refiner_loss(logits, batch, corruption_masks)
        total_loss.backward()
        clip_grad_norm_(refiner_model.parameters(), gradient_clip_norm)
        optimizer.step()

        if step == 1 or step % log_every == 0:
            payload = {
                "step": step,
                "phase": "train",
                **latest_metrics,
                "batch_shape": tuple(batch.targets["pitch"].shape),
                "mean_length": float(batch.lengths.float().mean().item()),
            }
            append_metrics(metrics_path, payload)
            print(
                f"[train] step={step} loss={payload['loss']:.4f} "
                f"corrupted={payload['corrupted_fraction']:.4f} "
                f"batch_shape={payload['batch_shape']} mean_len={payload['mean_length']:.1f}"
            )

        should_validate = step == 1 or step % validate_every == 0 or step == max_steps
        if should_validate:
            val_metrics = evaluate_model(
                refiner_model,
                primary_model,
                val_loader,
                device=device,
                vocab_sizes=vocab_sizes,
                corruption_config=active_corruption,
                max_batches=eval_batches,
            )
            append_metrics(metrics_path, {"step": step, "phase": "val", **val_metrics})
            print(
                f"[val] step={step} loss={val_metrics['loss']:.4f} "
                f"corrupted={val_metrics['corrupted_fraction']:.4f}"
            )
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(
                    checkpoints_dir / "best.pt",
                    model=refiner_model,
                    optimizer=optimizer,
                    step=step,
                    best_val_loss=best_val_loss,
                    config=config,
                )

        if step % checkpoint_every == 0 or step == max_steps:
            save_checkpoint(
                checkpoints_dir / f"step_{step:06d}.pt",
                model=refiner_model,
                optimizer=optimizer,
                step=step,
                best_val_loss=best_val_loss,
                config=config,
            )

    save_checkpoint(
        output_dir / "latest.pt",
        model=refiner_model,
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
        "latest_metrics": latest_metrics,
        "initialization": initialization,
        "output_dir": str(output_dir),
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    result = run_refiner_training(
        config,
        config_path=config_path,
        dry_run=args.dry_run,
        max_steps_override=args.max_steps,
        limit_pieces=args.limit_pieces,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
