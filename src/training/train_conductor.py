"""Train or dry-run the phrase-level conductor stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from losses.conductor import conductor_supervision_loss
from losses.motif import motif_recurrence_loss
from losses.phrase_boundary import phrase_boundary_loss
from losses.reconstruction import grouped_reconstruction_loss
from models.conductor import (
    ConductorConditionedGroupedDecoder,
    PhraseConductorConfig,
)
from models.decoder import BaselineDecoderConfig
from training.conductor_targets import DEFAULT_CONDUCTOR_TARGET_VOCAB_SIZES
from training.train_baseline import (
    append_metrics,
    build_datasets,
    build_feature_vocab_sizes,
    load_compatible_state_dict,
    load_config,
    move_batch_to_device,
    prepare_run_directory,
    resolve_device,
    save_checkpoint,
    set_seed,
    summarize_batch,
)
from training.data import build_weighted_window_sampler, create_autoregressive_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/conductor.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit-pieces", type=int, default=None)
    parser.add_argument("--conductor-only", action="store_true")
    return parser.parse_args()


def build_conductor_model(
    config: dict[str, Any],
    *,
    vocab_sizes: dict[str, int],
) -> ConductorConditionedGroupedDecoder:
    """Instantiate the conductor-conditioned decoder stage."""
    model_config = config["model"]
    decoder_config = BaselineDecoderConfig(
        d_model=model_config["d_model"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        dropout=model_config["dropout"],
        dim_feedforward=model_config.get("dim_feedforward", model_config["d_model"] * 4),
    )
    conductor_config = PhraseConductorConfig(
        d_model=model_config["d_model"],
        num_layers=model_config.get("conductor_layers", 2),
        num_heads=model_config.get("conductor_heads", model_config["num_heads"]),
        dropout=model_config.get("conductor_dropout", model_config["dropout"]),
        dim_feedforward=model_config.get(
            "conductor_dim_feedforward",
            model_config.get("dim_feedforward", model_config["d_model"] * 2),
        ),
    )
    target_vocab_sizes = DEFAULT_CONDUCTOR_TARGET_VOCAB_SIZES.copy()
    target_vocab_sizes.update(config.get("conductor_targets", {}))
    return ConductorConditionedGroupedDecoder(
        vocab_sizes=vocab_sizes,
        decoder_config=decoder_config,
        conductor_config=conductor_config,
        target_vocab_sizes=target_vocab_sizes,
    )


def maybe_initialize_from_checkpoint(
    model: ConductorConditionedGroupedDecoder,
    checkpoint_path: str | Path | None,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Optionally initialize shared decoder weights from an earlier checkpoint."""
    if checkpoint_path is None:
        return {"initialized_from": None, "missing_keys": [], "unexpected_keys": []}
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Initialization checkpoint not found: {path}")
    payload = torch.load(path, map_location=device)
    state_dict = payload.get("model_state", payload)
    load_result, skipped_keys = load_compatible_state_dict(model, state_dict)
    return {
        "initialized_from": str(path),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "skipped_mismatched_keys": skipped_keys,
    }


def compute_total_loss(
    model_output,
    batch,
    *,
    reconstruction_weight: float,
    conductor_weight: float,
    motif_weight: float,
    phrase_boundary_weight: float,
    phrase_boundary_class_weights: list[float] | None,
    conductor_target_weights: dict[str, float] | None,
    conductor_only: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine conductor and reconstruction losses for the current stage."""
    conductor_loss = conductor_supervision_loss(
        model_output.conductor_logits,
        batch.conductor_targets,
        target_weights=conductor_target_weights,
    )
    motif_loss = motif_recurrence_loss(
        model_output.phrase_hidden,
        batch.conductor_targets["recurrence"],
    )
    metrics = {
        "conductor_loss": float(conductor_loss.total_loss.item()),
        "motif_loss": float(motif_loss.total_loss.item()),
    }
    total_loss = conductor_weight * conductor_loss.total_loss + motif_weight * motif_loss.total_loss
    if not conductor_only:
        boundary_loss = phrase_boundary_loss(
            model_output.token_logits["phrase_flag"],
            batch.targets["phrase_flag"],
            class_weights=phrase_boundary_class_weights,
        )
        reconstruction_loss = grouped_reconstruction_loss(
            model_output.token_logits,
            batch.targets,
        )
        metrics["boundary_loss"] = float(boundary_loss.total_loss.item())
        metrics["reconstruction_loss"] = float(reconstruction_loss.total_loss.item())
        total_loss = total_loss + reconstruction_weight * reconstruction_loss.total_loss
        total_loss = total_loss + phrase_boundary_weight * boundary_loss.total_loss
    else:
        metrics["boundary_loss"] = 0.0
        metrics["reconstruction_loss"] = 0.0
    metrics["loss"] = float(total_loss.item())
    return total_loss, metrics


def evaluate_model(
    model: ConductorConditionedGroupedDecoder,
    dataloader,
    *,
    device: torch.device,
    reconstruction_weight: float,
    conductor_weight: float,
    motif_weight: float,
    phrase_boundary_weight: float,
    phrase_boundary_class_weights: list[float] | None,
    conductor_target_weights: dict[str, float] | None,
    conductor_only: bool,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run validation and return mean total/conductor/reconstruction losses."""
    model.eval()
    totals = {
        "loss": 0.0,
        "conductor_loss": 0.0,
        "motif_loss": 0.0,
        "boundary_loss": 0.0,
        "reconstruction_loss": 0.0,
    }
    batch_count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            output = model(
                batch.inputs,
                batch.attention_mask,
                phrase_ids=batch.phrase_ids,
                phrase_mask=batch.phrase_mask,
            )
            _, metrics = compute_total_loss(
                output,
                batch,
                reconstruction_weight=reconstruction_weight,
                conductor_weight=conductor_weight,
                motif_weight=motif_weight,
                phrase_boundary_weight=phrase_boundary_weight,
                phrase_boundary_class_weights=phrase_boundary_class_weights,
                conductor_target_weights=conductor_target_weights,
                conductor_only=conductor_only,
            )
            for key in totals:
                totals[key] += metrics[key]
            batch_count += 1
    if batch_count == 0:
        raise ValueError("Validation dataloader produced no batches.")
    return {
        key: value / batch_count
        for key, value in totals.items()
    }


def run_conductor_training(
    config: dict[str, Any],
    *,
    config_path: str | Path = "configs/conductor.yaml",
    dry_run: bool = False,
    max_steps_override: int | None = None,
    limit_pieces: int | None = None,
    conductor_only: bool = False,
) -> dict[str, Any]:
    """Run the conductor integration stage."""
    set_seed(config["seed"])
    training_config = config["training"]
    losses_config = config.get("losses", {})
    reconstruction_weight = losses_config.get("reconstruction_weight", 1.0)
    conductor_weight = losses_config.get("conductor_weight", 0.2)
    motif_weight = losses_config.get("motif_weight", 0.0)
    phrase_boundary_weight = losses_config.get("phrase_boundary_weight", 0.0)
    phrase_boundary_class_weights = losses_config.get("phrase_boundary_class_weights")
    conductor_target_weights = losses_config.get("conductor_target_weights")
    device = resolve_device(training_config.get("device", "auto"))
    vocab_sizes = build_feature_vocab_sizes(config["tokenization"])
    train_dataset, val_dataset = build_datasets(config, limit_pieces=limit_pieces)
    if len(train_dataset) == 0:
        raise ValueError("Training dataset produced no windows.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset produced no windows.")

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

    model = build_conductor_model(config, vocab_sizes=vocab_sizes).to(device)
    initialization = maybe_initialize_from_checkpoint(
        model,
        training_config.get("init_checkpoint"),
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 0.0),
    )

    if dry_run:
        train_batch = move_batch_to_device(next(iter(train_loader)), device)
        val_batch = move_batch_to_device(next(iter(val_loader)), device)
        with torch.no_grad():
            train_output = model(
                train_batch.inputs,
                train_batch.attention_mask,
                phrase_ids=train_batch.phrase_ids,
                phrase_mask=train_batch.phrase_mask,
            )
            val_output = model(
                val_batch.inputs,
                val_batch.attention_mask,
                phrase_ids=val_batch.phrase_ids,
                phrase_mask=val_batch.phrase_mask,
            )
            _, train_metrics = compute_total_loss(
                train_output,
                train_batch,
                reconstruction_weight=reconstruction_weight,
                conductor_weight=conductor_weight,
                motif_weight=motif_weight,
                phrase_boundary_weight=phrase_boundary_weight,
                phrase_boundary_class_weights=phrase_boundary_class_weights,
                conductor_target_weights=conductor_target_weights,
                conductor_only=conductor_only,
            )
            _, val_metrics = compute_total_loss(
                val_output,
                val_batch,
                reconstruction_weight=reconstruction_weight,
                conductor_weight=conductor_weight,
                motif_weight=motif_weight,
                phrase_boundary_weight=phrase_boundary_weight,
                phrase_boundary_class_weights=phrase_boundary_class_weights,
                conductor_target_weights=conductor_target_weights,
                conductor_only=conductor_only,
            )
        return {
            "mode": "dry_run",
            "objective": "conductor_only" if conductor_only else "joint",
            "device": str(device),
            "train_windows": len(train_dataset),
            "val_windows": len(val_dataset),
            "vocab_sizes": vocab_sizes,
            "initialization": initialization,
            "train_batch": summarize_batch(train_batch),
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
        "conductor_loss": 0.0,
        "motif_loss": 0.0,
        "boundary_loss": 0.0,
        "reconstruction_loss": 0.0,
    }
    for step in range(1, max_steps + 1):
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch.inputs,
            batch.attention_mask,
            phrase_ids=batch.phrase_ids,
            phrase_mask=batch.phrase_mask,
        )
        total_loss, latest_metrics = compute_total_loss(
            output,
            batch,
            reconstruction_weight=reconstruction_weight,
            conductor_weight=conductor_weight,
            motif_weight=motif_weight,
            phrase_boundary_weight=phrase_boundary_weight,
            phrase_boundary_class_weights=phrase_boundary_class_weights,
            conductor_target_weights=conductor_target_weights,
            conductor_only=conductor_only,
        )
        total_loss.backward()
        clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        if step == 1 or step % log_every == 0:
            payload = {
                "step": step,
                "phase": "train",
                "loss": latest_metrics["loss"],
                "conductor_loss": latest_metrics["conductor_loss"],
                "motif_loss": latest_metrics["motif_loss"],
                "boundary_loss": latest_metrics["boundary_loss"],
                "reconstruction_loss": latest_metrics["reconstruction_loss"],
                "batch_shape": tuple(batch.inputs["pitch"].shape),
                "mean_length": float(batch.lengths.float().mean().item()),
                "mean_phrases": float(batch.phrase_mask.sum(dim=1).float().mean().item()),
            }
            append_metrics(metrics_path, payload)
            print(
                f"[train] step={step} loss={payload['loss']:.4f} "
                f"cond={payload['conductor_loss']:.4f} "
                f"motif={payload['motif_loss']:.4f} "
                f"boundary={payload['boundary_loss']:.4f} "
                f"recon={payload['reconstruction_loss']:.4f} "
                f"batch_shape={payload['batch_shape']} mean_phrases={payload['mean_phrases']:.1f}"
            )

        should_validate = step == 1 or step % validate_every == 0 or step == max_steps
        if should_validate:
            val_metrics = evaluate_model(
                model,
                val_loader,
                device=device,
                reconstruction_weight=reconstruction_weight,
                conductor_weight=conductor_weight,
                motif_weight=motif_weight,
                phrase_boundary_weight=phrase_boundary_weight,
                phrase_boundary_class_weights=phrase_boundary_class_weights,
                conductor_target_weights=conductor_target_weights,
                conductor_only=conductor_only,
                max_batches=eval_batches,
            )
            append_metrics(metrics_path, {"step": step, "phase": "val", **val_metrics})
            print(
                f"[val] step={step} loss={val_metrics['loss']:.4f} "
                f"cond={val_metrics['conductor_loss']:.4f} "
                f"motif={val_metrics['motif_loss']:.4f} "
                f"boundary={val_metrics['boundary_loss']:.4f} "
                f"recon={val_metrics['reconstruction_loss']:.4f}"
            )
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
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
        "objective": "conductor_only" if conductor_only else "joint",
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
    result = run_conductor_training(
        config,
        config_path=config_path,
        dry_run=args.dry_run,
        max_steps_override=args.max_steps,
        limit_pieces=args.limit_pieces,
        conductor_only=args.conductor_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
