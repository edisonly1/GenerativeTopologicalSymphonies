"""Train or dry-run the grouped-token VAE decoder baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from losses.reconstruction import grouped_reconstruction_loss
from models.benchmarks import get_benchmark_model_spec, validate_benchmark_model_config
from models.decoder import BaselineDecoderConfig
from models.encoder import EncoderConfig
from models.vae import GroupedSequenceVAE, SequenceVAEConfig
from training.data import build_weighted_window_sampler, create_autoregressive_dataloader
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


def parse_args() -> argparse.Namespace:
    spec = get_benchmark_model_spec("vae_decoder")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=spec.default_config)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit-pieces", type=int, default=None)
    return parser.parse_args()


def build_vae_model(config: dict[str, Any], *, vocab_sizes: dict[str, int]) -> GroupedSequenceVAE:
    """Instantiate the grouped-token sequence VAE baseline."""
    model_config = config["model"]
    encoder_config = EncoderConfig(
        d_model=model_config["d_model"],
        num_layers=model_config.get("encoder_layers", model_config["num_layers"]),
        num_heads=model_config["num_heads"],
        dropout=model_config["dropout"],
        dim_feedforward=model_config.get("dim_feedforward", model_config["d_model"] * 4),
    )
    decoder_config = BaselineDecoderConfig(
        d_model=model_config["d_model"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        dropout=model_config["dropout"],
        dim_feedforward=model_config.get("dim_feedforward", model_config["d_model"] * 4),
    )
    vae_config = SequenceVAEConfig(
        latent_dim=model_config.get("latent_dim", 128),
        dropout=model_config.get("vae_dropout", model_config["dropout"]),
    )
    return GroupedSequenceVAE(
        vocab_sizes=vocab_sizes,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        vae_config=vae_config,
    )


def kl_divergence(latent_mean: torch.Tensor, latent_logvar: torch.Tensor) -> torch.Tensor:
    """Compute mean KL divergence to the unit Gaussian prior."""
    return 0.5 * torch.mean(
        torch.sum(torch.exp(latent_logvar) + latent_mean.square() - 1.0 - latent_logvar, dim=-1)
    )


def compute_total_loss(
    model_output,
    batch,
    *,
    kl_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine grouped-token reconstruction loss with KL regularization."""
    reconstruction_loss = grouped_reconstruction_loss(model_output.token_logits, batch.targets)
    kl_loss = kl_divergence(model_output.latent_mean, model_output.latent_logvar)
    total_loss = reconstruction_loss.total_loss + kl_weight * kl_loss
    return total_loss, {
        "loss": float(total_loss.item()),
        "reconstruction_loss": float(reconstruction_loss.total_loss.item()),
        "kl_loss": float(kl_loss.item()),
    }


def evaluate_model(
    model: GroupedSequenceVAE,
    dataloader,
    *,
    device: torch.device,
    kl_weight: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run validation and return mean VAE metrics."""
    model.eval()
    totals = {"loss": 0.0, "reconstruction_loss": 0.0, "kl_loss": 0.0}
    batch_count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            output = model(batch.inputs, batch.attention_mask)
            _, metrics = compute_total_loss(output, batch, kl_weight=kl_weight)
            for key in totals:
                totals[key] += metrics[key]
            batch_count += 1
    if batch_count == 0:
        raise ValueError("Validation dataloader produced no batches.")
    return {key: value / batch_count for key, value in totals.items()}


def run_vae_training(
    config: dict[str, Any],
    *,
    config_path: str | Path = "configs/vae_decoder_asap_score.yaml",
    dry_run: bool = False,
    max_steps_override: int | None = None,
    limit_pieces: int | None = None,
) -> dict[str, Any]:
    """Run the grouped-token VAE baseline or a dry-run validation pass."""
    set_seed(config["seed"])
    training_config = config["training"]
    device = resolve_device(training_config.get("device", "auto"))
    kl_weight = config.get("losses", {}).get("kl_weight", 0.05)
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
    model = build_vae_model(config, vocab_sizes=vocab_sizes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 0.0),
    )
    max_steps = max_steps_override or training_config["max_steps"]
    eval_batches = training_config.get("eval_batches")

    if dry_run:
        batch = next(iter(train_loader))
        batch = move_batch_to_device(batch, device)
        output = model(batch.inputs, batch.attention_mask)
        train_loss, train_metrics = compute_total_loss(output, batch, kl_weight=kl_weight)
        val_metrics = evaluate_model(
            model,
            val_loader,
            device=device,
            kl_weight=kl_weight,
            max_batches=eval_batches,
        )
        return {
            "mode": "dry_run",
            "device": str(device),
            "train_windows": len(train_dataset),
            "val_windows": len(val_dataset),
            "train_loss": float(train_loss.item()),
            "train_metrics": train_metrics,
            "train_batch": summarize_batch(batch),
            "val_metrics": val_metrics,
        }

    output_dir, checkpoints_dir, metrics_path = prepare_run_directory(Path(config_path), config)
    step = 0
    best_val_loss = float("inf")
    latest_metrics: dict[str, float] = {}
    gradient_clip_norm = training_config.get("gradient_clip_norm", 1.0)
    log_every = training_config.get("log_every", 10)
    validate_every = training_config.get("validate_every", 100)
    checkpoint_every = training_config.get("checkpoint_every", validate_every)

    train_iterator = iter(train_loader)
    while step < max_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        batch = move_batch_to_device(batch, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.inputs, batch.attention_mask)
        total_loss, metrics = compute_total_loss(output, batch, kl_weight=kl_weight)
        total_loss.backward()
        clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        step += 1

        latest_metrics = metrics
        if step % log_every == 0 or step == 1 or step == max_steps:
            mean_length = float(batch.lengths.float().mean().item())
            print(
                "[train] "
                f"step={step} loss={metrics['loss']:.4f} "
                f"recon={metrics['reconstruction_loss']:.4f} "
                f"kl={metrics['kl_loss']:.4f} "
                f"batch_shape={tuple(batch.inputs['pitch'].shape)} mean_len={mean_length:.1f}"
            )
            append_metrics(metrics_path, {"split": "train", "step": step, **metrics})

        if step % validate_every == 0 or step == 1 or step == max_steps:
            val_metrics = evaluate_model(
                model,
                val_loader,
                device=device,
                kl_weight=kl_weight,
                max_batches=eval_batches,
            )
            print(f"[val] step={step} loss={val_metrics['loss']:.4f}")
            append_metrics(metrics_path, {"split": "val", "step": step, **val_metrics})
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
        "device": str(device),
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "max_steps": max_steps,
        "best_val_loss": best_val_loss,
        "latest_metrics": latest_metrics,
        "output_dir": str(output_dir),
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    validate_benchmark_model_config(config, expected_model="vae_decoder")
    result = run_vae_training(
        config,
        config_path=config_path,
        dry_run=args.dry_run,
        max_steps_override=args.max_steps,
        limit_pieces=args.limit_pieces,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
