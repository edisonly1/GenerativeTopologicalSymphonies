"""Losses for the denoising refiner stage."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from training.data import FEATURE_NAMES


@dataclass(slots=True)
class RefinerLossOutput:
    """Structured reconstruction loss over corrupted positions."""

    total_loss: Tensor
    per_feature_losses: dict[str, float]
    corrupted_token_count: int


def masked_grouped_reconstruction_loss(
    logits: dict[str, Tensor],
    targets: dict[str, Tensor],
    corruption_masks: dict[str, Tensor],
    *,
    ignore_index: int = -100,
) -> RefinerLossOutput:
    """Compute reconstruction loss only on corrupted positions."""
    losses: list[Tensor] = []
    per_feature_losses: dict[str, float] = {}
    corrupted_token_count = 0
    reference_logits = next(iter(logits.values()))

    for feature in FEATURE_NAMES:
        feature_targets = targets[feature].clone()
        feature_targets[~corruption_masks[feature]] = ignore_index
        valid_mask = feature_targets != ignore_index
        if not valid_mask.any():
            per_feature_losses[feature] = 0.0
            continue
        feature_loss = F.cross_entropy(
            logits[feature].transpose(1, 2),
            feature_targets,
            ignore_index=ignore_index,
        )
        losses.append(feature_loss)
        per_feature_losses[feature] = float(feature_loss.detach().cpu())
        corrupted_token_count += int(valid_mask.sum().item())

    if losses:
        total_loss = torch.stack(losses).mean()
    else:
        total_loss = reference_logits.sum() * 0.0
    return RefinerLossOutput(
        total_loss=total_loss,
        per_feature_losses=per_feature_losses,
        corrupted_token_count=corrupted_token_count,
    )
