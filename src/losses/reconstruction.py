"""Primary symbolic reconstruction objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from training.data import FEATURE_NAMES


@dataclass(slots=True)
class ReconstructionLossOutput:
    """Structured reconstruction-loss output for grouped event factors."""

    total_loss: Tensor
    per_feature_losses: dict[str, float]
    token_count: int


def grouped_reconstruction_loss(
    logits: dict[str, Tensor],
    targets: dict[str, Tensor],
    *,
    ignore_index: int = -100,
) -> ReconstructionLossOutput:
    """Compute grouped next-step reconstruction loss across all event factors."""
    losses: list[Tensor] = []
    per_feature_losses: dict[str, float] = {}
    token_count = 0

    for feature in FEATURE_NAMES:
        feature_logits = logits[feature].transpose(1, 2)
        feature_targets = targets[feature]
        feature_loss = F.cross_entropy(
            feature_logits,
            feature_targets,
            ignore_index=ignore_index,
        )
        losses.append(feature_loss)
        per_feature_losses[feature] = float(feature_loss.detach().cpu())
        token_count += int((feature_targets != ignore_index).sum().item())

    total_loss = torch.stack(losses).mean()
    return ReconstructionLossOutput(
        total_loss=total_loss,
        per_feature_losses=per_feature_losses,
        token_count=token_count,
    )
