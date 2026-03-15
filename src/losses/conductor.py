"""Phrase-level supervision losses for the conductor stage."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from training.conductor_targets import CONDUCTOR_TARGET_NAMES


@dataclass(slots=True)
class ConductorLossOutput:
    """Structured conductor-loss output."""

    total_loss: Tensor
    per_target_losses: dict[str, float]
    phrase_count: int


def conductor_supervision_loss(
    logits: dict[str, Tensor],
    targets: dict[str, Tensor],
    *,
    target_weights: dict[str, float] | None = None,
    ignore_index: int = -100,
) -> ConductorLossOutput:
    """Compute phrase-level classification loss across conductor targets."""
    losses: list[Tensor] = []
    per_target_losses: dict[str, float] = {}
    phrase_count = 0
    total_weight = 0.0
    reference_logits = next(iter(logits.values()))

    for target_name in CONDUCTOR_TARGET_NAMES:
        target_logits = logits[target_name].transpose(1, 2)
        target_values = targets[target_name]
        valid_mask = target_values != ignore_index
        if not valid_mask.any():
            per_target_losses[target_name] = 0.0
            continue
        target_loss = F.cross_entropy(
            target_logits,
            target_values,
            ignore_index=ignore_index,
        )
        target_weight = 1.0 if target_weights is None else target_weights.get(target_name, 1.0)
        losses.append(target_loss * target_weight)
        per_target_losses[target_name] = float(target_loss.detach().cpu())
        phrase_count += int(valid_mask.sum().item())
        total_weight += target_weight

    if losses:
        total_loss = torch.stack(losses).sum() / max(total_weight, 1e-8)
    else:
        total_loss = reference_logits.sum() * 0.0
    return ConductorLossOutput(
        total_loss=total_loss,
        per_target_losses=per_target_losses,
        phrase_count=phrase_count,
    )
