"""Motif and recurrence consistency objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(slots=True)
class MotifLossOutput:
    """Structured output for motif/recurrence consistency loss."""

    total_loss: Tensor
    positive_count: int
    negative_count: int


def motif_recurrence_loss(
    phrase_hidden: Tensor,
    recurrence_targets: Tensor,
    *,
    ignore_index: int = -100,
    negative_margin: float = 0.4,
) -> MotifLossOutput:
    """Encourage recurrent phrases to resemble earlier phrases and novel ones not to."""
    normalized_hidden = F.normalize(phrase_hidden, dim=-1)
    losses: list[Tensor] = []
    positive_count = 0
    negative_count = 0

    batch_size, phrase_count, _ = normalized_hidden.shape
    for batch_index in range(batch_size):
        for phrase_index in range(phrase_count):
            target_value = int(recurrence_targets[batch_index, phrase_index].item())
            if target_value == ignore_index or phrase_index == 0:
                continue
            previous_hidden = normalized_hidden[batch_index, :phrase_index]
            if previous_hidden.numel() == 0:
                continue
            current_hidden = normalized_hidden[batch_index, phrase_index]
            similarities = previous_hidden @ current_hidden
            best_similarity = similarities.max()
            if target_value == 1:
                losses.append(1.0 - best_similarity)
                positive_count += 1
            else:
                losses.append(F.relu(best_similarity - negative_margin))
                negative_count += 1

    total_loss = torch.stack(losses).mean() if losses else phrase_hidden.sum() * 0.0
    return MotifLossOutput(
        total_loss=total_loss,
        positive_count=positive_count,
        negative_count=negative_count,
    )
