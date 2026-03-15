"""Auxiliary loss for preserving phrase-boundary tokens during decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


SHIFTED_PHRASE_BOUNDARY_TOKEN_IDS = (2, 3, 4)


@dataclass(slots=True)
class PhraseBoundaryLossOutput:
    """Structured output for the phrase-boundary auxiliary objective."""

    total_loss: Tensor
    valid_token_count: int
    boundary_token_count: int


def phrase_boundary_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    class_weights: Sequence[float] | Tensor | None = None,
    ignore_index: int = -100,
) -> PhraseBoundaryLossOutput:
    """Upweight phrase-start and phrase-end token prediction errors."""
    if class_weights is None:
        weight_tensor = None
    else:
        weight_tensor = torch.as_tensor(
            class_weights,
            dtype=logits.dtype,
            device=logits.device,
        )
        if weight_tensor.numel() != logits.shape[-1]:
            raise ValueError(
                "phrase boundary class_weights must match phrase_flag vocabulary size."
            )

    loss = F.cross_entropy(
        logits.transpose(1, 2),
        targets,
        weight=weight_tensor,
        ignore_index=ignore_index,
    )
    valid_mask = targets != ignore_index
    boundary_mask = torch.zeros_like(valid_mask)
    for token_id in SHIFTED_PHRASE_BOUNDARY_TOKEN_IDS:
        boundary_mask |= targets == token_id
    return PhraseBoundaryLossOutput(
        total_loss=loss,
        valid_token_count=int(valid_mask.sum().item()),
        boundary_token_count=int((boundary_mask & valid_mask).sum().item()),
    )
