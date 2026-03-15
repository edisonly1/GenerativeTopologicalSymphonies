"""Denoising refinement module."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.decoder import SinusoidalPositionEncoding
from training.data import FEATURE_NAMES


@dataclass(slots=True)
class RefinerConfig:
    """Configuration for the conditional denoising refiner."""

    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    dim_feedforward: int = 1024


class ConditionalDenoisingRefiner(nn.Module):
    """Bidirectional transformer refiner over corrupted grouped-token sequences."""

    def __init__(
        self,
        *,
        vocab_sizes: dict[str, int],
        config: RefinerConfig,
    ) -> None:
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.config = config
        self.feature_embeddings = nn.ModuleDict(
            {
                feature: nn.Embedding(vocab_size, config.d_model, padding_idx=0)
                for feature, vocab_size in vocab_sizes.items()
            }
        )
        self.position_encoding = SinusoidalPositionEncoding(config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )
        self.condition_projection = nn.Linear(config.d_model * 2, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_heads = nn.ModuleDict(
            {
                feature: nn.Linear(config.d_model, vocab_size)
                for feature, vocab_size in vocab_sizes.items()
            }
        )

    def embed_inputs(self, inputs: dict[str, Tensor]) -> Tensor:
        """Embed grouped-token inputs before denoising."""
        if not inputs:
            raise ValueError("Refiner inputs may not be empty.")
        batch_size, sequence_length = next(iter(inputs.values())).shape
        if sequence_length == 0:
            raise ValueError("Refiner received an empty sequence.")
        device = next(self.parameters()).device
        hidden = torch.zeros((batch_size, sequence_length, self.config.d_model), device=device)
        for feature in FEATURE_NAMES:
            hidden = hidden + self.feature_embeddings[feature](inputs[feature].to(device))
        return self.dropout(hidden + self.position_encoding(sequence_length, device))

    def forward(
        self,
        inputs: dict[str, Tensor],
        attention_mask: Tensor,
        *,
        condition_state: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Denoise corrupted inputs conditioned on frozen primary-model context."""
        hidden = self.embed_inputs(inputs)
        if condition_state is None:
            condition_state = torch.zeros_like(hidden)
        conditioned = self.condition_projection(torch.cat([hidden, condition_state.to(hidden.device)], dim=-1))
        conditioned = self.dropout(conditioned)
        hidden = self.transformer(
            conditioned,
            src_key_padding_mask=~attention_mask.to(conditioned.device),
        )
        hidden = self.final_norm(hidden)
        return {
            feature: head(hidden)
            for feature, head in self.output_heads.items()
        }
