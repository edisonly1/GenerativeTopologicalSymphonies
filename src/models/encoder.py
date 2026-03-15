"""Bidirectional grouped-token encoder definitions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.decoder import SinusoidalPositionEncoding
from training.data import FEATURE_NAMES


@dataclass(slots=True)
class EncoderConfig:
    """Configuration for the bidirectional grouped-token encoder."""

    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    dim_feedforward: int = 1024


class GroupedSequenceEncoder(nn.Module):
    """Bidirectional transformer encoder over grouped symbolic features."""

    def __init__(
        self,
        *,
        vocab_sizes: dict[str, int],
        config: EncoderConfig,
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
        self.dropout = nn.Dropout(config.dropout)
        self.final_norm = nn.LayerNorm(config.d_model)

    def embed_inputs(self, inputs: dict[str, Tensor]) -> Tensor:
        """Embed grouped-token inputs before bidirectional encoding."""
        if not inputs:
            raise ValueError("Encoder inputs may not be empty.")
        batch_size, sequence_length = next(iter(inputs.values())).shape
        if sequence_length == 0:
            raise ValueError("Encoder received an empty sequence.")
        device = next(self.parameters()).device
        hidden = torch.zeros((batch_size, sequence_length, self.config.d_model), device=device)
        for feature in FEATURE_NAMES:
            hidden = hidden + self.feature_embeddings[feature](inputs[feature].to(device))
        return self.dropout(hidden + self.position_encoding(sequence_length, device))

    def forward(self, inputs: dict[str, Tensor], attention_mask: Tensor) -> Tensor:
        """Encode grouped-token inputs into contextual hidden states."""
        hidden = self.embed_inputs(inputs)
        hidden = self.transformer(
            hidden,
            src_key_padding_mask=~attention_mask.to(hidden.device),
        )
        return self.final_norm(hidden)
