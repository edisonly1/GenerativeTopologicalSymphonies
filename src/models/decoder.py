"""Baseline grouped-token decoder model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from training.data import FEATURE_NAMES


@dataclass(slots=True)
class BaselineDecoderConfig:
    """Configuration for the baseline grouped-token decoder."""

    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    dim_feedforward: int = 1024


class SinusoidalPositionEncoding(nn.Module):
    """Sinusoidal positional encoding without a fixed maximum sequence length."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, sequence_length: int, device: torch.device) -> Tensor:
        """Return positional encodings for the requested sequence length."""
        position = torch.arange(sequence_length, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10_000.0) / self.d_model)
        )
        encoding = torch.zeros((sequence_length, self.d_model), device=device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding.unsqueeze(0)


class BaselineGroupedDecoder(nn.Module):
    """Decoder-only Transformer over grouped symbolic event factors."""

    def __init__(
        self,
        *,
        vocab_sizes: dict[str, int],
        config: BaselineDecoderConfig,
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
        self.output_heads = nn.ModuleDict(
            {
                feature: nn.Linear(config.d_model, vocab_size)
                for feature, vocab_size in vocab_sizes.items()
            }
        )

    def _causal_mask(self, sequence_length: int, device: torch.device) -> Tensor:
        """Build a boolean causal attention mask."""
        return torch.triu(
            torch.ones((sequence_length, sequence_length), device=device, dtype=torch.bool),
            diagonal=1,
        )

    def embed_inputs(self, inputs: dict[str, Tensor]) -> Tensor:
        """Embed grouped-token inputs before sequence modeling."""
        if not inputs:
            raise ValueError("Decoder inputs may not be empty.")

        batch_size, sequence_length = next(iter(inputs.values())).shape
        device = next(self.parameters()).device
        if sequence_length == 0:
            raise ValueError("Decoder received an empty sequence.")

        hidden = torch.zeros((batch_size, sequence_length, self.config.d_model), device=device)
        for feature in FEATURE_NAMES:
            hidden = hidden + self.feature_embeddings[feature](inputs[feature].to(device))
        return self.dropout(hidden + self.position_encoding(sequence_length, device))

    def decode_context(self, hidden: Tensor, attention_mask: Tensor) -> Tensor:
        """Apply the causal decoder stack to embedded inputs."""
        sequence_length = hidden.shape[1]
        device = hidden.device
        hidden = self.transformer(
            hidden,
            mask=self._causal_mask(sequence_length, device),
            src_key_padding_mask=~attention_mask.to(device),
        )
        return self.final_norm(hidden)

    def encode_sequence(self, inputs: dict[str, Tensor], attention_mask: Tensor) -> Tensor:
        """Encode grouped-token inputs into contextual hidden states."""
        hidden = self.embed_inputs(inputs)
        return self.decode_context(hidden, attention_mask)

    def decode_hidden(self, hidden: Tensor) -> dict[str, Tensor]:
        """Project hidden states into per-feature logits."""
        return {
            feature: head(hidden)
            for feature, head in self.output_heads.items()
        }

    def forward(self, inputs: dict[str, Tensor], attention_mask: Tensor) -> dict[str, Tensor]:
        """Run the grouped decoder and return per-feature logits."""
        hidden = self.encode_sequence(inputs, attention_mask)
        return self.decode_hidden(hidden)
