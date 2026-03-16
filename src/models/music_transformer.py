"""Relative-attention autoregressive decoder inspired by Music Transformer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from training.data import FEATURE_NAMES


@dataclass(slots=True)
class MusicTransformerConfig:
    """Configuration for a relative-attention grouped-token decoder."""

    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    dim_feedforward: int = 1024
    relative_attention_buckets: int = 32
    max_relative_distance: int = 128


class RelativePositionBias(nn.Module):
    """Learned relative attention bias over token distances."""

    def __init__(self, *, num_heads: int, num_buckets: int, max_distance: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bias = nn.Embedding(num_buckets, num_heads)

    def _bucket_distances(self, relative_positions: Tensor) -> Tensor:
        distances = relative_positions.clamp(min=0)
        if self.num_buckets <= 1:
            return torch.zeros_like(distances)
        max_exact = max(self.num_buckets // 2, 1)
        is_small = distances < max_exact
        safe_max_distance = max(self.max_distance, max_exact + 1)
        ratio = torch.log(
            distances.to(torch.float32).clamp(min=1) / float(max_exact)
        ) / math.log(float(safe_max_distance) / float(max_exact))
        large_bucket = max_exact + (ratio * (self.num_buckets - max_exact)).to(torch.long)
        large_bucket = large_bucket.clamp(max=self.num_buckets - 1)
        return torch.where(is_small, distances, large_bucket)

    def forward(self, sequence_length: int, device: torch.device) -> Tensor:
        positions = torch.arange(sequence_length, device=device)
        relative_positions = positions[:, None] - positions[None, :]
        bucket_ids = self._bucket_distances(relative_positions)
        bias = self.bias(bucket_ids)
        return bias.permute(2, 0, 1)


class MusicTransformerDecoderLayer(nn.Module):
    """Causal self-attention layer with learned relative-position bias."""

    def __init__(self, *, config: MusicTransformerConfig) -> None:
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.config = config
        self.head_dim = config.d_model // config.num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.relative_bias = RelativePositionBias(
            num_heads=config.num_heads,
            num_buckets=config.relative_attention_buckets,
            max_distance=config.max_relative_distance,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, config.d_model),
        )

    def _split_heads(self, tensor: Tensor) -> Tensor:
        batch_size, sequence_length, _ = tensor.shape
        return tensor.view(
            batch_size,
            sequence_length,
            self.config.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)

    def _merge_heads(self, tensor: Tensor) -> Tensor:
        batch_size, _, sequence_length, _ = tensor.shape
        return tensor.permute(0, 2, 1, 3).reshape(
            batch_size,
            sequence_length,
            self.config.d_model,
        )

    def forward(self, hidden: Tensor, attention_mask: Tensor) -> Tensor:
        sequence_length = hidden.shape[1]
        residual = hidden
        hidden = self.norm1(hidden)

        query = self._split_heads(self.q_proj(hidden))
        key = self._split_heads(self.k_proj(hidden))
        value = self._split_heads(self.v_proj(hidden))

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attention_scores = attention_scores + self.relative_bias(sequence_length, hidden.device).unsqueeze(0)

        causal_mask = torch.triu(
            torch.ones((sequence_length, sequence_length), device=hidden.device, dtype=torch.bool),
            diagonal=1,
        )
        attention_scores = attention_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        key_mask = ~attention_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)
        attention_scores = attention_scores.masked_fill(key_mask, float("-inf"))

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0, posinf=0.0, neginf=0.0)
        attention_weights = self.dropout(attention_weights)
        attended = torch.matmul(attention_weights, value)
        attended = self.out_proj(self._merge_heads(attended))
        attended = attended * attention_mask.unsqueeze(-1).to(attended.dtype)
        hidden = residual + self.dropout(attended)

        residual = hidden
        hidden = self.norm2(hidden)
        hidden = residual + self.dropout(self.feedforward(hidden))
        return hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)


class MusicTransformerGroupedDecoder(nn.Module):
    """Grouped-token decoder with relative self-attention."""

    def __init__(
        self,
        *,
        vocab_sizes: dict[str, int],
        config: MusicTransformerConfig,
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
        self.input_projection = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [MusicTransformerDecoderLayer(config=config) for _ in range(config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_heads = nn.ModuleDict(
            {
                feature: nn.Linear(config.d_model, vocab_size)
                for feature, vocab_size in vocab_sizes.items()
            }
        )

    def embed_inputs(self, inputs: dict[str, Tensor]) -> Tensor:
        """Embed grouped-token inputs before autoregressive decoding."""
        if not inputs:
            raise ValueError("Decoder inputs may not be empty.")
        batch_size, sequence_length = next(iter(inputs.values())).shape
        if sequence_length == 0:
            raise ValueError("Decoder received an empty sequence.")
        device = next(self.parameters()).device
        hidden = torch.zeros((batch_size, sequence_length, self.config.d_model), device=device)
        for feature in FEATURE_NAMES:
            hidden = hidden + self.feature_embeddings[feature](inputs[feature].to(device))
        return self.dropout(self.input_projection(hidden))

    def decode_hidden(self, hidden: Tensor) -> dict[str, Tensor]:
        """Project hidden states into grouped-token logits."""
        return {
            feature: head(hidden)
            for feature, head in self.output_heads.items()
        }

    def forward(self, inputs: dict[str, Tensor], attention_mask: Tensor) -> dict[str, Tensor]:
        """Run relative-attention decoding and return per-feature logits."""
        hidden = self.embed_inputs(inputs)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)
        hidden = self.final_norm(hidden)
        hidden = hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)
        return self.decode_hidden(hidden)
