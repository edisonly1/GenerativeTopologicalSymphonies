"""1D U-Net denoiser baseline for symbolic grouped-token sequences."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from training.data import FEATURE_NAMES


@dataclass(slots=True)
class DiffusionUNetConfig:
    """Configuration for the symbolic U-Net denoiser baseline."""

    d_model: int = 256
    base_channels: int = 256
    dropout: float = 0.1


class ResidualConvBlock(nn.Module):
    """Small residual 1D convolution block."""

    def __init__(self, in_channels: int, out_channels: int, *, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(1, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.conv1(self.activation(self.norm1(inputs)))
        hidden = self.dropout(hidden)
        hidden = self.conv2(self.activation(self.norm2(hidden)))
        return hidden + self.skip(inputs)


class DiffusionUNetDenoiser(nn.Module):
    """Grouped-token denoiser with a lightweight 1D U-Net backbone."""

    def __init__(
        self,
        *,
        vocab_sizes: dict[str, int],
        config: DiffusionUNetConfig,
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
        self.input_projection = nn.Linear(config.d_model, config.base_channels)
        self.condition_projection = nn.Linear(config.d_model, config.base_channels)
        self.down_block = ResidualConvBlock(
            config.base_channels,
            config.base_channels,
            dropout=config.dropout,
        )
        self.downsample = nn.Conv1d(
            config.base_channels,
            config.base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.mid_block = ResidualConvBlock(
            config.base_channels * 2,
            config.base_channels * 2,
            dropout=config.dropout,
        )
        self.upsample = nn.ConvTranspose1d(
            config.base_channels * 2,
            config.base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.up_block = ResidualConvBlock(
            config.base_channels * 2,
            config.base_channels,
            dropout=config.dropout,
        )
        self.output_projection = nn.Linear(config.base_channels, config.d_model)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.output_heads = nn.ModuleDict(
            {
                feature: nn.Linear(config.d_model, vocab_size)
                for feature, vocab_size in vocab_sizes.items()
            }
        )

    def embed_inputs(self, inputs: dict[str, Tensor]) -> Tensor:
        """Embed grouped-token inputs into a dense sequence tensor."""
        if not inputs:
            raise ValueError("Denoiser inputs may not be empty.")
        batch_size, sequence_length = next(iter(inputs.values())).shape
        if sequence_length == 0:
            raise ValueError("Denoiser received an empty sequence.")
        device = next(self.parameters()).device
        hidden = torch.zeros((batch_size, sequence_length, self.config.d_model), device=device)
        for feature in FEATURE_NAMES:
            hidden = hidden + self.feature_embeddings[feature](inputs[feature].to(device))
        return self.dropout(hidden)

    def _match_length(self, tensor: Tensor, target_length: int) -> Tensor:
        if tensor.shape[-1] == target_length:
            return tensor
        if tensor.shape[-1] > target_length:
            return tensor[..., :target_length]
        pad = target_length - tensor.shape[-1]
        return torch.nn.functional.pad(tensor, (0, pad))

    def forward(
        self,
        inputs: dict[str, Tensor],
        attention_mask: Tensor,
        *,
        condition_state: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Denoise a corrupted grouped-token sequence."""
        hidden = self.embed_inputs(inputs)
        hidden = self.input_projection(hidden)
        if condition_state is not None:
            hidden = hidden + self.condition_projection(condition_state.to(hidden.device))
        hidden = hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)
        hidden = hidden.transpose(1, 2)

        skip = self.down_block(hidden)
        downsampled = self.downsample(skip)
        bottleneck = self.mid_block(downsampled)
        upsampled = self.upsample(bottleneck)
        upsampled = self._match_length(upsampled, skip.shape[-1])
        merged = torch.cat([upsampled, skip], dim=1)
        hidden = self.up_block(merged).transpose(1, 2)
        hidden = self.final_norm(self.output_projection(hidden))
        hidden = hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)
        return {
            feature: head(hidden)
            for feature, head in self.output_heads.items()
        }
