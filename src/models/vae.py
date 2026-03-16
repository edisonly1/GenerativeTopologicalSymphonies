"""Sequence VAE baseline over grouped symbolic event tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.decoder import BaselineDecoderConfig, BaselineGroupedDecoder
from models.encoder import EncoderConfig, GroupedSequenceEncoder


@dataclass(slots=True)
class SequenceVAEConfig:
    """Configuration for a grouped-token sequence VAE."""

    latent_dim: int = 128
    dropout: float = 0.1


@dataclass(slots=True)
class SequenceVAEOutput:
    """Structured output for the grouped-token sequence VAE."""

    token_logits: dict[str, Tensor]
    latent_mean: Tensor
    latent_logvar: Tensor
    latent_sample: Tensor


class GroupedSequenceVAE(nn.Module):
    """Bidirectional encoder plus causal decoder with sequence-level latent code."""

    def __init__(
        self,
        *,
        vocab_sizes: dict[str, int],
        encoder_config: EncoderConfig,
        decoder_config: BaselineDecoderConfig,
        vae_config: SequenceVAEConfig,
    ) -> None:
        super().__init__()
        self.encoder = GroupedSequenceEncoder(vocab_sizes=vocab_sizes, config=encoder_config)
        self.decoder = BaselineGroupedDecoder(vocab_sizes=vocab_sizes, config=decoder_config)
        self.vae_config = vae_config
        self.to_mean = nn.Linear(encoder_config.d_model, vae_config.latent_dim)
        self.to_logvar = nn.Linear(encoder_config.d_model, vae_config.latent_dim)
        self.latent_to_hidden = nn.Linear(vae_config.latent_dim, decoder_config.d_model)
        self.dropout = nn.Dropout(vae_config.dropout)

    def encode(self, inputs: dict[str, Tensor], attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Encode a sequence into mean and log-variance latent parameters."""
        encoded = self.encoder(inputs, attention_mask)
        mask = attention_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        mean = self.to_mean(pooled)
        logvar = self.to_logvar(pooled)
        return mean, logvar

    def reparameterize(self, mean: Tensor, logvar: Tensor) -> Tensor:
        """Sample a latent code with the reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            noise = torch.randn_like(std)
            return mean + noise * std
        return mean

    def decode(self, inputs: dict[str, Tensor], attention_mask: Tensor, latent_sample: Tensor) -> dict[str, Tensor]:
        """Decode grouped-token inputs conditioned on a sequence latent."""
        hidden = self.decoder.embed_inputs(inputs)
        latent_bias = self.latent_to_hidden(latent_sample).unsqueeze(1)
        hidden = self.dropout(hidden + latent_bias)
        hidden = self.decoder.decode_context(hidden, attention_mask)
        return self.decoder.decode_hidden(hidden)

    def forward(self, inputs: dict[str, Tensor], attention_mask: Tensor) -> SequenceVAEOutput:
        """Encode, sample, and decode a grouped-token sequence."""
        mean, logvar = self.encode(inputs, attention_mask)
        latent_sample = self.reparameterize(mean, logvar)
        token_logits = self.decode(inputs, attention_mask, latent_sample)
        return SequenceVAEOutput(
            token_logits=token_logits,
            latent_mean=mean,
            latent_logvar=logvar,
            latent_sample=latent_sample,
        )
