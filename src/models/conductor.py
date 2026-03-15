"""Phrase-level conductor and conductor-conditioned decoder models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.decoder import BaselineDecoderConfig, BaselineGroupedDecoder, SinusoidalPositionEncoding
from training.conductor_targets import (
    CONDUCTOR_TARGET_NAMES,
    DEFAULT_CONDUCTOR_TARGET_VOCAB_SIZES,
)


@dataclass(slots=True)
class PhraseConductorConfig:
    """Configuration for the phrase-level conductor stack."""

    d_model: int = 256
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    dim_feedforward: int = 512


@dataclass(slots=True)
class ConductorDecoderOutput:
    """Structured output for the conductor-conditioned decoder."""

    token_logits: dict[str, Tensor]
    conductor_logits: dict[str, Tensor]
    phrase_hidden: Tensor
    control_state: Tensor


def mean_pool_phrase_states(
    token_states: Tensor,
    phrase_ids: Tensor,
    *,
    attention_mask: Tensor,
    phrase_mask: Tensor,
) -> Tensor:
    """Pool token states into phrase states by mean over phrase membership."""
    batch_size, _, hidden_size = token_states.shape
    phrase_count = phrase_mask.shape[1]
    pooled = token_states.new_zeros((batch_size, phrase_count, hidden_size))
    for batch_index in range(batch_size):
        valid_tokens = attention_mask[batch_index]
        for phrase_index in range(phrase_count):
            if not phrase_mask[batch_index, phrase_index]:
                continue
            member_mask = valid_tokens & (phrase_ids[batch_index] == phrase_index)
            if member_mask.any():
                pooled[batch_index, phrase_index] = token_states[batch_index, member_mask].mean(dim=0)
    return pooled


def broadcast_phrase_states(phrase_states: Tensor, phrase_ids: Tensor, *, attention_mask: Tensor) -> Tensor:
    """Broadcast phrase-level states back to token positions."""
    safe_phrase_ids = phrase_ids.clamp(min=0)
    expanded_phrase_ids = safe_phrase_ids.unsqueeze(-1).expand(-1, -1, phrase_states.shape[-1])
    token_states = torch.gather(phrase_states, dim=1, index=expanded_phrase_ids)
    return token_states * attention_mask.unsqueeze(-1).to(token_states.dtype)


class PhraseConductor(nn.Module):
    """Phrase summarizer plus phrase-level control heads."""

    def __init__(
        self,
        *,
        config: PhraseConductorConfig,
        target_vocab_sizes: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.target_vocab_sizes = target_vocab_sizes or DEFAULT_CONDUCTOR_TARGET_VOCAB_SIZES.copy()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.position_encoding = SinusoidalPositionEncoding(config.d_model)
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_heads = nn.ModuleDict(
            {
                target_name: nn.Linear(config.d_model, self.target_vocab_sizes[target_name])
                for target_name in CONDUCTOR_TARGET_NAMES
            }
        )
        self.target_embeddings = nn.ModuleDict(
            {
                target_name: nn.Embedding(self.target_vocab_sizes[target_name], config.d_model)
                for target_name in CONDUCTOR_TARGET_NAMES
            }
        )
        self.control_projection = nn.Linear(len(CONDUCTOR_TARGET_NAMES) * config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        token_states: Tensor,
        *,
        phrase_ids: Tensor,
        attention_mask: Tensor,
        phrase_mask: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        """Predict phrase-level controls and a conditioning state."""
        pooled = mean_pool_phrase_states(
            token_states,
            phrase_ids,
            attention_mask=attention_mask,
            phrase_mask=phrase_mask,
        )
        phrase_count = pooled.shape[1]
        phrase_positions = self.position_encoding(phrase_count, pooled.device)
        phrase_hidden = self.transformer(
            self.dropout(pooled + phrase_positions),
            src_key_padding_mask=~phrase_mask,
        )
        phrase_hidden = self.final_norm(phrase_hidden)
        conductor_logits = {
            target_name: head(phrase_hidden)
            for target_name, head in self.output_heads.items()
        }
        control_features = []
        for target_name in CONDUCTOR_TARGET_NAMES:
            probabilities = torch.softmax(conductor_logits[target_name], dim=-1)
            control_features.append(probabilities @ self.target_embeddings[target_name].weight)
        control_state = phrase_hidden + self.control_projection(torch.cat(control_features, dim=-1))
        control_state = control_state * phrase_mask.unsqueeze(-1).to(control_state.dtype)
        return phrase_hidden, conductor_logits, control_state


class ConductorConditionedGroupedDecoder(BaselineGroupedDecoder):
    """Grouped-token decoder conditioned on phrase-level conductor plans."""

    def __init__(
        self,
        *,
        vocab_sizes: dict[str, int],
        decoder_config: BaselineDecoderConfig,
        conductor_config: PhraseConductorConfig,
        target_vocab_sizes: dict[str, int] | None = None,
    ) -> None:
        super().__init__(vocab_sizes=vocab_sizes, config=decoder_config)
        self.conductor = PhraseConductor(
            config=conductor_config,
            target_vocab_sizes=target_vocab_sizes,
        )
        self.condition_projection = nn.Linear(decoder_config.d_model * 2, decoder_config.d_model)
        self.condition_dropout = nn.Dropout(decoder_config.dropout)

    def forward(
        self,
        inputs: dict[str, Tensor],
        attention_mask: Tensor,
        *,
        phrase_ids: Tensor,
        phrase_mask: Tensor,
    ) -> ConductorDecoderOutput:
        """Run the conductor stage and the note-level decoder."""
        token_embeddings = self.embed_inputs(inputs)
        phrase_hidden, conductor_logits, control_state = self.conductor(
            token_embeddings,
            phrase_ids=phrase_ids,
            attention_mask=attention_mask,
            phrase_mask=phrase_mask,
        )
        broadcast_state = broadcast_phrase_states(
            control_state,
            phrase_ids,
            attention_mask=attention_mask,
        )
        conditioned_inputs = self.condition_projection(torch.cat([token_embeddings, broadcast_state], dim=-1))
        conditioned_inputs = self.condition_dropout(conditioned_inputs)
        hidden = self.decode_context(conditioned_inputs, attention_mask)
        token_logits = self.decode_hidden(hidden)
        return ConductorDecoderOutput(
            token_logits=token_logits,
            conductor_logits=conductor_logits,
            phrase_hidden=phrase_hidden,
            control_state=control_state,
        )
