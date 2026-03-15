"""Torus-conditioned phrase-planning decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.conductor import (
    ConductorConditionedGroupedDecoder,
    PhraseConductorConfig,
    broadcast_phrase_states,
)
from models.decoder import BaselineDecoderConfig
from models.torus_latent import TorusLatentBottleneck, TorusLatentConfig


@dataclass(slots=True)
class TorusDecoderOutput:
    """Structured output for the torus-conditioned decoder."""

    token_logits: dict[str, Tensor]
    conductor_logits: dict[str, Tensor]
    phrase_hidden: Tensor
    control_state: Tensor
    torus_embedding: Tensor
    torus_pairs: Tensor
    torus_angles: Tensor
    torus_radii: Tensor
    global_style: Tensor
    latent_coordinates: Tensor
    latent_state: Tensor
    latent_geometry: str
    axis_labels: tuple[str, ...]


class TorusConditionedGroupedDecoder(ConductorConditionedGroupedDecoder):
    """Decoder conditioned on conductor plans plus torus latent structure."""

    def __init__(
        self,
        *,
        vocab_sizes: dict[str, int],
        decoder_config: BaselineDecoderConfig,
        conductor_config: PhraseConductorConfig,
        torus_config: TorusLatentConfig,
        target_vocab_sizes: dict[str, int] | None = None,
    ) -> None:
        super().__init__(
            vocab_sizes=vocab_sizes,
            decoder_config=decoder_config,
            conductor_config=conductor_config,
            target_vocab_sizes=target_vocab_sizes,
        )
        self.torus = TorusLatentBottleneck(config=torus_config)
        self.torus_condition_projection = nn.Linear(decoder_config.d_model * 2, decoder_config.d_model)
        self.torus_condition_dropout = nn.Dropout(decoder_config.dropout)

    def forward(
        self,
        inputs: dict[str, Tensor],
        attention_mask: Tensor,
        *,
        phrase_ids: Tensor,
        phrase_mask: Tensor,
    ) -> TorusDecoderOutput:
        """Run conductor planning, torus bottlenecking, and note-level decoding."""
        token_embeddings = self.embed_inputs(inputs)
        phrase_hidden, conductor_logits, control_state = self.conductor(
            token_embeddings,
            phrase_ids=phrase_ids,
            attention_mask=attention_mask,
            phrase_mask=phrase_mask,
        )
        torus_output = self.torus(control_state, phrase_mask=phrase_mask)
        torus_condition = self.torus_condition_projection(
            torch.cat([control_state, torus_output.torus_embedding], dim=-1)
        )
        torus_condition = self.torus_condition_dropout(torus_condition)
        torus_condition = torus_condition * phrase_mask.unsqueeze(-1).to(torus_condition.dtype)
        broadcast_state = broadcast_phrase_states(
            torus_condition,
            phrase_ids,
            attention_mask=attention_mask,
        )
        conditioned_inputs = self.condition_projection(torch.cat([token_embeddings, broadcast_state], dim=-1))
        conditioned_inputs = self.condition_dropout(conditioned_inputs)
        hidden = self.decode_context(conditioned_inputs, attention_mask)
        token_logits = self.decode_hidden(hidden)
        return TorusDecoderOutput(
            token_logits=token_logits,
            conductor_logits=conductor_logits,
            phrase_hidden=phrase_hidden,
            control_state=control_state,
            torus_embedding=torus_output.torus_embedding,
            torus_pairs=torus_output.torus_pairs,
            torus_angles=torus_output.torus_angles,
            torus_radii=torus_output.torus_radii,
            global_style=torus_output.global_style,
            latent_coordinates=torus_output.latent_coordinates,
            latent_state=torus_output.latent_state,
            latent_geometry=torus_output.latent_geometry,
            axis_labels=torus_output.axis_labels,
        )
