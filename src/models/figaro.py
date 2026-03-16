"""FIGARO-style control-conditioned grouped-token baseline."""

from __future__ import annotations

from dataclasses import dataclass

from models.conductor import (
    ConductorConditionedGroupedDecoder,
    ConductorDecoderOutput,
    PhraseConductorConfig,
)
from models.decoder import BaselineDecoderConfig


@dataclass(slots=True)
class FigaroStyleConfig:
    """Configuration label for the FIGARO-style baseline."""

    control_dropout: float = 0.1


class FigaroStyleGroupedDecoder(ConductorConditionedGroupedDecoder):
    """Control-conditioned autoregressive baseline inspired by FIGARO."""

    def __init__(
        self,
        *,
        vocab_sizes: dict[str, int],
        decoder_config: BaselineDecoderConfig,
        conductor_config: PhraseConductorConfig,
        target_vocab_sizes: dict[str, int] | None = None,
    ) -> None:
        super().__init__(
            vocab_sizes=vocab_sizes,
            decoder_config=decoder_config,
            conductor_config=conductor_config,
            target_vocab_sizes=target_vocab_sizes,
        )


__all__ = [
    "ConductorDecoderOutput",
    "FigaroStyleConfig",
    "FigaroStyleGroupedDecoder",
]
