"""Loss functions for training the GTS models."""

from .conductor import ConductorLossOutput, conductor_supervision_loss
from .motif import MotifLossOutput, motif_recurrence_loss
from .phrase_boundary import PhraseBoundaryLossOutput, phrase_boundary_loss
from .refiner import RefinerLossOutput, masked_grouped_reconstruction_loss
from .reconstruction import ReconstructionLossOutput, grouped_reconstruction_loss
from .tension import TensionLossOutput, tension_regularization_loss
from .torus import TorusLossOutput, torus_losses

__all__ = [
    "ConductorLossOutput",
    "MotifLossOutput",
    "PhraseBoundaryLossOutput",
    "RefinerLossOutput",
    "ReconstructionLossOutput",
    "TensionLossOutput",
    "TorusLossOutput",
    "conductor_supervision_loss",
    "masked_grouped_reconstruction_loss",
    "motif_recurrence_loss",
    "phrase_boundary_loss",
    "grouped_reconstruction_loss",
    "tension_regularization_loss",
    "torus_losses",
]
