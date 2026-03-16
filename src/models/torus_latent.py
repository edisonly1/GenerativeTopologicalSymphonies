"""Torus-constrained latent utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


LATENT_GEOMETRY_LABELS = ("pitch_cycle", "rhythm_cycle", "harmony_cycle")
TORUS_GEOMETRIES = frozenset({"legacy_torus", "torus_t3"})
EUCLIDEAN_GEOMETRIES = frozenset({"euclidean_r3", "plane_r2", "hypercube_r3"})
SPHERE_GEOMETRIES = frozenset({"sphere_s2"})


@dataclass(slots=True)
class TorusLatentConfig:
    """Configuration for the torus latent bottleneck."""

    d_model: int = 256
    latent_geometry: Literal[
        "legacy_torus",
        "torus_t3",
        "euclidean_r3",
        "plane_r2",
        "sphere_s2",
        "hypercube_r3",
    ] = "legacy_torus"
    latent_style_dim: int = 64
    torus_axis_count: int = 3
    euclidean_dim: int = 3
    plane_dim: int = 2
    sphere_dim: int = 3
    hypercube_dim: int = 3
    dropout: float = 0.1
    eps: float = 1e-6


@dataclass(slots=True)
class TorusLatentOutput:
    """Structured output of the torus bottleneck."""

    torus_embedding: Tensor
    torus_pairs: Tensor
    torus_angles: Tensor
    torus_radii: Tensor
    global_style: Tensor
    latent_coordinates: Tensor
    latent_state: Tensor
    latent_geometry: str
    axis_labels: tuple[str, ...]


class TorusLatentBottleneck(nn.Module):
    """Project phrase states onto a product of circles and a global style code."""

    def __init__(self, *, config: TorusLatentConfig) -> None:
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.d_model)
        self.to_coordinates: nn.Linear | None = None
        if config.latent_geometry == "legacy_torus":
            if config.latent_style_dim <= 0 or config.latent_style_dim % 2 != 0:
                raise ValueError("latent_style_dim must be a positive even integer.")
            self.circle_count = config.latent_style_dim // 2
            self.axis_labels = tuple(f"cycle_{index}" for index in range(self.circle_count))
            self.to_torus_pairs = nn.Linear(config.d_model, config.latent_style_dim)
            local_input_dim = config.latent_style_dim
            global_input_dim = config.latent_style_dim
            self.axis_projections = None
        elif config.latent_geometry == "torus_t3":
            if config.torus_axis_count != 3:
                raise ValueError("torus_t3 expects torus_axis_count = 3.")
            self.circle_count = config.torus_axis_count
            self.axis_labels = LATENT_GEOMETRY_LABELS[: self.circle_count]
            self.axis_projections = nn.ModuleList(
                [nn.Linear(config.d_model, 2) for _ in range(self.circle_count)]
            )
            self.to_torus_pairs = None
            local_input_dim = self.circle_count * 2
            global_input_dim = self.circle_count * 2
        elif config.latent_geometry == "euclidean_r3":
            if config.euclidean_dim != 3:
                raise ValueError("euclidean_r3 expects euclidean_dim = 3.")
            self.circle_count = config.euclidean_dim
            self.axis_labels = LATENT_GEOMETRY_LABELS[: self.circle_count]
            self.to_torus_pairs = None
            self.axis_projections = None
            self.to_coordinates = nn.Linear(config.d_model, config.euclidean_dim)
            local_input_dim = config.euclidean_dim
            global_input_dim = config.euclidean_dim
        elif config.latent_geometry == "plane_r2":
            if config.plane_dim != 2:
                raise ValueError("plane_r2 expects plane_dim = 2.")
            self.circle_count = config.plane_dim
            self.axis_labels = LATENT_GEOMETRY_LABELS[: self.circle_count]
            self.to_torus_pairs = None
            self.axis_projections = None
            self.to_coordinates = nn.Linear(config.d_model, config.plane_dim)
            local_input_dim = config.plane_dim
            global_input_dim = config.plane_dim
        elif config.latent_geometry == "sphere_s2":
            if config.sphere_dim != 3:
                raise ValueError("sphere_s2 expects sphere_dim = 3.")
            self.circle_count = config.sphere_dim
            self.axis_labels = LATENT_GEOMETRY_LABELS[: self.circle_count]
            self.to_torus_pairs = None
            self.axis_projections = None
            self.to_coordinates = nn.Linear(config.d_model, config.sphere_dim)
            local_input_dim = config.sphere_dim
            global_input_dim = config.sphere_dim
        elif config.latent_geometry == "hypercube_r3":
            if config.hypercube_dim != 3:
                raise ValueError("hypercube_r3 expects hypercube_dim = 3.")
            self.circle_count = config.hypercube_dim
            self.axis_labels = LATENT_GEOMETRY_LABELS[: self.circle_count]
            self.to_torus_pairs = None
            self.axis_projections = None
            self.to_coordinates = nn.Linear(config.d_model, config.hypercube_dim)
            local_input_dim = config.hypercube_dim
            global_input_dim = config.hypercube_dim
        else:
            raise ValueError(f"Unsupported latent geometry: {config.latent_geometry}")
        self.local_projection = nn.Linear(local_input_dim, config.d_model)
        self.global_projection = nn.Linear(global_input_dim, config.d_model)
        self.output_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def _masked_style_mean(self, latent_state: Tensor, phrase_mask: Tensor) -> Tensor:
        """Pool per-phrase torus states into a global style summary."""
        mask = phrase_mask.unsqueeze(-1).to(latent_state.dtype)
        summed = (latent_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def _legacy_torus_forward(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Project phrase states into the legacy many-circle torus."""
        raw_pairs = self.to_torus_pairs(hidden).view(
            hidden.shape[0],
            hidden.shape[1],
            self.circle_count,
            2,
        )
        torus_radii = torch.linalg.vector_norm(raw_pairs, dim=-1)
        safe_radii = torus_radii.unsqueeze(-1).clamp(min=self.config.eps)
        torus_pairs = raw_pairs / safe_radii
        torus_angles = torch.atan2(torus_pairs[..., 1], torus_pairs[..., 0])
        latent_state = torus_pairs.reshape(hidden.shape[0], hidden.shape[1], -1)
        return latent_state, torus_pairs, torus_angles, torus_radii

    def _t3_torus_forward(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Project phrase states into an explicit T^3 latent."""
        raw_pairs = torch.stack([projection(hidden) for projection in self.axis_projections], dim=2)
        torus_radii = torch.linalg.vector_norm(raw_pairs, dim=-1)
        safe_radii = torus_radii.unsqueeze(-1).clamp(min=self.config.eps)
        torus_pairs = raw_pairs / safe_radii
        torus_angles = torch.atan2(torus_pairs[..., 1], torus_pairs[..., 0])
        latent_state = torus_pairs.reshape(hidden.shape[0], hidden.shape[1], -1)
        return latent_state, torus_pairs, torus_angles, torus_radii

    def _euclidean_forward(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Project phrase states into a matched Euclidean R^3 latent."""
        coordinates = self.to_coordinates(hidden)
        torus_pairs = coordinates.unsqueeze(-1)
        torus_angles = coordinates
        torus_radii = torch.linalg.vector_norm(coordinates, dim=-1, keepdim=True).expand_as(coordinates)
        return coordinates, torus_pairs, torus_angles, torus_radii

    def _plane_forward(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Project phrase states into a learned 2D plane."""
        coordinates = self.to_coordinates(hidden)
        torus_pairs = coordinates.unsqueeze(-1)
        torus_angles = coordinates
        torus_radii = torch.linalg.vector_norm(coordinates, dim=-1, keepdim=True).expand_as(coordinates)
        return coordinates, torus_pairs, torus_angles, torus_radii

    def _sphere_forward(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Project phrase states onto a unit sphere."""
        raw_coordinates = self.to_coordinates(hidden)
        torus_radii = torch.linalg.vector_norm(raw_coordinates, dim=-1, keepdim=True).clamp(min=self.config.eps)
        coordinates = raw_coordinates / torus_radii
        torus_pairs = coordinates.unsqueeze(-1)
        torus_angles = coordinates
        expanded_radii = torus_radii.expand_as(coordinates)
        return coordinates, torus_pairs, torus_angles, expanded_radii

    def _hypercube_forward(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Project phrase states into a bounded 3D hypercube."""
        coordinates = torch.tanh(self.to_coordinates(hidden))
        torus_pairs = coordinates.unsqueeze(-1)
        torus_angles = coordinates
        torus_radii = torch.linalg.vector_norm(coordinates, dim=-1, keepdim=True).expand_as(coordinates)
        return coordinates, torus_pairs, torus_angles, torus_radii

    def forward(
        self,
        phrase_states: Tensor,
        *,
        phrase_mask: Tensor,
    ) -> TorusLatentOutput:
        """Project phrase states into torus coordinates and a style embedding."""
        hidden = self.input_norm(phrase_states)
        if self.config.latent_geometry == "legacy_torus":
            latent_state, torus_pairs, torus_angles, torus_radii = self._legacy_torus_forward(hidden)
            latent_coordinates = torus_angles
        elif self.config.latent_geometry == "torus_t3":
            latent_state, torus_pairs, torus_angles, torus_radii = self._t3_torus_forward(hidden)
            latent_coordinates = torus_angles
        elif self.config.latent_geometry == "plane_r2":
            latent_state, torus_pairs, torus_angles, torus_radii = self._plane_forward(hidden)
            latent_coordinates = latent_state
        elif self.config.latent_geometry == "sphere_s2":
            latent_state, torus_pairs, torus_angles, torus_radii = self._sphere_forward(hidden)
            latent_coordinates = latent_state
        elif self.config.latent_geometry == "hypercube_r3":
            latent_state, torus_pairs, torus_angles, torus_radii = self._hypercube_forward(hidden)
            latent_coordinates = latent_state
        else:
            latent_state, torus_pairs, torus_angles, torus_radii = self._euclidean_forward(hidden)
            latent_coordinates = latent_state
        global_style = self._masked_style_mean(latent_state, phrase_mask)
        local_embedding = self.local_projection(latent_state)
        global_embedding = self.global_projection(global_style).unsqueeze(1)
        torus_embedding = self.output_norm(local_embedding + global_embedding)
        torus_embedding = self.dropout(torus_embedding)
        torus_embedding = torus_embedding * phrase_mask.unsqueeze(-1).to(torus_embedding.dtype)
        return TorusLatentOutput(
            torus_embedding=torus_embedding,
            torus_pairs=torus_pairs,
            torus_angles=torus_angles,
            torus_radii=torus_radii,
            global_style=global_style,
            latent_coordinates=latent_coordinates,
            latent_state=latent_state,
            latent_geometry=self.config.latent_geometry,
            axis_labels=self.axis_labels,
        )
