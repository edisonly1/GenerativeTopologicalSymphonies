"""Losses for torus normalization and continuity."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

TORUS_GEOMETRIES = frozenset({"legacy_torus", "torus_t3"})
EUCLIDEAN_GEOMETRIES = frozenset({"euclidean_r3", "plane_r2", "hypercube_r3"})
SPHERE_GEOMETRIES = frozenset({"sphere_s2"})


@dataclass(slots=True)
class TorusLossOutput:
    """Structured torus-regularization output."""

    total_loss: Tensor
    circle_loss: Tensor
    smoothness_loss: Tensor
    geometry_loss: Tensor
    dispersion_loss: Tensor
    valid_phrase_count: int
    valid_transition_count: int


def _zero(reference: Tensor) -> Tensor:
    """Return a differentiable zero tensor on the reference device."""
    return reference.sum() * 0.0


def _pairwise_torus_distances(coordinates: Tensor) -> Tensor:
    """Compute wrap-aware pairwise torus distances for one phrase set."""
    delta = coordinates.unsqueeze(1) - coordinates.unsqueeze(0)
    wrapped = torch.atan2(torch.sin(delta), torch.cos(delta))
    return torch.linalg.vector_norm(wrapped, dim=-1)


def _pairwise_euclidean_distances(coordinates: Tensor) -> Tensor:
    """Compute Euclidean pairwise distances for one phrase set."""
    return torch.cdist(coordinates, coordinates, p=2)


def _pairwise_sphere_distances(coordinates: Tensor, *, eps: float = 1e-6) -> Tensor:
    """Compute great-circle distances on the unit sphere."""
    normalized = coordinates / torch.linalg.vector_norm(
        coordinates,
        dim=-1,
        keepdim=True,
    ).clamp(min=eps)
    cosine = (normalized.unsqueeze(1) * normalized.unsqueeze(0)).sum(dim=-1).clamp(
        min=-1.0 + eps,
        max=1.0 - eps,
    )
    return torch.arccos(cosine)


def _upper_triangle_values(matrix: Tensor) -> Tensor:
    """Extract the strict upper-triangle entries of a square distance matrix."""
    if matrix.shape[0] <= 1:
        return matrix.new_empty((0,))
    indices = torch.triu_indices(matrix.shape[0], matrix.shape[1], offset=1, device=matrix.device)
    return matrix[indices[0], indices[1]]


def _normalize_distance_vector(values: Tensor, *, eps: float = 1e-6) -> Tensor:
    """Normalize a distance vector by its mean non-zero scale."""
    if values.numel() == 0:
        return values
    scale = values.mean().clamp(min=eps)
    return values / scale


def _geometry_matching_loss(
    source_states: Tensor,
    latent_coordinates: Tensor,
    phrase_mask: Tensor,
    *,
    geometry_kind: str,
) -> Tensor:
    """Match latent pairwise distances to source phrase-state distances."""
    sample_losses: list[Tensor] = []
    for batch_index in range(source_states.shape[0]):
        valid_mask = phrase_mask[batch_index]
        phrase_count = int(valid_mask.sum().item())
        if phrase_count <= 1:
            continue
        source = source_states[batch_index][valid_mask]
        latent = latent_coordinates[batch_index][valid_mask]
        source_distances = _pairwise_euclidean_distances(source)
        if geometry_kind in EUCLIDEAN_GEOMETRIES:
            latent_distances = _pairwise_euclidean_distances(latent)
        elif geometry_kind in SPHERE_GEOMETRIES:
            latent_distances = _pairwise_sphere_distances(latent)
        else:
            latent_distances = _pairwise_torus_distances(latent)
        source_vector = _upper_triangle_values(source_distances)
        latent_vector = _upper_triangle_values(latent_distances)
        if source_vector.numel() == 0:
            continue
        normalized_source = _normalize_distance_vector(source_vector)
        normalized_latent = _normalize_distance_vector(latent_vector)
        sample_losses.append(torch.mean((normalized_latent - normalized_source) ** 2))
    if not sample_losses:
        return _zero(source_states)
    return torch.stack(sample_losses).mean()


def _dispersion_loss(
    latent_coordinates: Tensor,
    torus_angles: Tensor,
    phrase_mask: Tensor,
    *,
    geometry_kind: str,
    min_axis_variance: float,
) -> Tensor:
    """Encourage latent axes to use their available dimensionality."""
    if min_axis_variance <= 0.0:
        return _zero(latent_coordinates)
    sample_losses: list[Tensor] = []
    for batch_index in range(phrase_mask.shape[0]):
        valid_mask = phrase_mask[batch_index]
        phrase_count = int(valid_mask.sum().item())
        if phrase_count <= 1:
            continue
        if geometry_kind in TORUS_GEOMETRIES:
            angles = torus_angles[batch_index][valid_mask]
            coordinates = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        else:
            coordinates = latent_coordinates[batch_index][valid_mask]
        centered = coordinates - coordinates.mean(dim=0, keepdim=True)
        axis_variance = centered.pow(2).mean(dim=0)
        sample_losses.append(torch.relu(min_axis_variance - axis_variance).mean())
    if not sample_losses:
        return _zero(latent_coordinates)
    return torch.stack(sample_losses).mean()


def torus_losses(
    torus_radii: Tensor,
    torus_angles: Tensor,
    phrase_mask: Tensor,
    *,
    geometry_kind: str = "legacy_torus",
    source_states: Tensor | None = None,
    latent_coordinates: Tensor | None = None,
    circle_weight: float = 0.0,
    smooth_weight: float = 0.0,
    geometry_weight: float = 0.0,
    dispersion_weight: float = 0.0,
    min_axis_variance: float = 0.0,
) -> TorusLossOutput:
    """Apply unit-circle and wrap-aware smoothness penalties to torus states."""
    if geometry_kind in EUCLIDEAN_GEOMETRIES:
        if latent_coordinates is None:
            raise ValueError("Euclidean-family latent losses require latent_coordinates.")
        valid_phrase_mask = phrase_mask.unsqueeze(-1).expand_as(latent_coordinates)
        valid_phrase_count = int(valid_phrase_mask.sum().item())
        circle_loss = _zero(latent_coordinates)
        transition_mask = phrase_mask[:, 1:] & phrase_mask[:, :-1]
        valid_transition_count = int(transition_mask.sum().item())
        if valid_transition_count > 0:
            euclidean_delta = latent_coordinates[:, 1:] - latent_coordinates[:, :-1]
            smoothness_error = euclidean_delta.square().mean(dim=-1)
            smoothness_loss = smoothness_error[transition_mask].mean()
        else:
            smoothness_loss = _zero(latent_coordinates)
    elif geometry_kind in SPHERE_GEOMETRIES:
        if latent_coordinates is None:
            raise ValueError("Sphere latent losses require latent_coordinates.")
        valid_phrase_mask = phrase_mask.unsqueeze(-1).expand_as(latent_coordinates)
        valid_phrase_count = int(valid_phrase_mask.sum().item())
        radii = torch.linalg.vector_norm(latent_coordinates, dim=-1)
        if valid_phrase_count > 0:
            circle_loss = ((radii - 1.0) ** 2)[phrase_mask].mean()
        else:
            circle_loss = _zero(latent_coordinates)
        transition_mask = phrase_mask[:, 1:] & phrase_mask[:, :-1]
        valid_transition_count = int(transition_mask.sum().item())
        if valid_transition_count > 0:
            current = latent_coordinates[:, :-1]
            nxt = latent_coordinates[:, 1:]
            current = current / torch.linalg.vector_norm(current, dim=-1, keepdim=True).clamp(min=1e-6)
            nxt = nxt / torch.linalg.vector_norm(nxt, dim=-1, keepdim=True).clamp(min=1e-6)
            cosine = (current * nxt).sum(dim=-1).clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
            smoothness_error = torch.arccos(cosine).square()
            smoothness_loss = smoothness_error[transition_mask].mean()
        else:
            smoothness_loss = _zero(latent_coordinates)
    else:
        if latent_coordinates is None:
            latent_coordinates = torus_angles
        valid_phrase_mask = phrase_mask.unsqueeze(-1).expand_as(torus_radii)
        valid_phrase_count = int(valid_phrase_mask.sum().item())

        if valid_phrase_count > 0:
            radius_error = (torus_radii - 1.0) ** 2
            circle_loss = radius_error[valid_phrase_mask].mean()
        else:
            circle_loss = _zero(torus_radii)

        transition_mask = phrase_mask[:, 1:] & phrase_mask[:, :-1]
        valid_transition_count = int(transition_mask.sum().item())
        if valid_transition_count > 0:
            angle_delta = torus_angles[:, 1:] - torus_angles[:, :-1]
            wrapped_delta = torch.atan2(torch.sin(angle_delta), torch.cos(angle_delta))
            smoothness_error = wrapped_delta.square().mean(dim=-1)
            smoothness_loss = smoothness_error[transition_mask].mean()
        else:
            smoothness_loss = _zero(torus_angles)

    if source_states is not None and latent_coordinates is not None and geometry_weight > 0.0:
        geometry_loss = _geometry_matching_loss(
            source_states,
            latent_coordinates,
            phrase_mask,
            geometry_kind=geometry_kind,
        )
    else:
        reference = latent_coordinates if latent_coordinates is not None else torus_angles
        geometry_loss = _zero(reference)

    if latent_coordinates is not None and dispersion_weight > 0.0:
        dispersion_loss = _dispersion_loss(
            latent_coordinates,
            torus_angles,
            phrase_mask,
            geometry_kind=geometry_kind,
            min_axis_variance=min_axis_variance,
        )
    else:
        reference = latent_coordinates if latent_coordinates is not None else torus_angles
        dispersion_loss = _zero(reference)

    total_loss = circle_weight * circle_loss + smooth_weight * smoothness_loss
    total_loss = total_loss + geometry_weight * geometry_loss + dispersion_weight * dispersion_loss
    return TorusLossOutput(
        total_loss=total_loss,
        circle_loss=circle_loss,
        smoothness_loss=smoothness_loss,
        geometry_loss=geometry_loss,
        dispersion_loss=dispersion_loss,
        valid_phrase_count=valid_phrase_count,
        valid_transition_count=valid_transition_count,
    )
