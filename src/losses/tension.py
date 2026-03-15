"""Losses tied to harmonic tension and resolution."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(slots=True)
class TensionLossOutput:
    """Structured output for the harmonic tension regularizer."""

    total_loss: Tensor
    energy_loss: Tensor
    descent_loss: Tensor
    monotonic_loss: Tensor
    pitch_energy: Tensor
    rhythm_energy: Tensor
    cadence_energy: Tensor
    resolution_energy: Tensor
    mean_energy: Tensor
    valid_phrase_count: int
    valid_transition_count: int


def _zero(reference: Tensor) -> Tensor:
    """Return a differentiable zero tensor on the reference device."""
    return reference.sum() * 0.0


def _pitch_class_projection(vocab_size: int, device: torch.device) -> Tensor:
    """Build a projection from pitch-token probabilities to pitch classes."""
    projection = torch.zeros((vocab_size, 12), device=device)
    if vocab_size <= 1:
        return projection
    token_ids = torch.arange(1, vocab_size, device=device)
    pitch_classes = (token_ids - 1) % 12
    projection[token_ids, pitch_classes] = 1.0
    return projection


def tension_regularization_loss(
    *,
    pitch_logits: Tensor,
    duration_logits: Tensor,
    phrase_flag_logits: Tensor,
    phrase_ids: Tensor,
    phrase_mask: Tensor,
    attention_mask: Tensor,
    conductor_targets: dict[str, Tensor],
    latent_state: Tensor | None = None,
    torus_pairs: Tensor | None = None,
    tension_vocab_size: int = 4,
    density_vocab_size: int = 4,
    pitch_weight: float = 0.35,
    rhythm_weight: float = 0.20,
    cadence_weight: float = 0.25,
    resolution_weight: float = 0.20,
    descent_weight: float = 1.0,
    monotonic_weight: float = 0.25,
    descent_step_size: float = 0.15,
    ignore_index: int = -100,
) -> TensionLossOutput:
    """Regularize phrase trajectories toward smooth tension-resolution descent."""
    if latent_state is None:
        if torus_pairs is None:
            raise ValueError("tension_regularization_loss requires latent_state or torus_pairs.")
        latent_state = torus_pairs.reshape(torus_pairs.shape[0], torus_pairs.shape[1], -1)
    batch_size, phrase_count = phrase_mask.shape
    device = pitch_logits.device
    pitch_projection = _pitch_class_projection(pitch_logits.shape[-1], device)
    duration_values = torch.arange(duration_logits.shape[-1], device=device, dtype=duration_logits.dtype)
    max_duration_value = max(1.0, float(duration_logits.shape[-1] - 2))

    harmonic_targets = conductor_targets["harmonic_zone"]
    tension_targets = conductor_targets["tension"]
    density_targets = conductor_targets["density"]
    cadence_targets = conductor_targets["cadence"]
    valid_phrase_mask = (
        phrase_mask
        & (harmonic_targets != ignore_index)
        & (tension_targets != ignore_index)
        & (density_targets != ignore_index)
        & (cadence_targets != ignore_index)
    )
    valid_phrase_count = int(valid_phrase_mask.sum().item())
    tension_target_values = (
        tension_targets.clamp(min=0).to(pitch_logits.dtype) / max(1, tension_vocab_size - 1)
    ) * valid_phrase_mask.to(pitch_logits.dtype)
    density_target_values = (
        density_targets.clamp(min=0).to(pitch_logits.dtype) / max(1, density_vocab_size - 1)
    ) * valid_phrase_mask.to(pitch_logits.dtype)
    cadence_target_values = cadence_targets.clamp(min=0).to(pitch_logits.dtype) * valid_phrase_mask.to(
        pitch_logits.dtype
    )

    stable_mass_rows: list[Tensor] = []
    rhythmic_density_rows: list[Tensor] = []
    cadence_strength_rows: list[Tensor] = []
    pitch_energy_rows: list[Tensor] = []
    rhythm_energy_rows: list[Tensor] = []
    cadence_energy_rows: list[Tensor] = []
    resolution_energy_rows: list[Tensor] = []
    mean_energy_rows: list[Tensor] = []
    state_rows: list[Tensor] = []

    for batch_index in range(batch_size):
        batch_stable_mass: list[Tensor] = []
        batch_rhythmic_density: list[Tensor] = []
        batch_cadence_strength: list[Tensor] = []
        batch_pitch_energy: list[Tensor] = []
        batch_rhythm_energy: list[Tensor] = []
        batch_cadence_energy: list[Tensor] = []
        batch_resolution_energy: list[Tensor] = []
        batch_mean_energy: list[Tensor] = []
        batch_states: list[Tensor] = []
        for local_phrase_index in range(phrase_count):
            zero = _zero(pitch_logits[batch_index, 0])
            if not bool(valid_phrase_mask[batch_index, local_phrase_index].item()):
                batch_stable_mass.append(zero)
                batch_rhythmic_density.append(zero)
                batch_cadence_strength.append(zero)
                batch_pitch_energy.append(zero)
                batch_rhythm_energy.append(zero)
                batch_cadence_energy.append(zero)
                batch_resolution_energy.append(zero)
                batch_mean_energy.append(zero)
                batch_states.append(
                    torch.cat(
                        [
                            latent_state[batch_index, local_phrase_index].reshape(-1),
                            zero.repeat(3),
                        ],
                    ),
                )
                continue

            token_mask = attention_mask[batch_index] & (phrase_ids[batch_index] == local_phrase_index)
            if not bool(token_mask.any().item()):
                batch_stable_mass.append(zero)
                batch_rhythmic_density.append(zero)
                batch_cadence_strength.append(zero)
                batch_pitch_energy.append(zero)
                batch_rhythm_energy.append(zero)
                batch_cadence_energy.append(zero)
                batch_resolution_energy.append(zero)
                batch_mean_energy.append(zero)
                batch_states.append(
                    torch.cat(
                        [
                            latent_state[batch_index, local_phrase_index].reshape(-1),
                            zero.repeat(3),
                        ],
                    ),
                )
                continue

            phrase_pitch_probs = torch.softmax(pitch_logits[batch_index, token_mask], dim=-1)
            pitch_class_histogram = (phrase_pitch_probs @ pitch_projection).mean(dim=0)
            harmonic_zone = int(harmonic_targets[batch_index, local_phrase_index].item())
            stable_mask = torch.zeros(12, device=device, dtype=phrase_pitch_probs.dtype)
            stable_mask[harmonic_zone] = 1.0
            stable_mask[(harmonic_zone + 4) % 12] = 1.0
            stable_mask[(harmonic_zone + 7) % 12] = 1.0
            stable_mass = (pitch_class_histogram * stable_mask).sum()

            phrase_duration_probs = torch.softmax(duration_logits[batch_index, token_mask], dim=-1)
            expected_duration = (phrase_duration_probs * duration_values).sum(dim=-1).clamp(min=1.0)
            rhythmic_density = (1.0 / expected_duration).mean()

            token_indices = torch.nonzero(token_mask, as_tuple=False).squeeze(-1)
            final_token_index = int(token_indices[-1].item())
            final_pitch_probs = torch.softmax(pitch_logits[batch_index, final_token_index], dim=-1)
            final_pitch_histogram = final_pitch_probs @ pitch_projection
            final_stable_mass = (final_pitch_histogram * stable_mask).sum()
            final_duration_probs = torch.softmax(duration_logits[batch_index, final_token_index], dim=-1)
            final_duration_norm = ((final_duration_probs * duration_values).sum() - 1.0) / max_duration_value
            final_duration_norm = final_duration_norm.clamp(min=0.0, max=1.0)
            final_phrase_flag_probs = torch.softmax(phrase_flag_logits[batch_index, final_token_index], dim=-1)
            end_probability = final_phrase_flag_probs[3:5].sum() if final_phrase_flag_probs.shape[0] >= 5 else zero
            cadence_strength = (final_stable_mass + final_duration_norm + end_probability) / 3.0

            tension_target = torch.tensor(
                float(tension_targets[batch_index, local_phrase_index].item())
                / max(1, tension_vocab_size - 1),
                device=device,
                dtype=stable_mass.dtype,
            )
            density_target = torch.tensor(
                float(density_targets[batch_index, local_phrase_index].item())
                / max(1, density_vocab_size - 1),
                device=device,
                dtype=stable_mass.dtype,
            )
            cadence_target = cadence_targets[batch_index, local_phrase_index].to(stable_mass.dtype)

            pitch_energy = 1.0 - stable_mass
            rhythm_energy = torch.abs(rhythmic_density - density_target)
            cadence_energy = torch.abs(cadence_strength - cadence_target)
            base_energy = (
                pitch_weight * pitch_energy
                + rhythm_weight * rhythm_energy
                + cadence_weight * cadence_energy
            )
            resolution_energy = torch.abs(base_energy - tension_target)
            mean_energy = base_energy + resolution_weight * resolution_energy

            batch_stable_mass.append(stable_mass)
            batch_rhythmic_density.append(rhythmic_density)
            batch_cadence_strength.append(cadence_strength)
            batch_pitch_energy.append(pitch_energy)
            batch_rhythm_energy.append(rhythm_energy)
            batch_cadence_energy.append(cadence_energy)
            batch_resolution_energy.append(resolution_energy)
            batch_mean_energy.append(mean_energy)
            batch_states.append(
                torch.cat(
                    [
                        latent_state[batch_index, local_phrase_index].reshape(-1),
                        torch.stack([stable_mass, rhythmic_density, cadence_strength]),
                    ],
                ),
            )

        stable_mass_rows.append(torch.stack(batch_stable_mass))
        rhythmic_density_rows.append(torch.stack(batch_rhythmic_density))
        cadence_strength_rows.append(torch.stack(batch_cadence_strength))
        pitch_energy_rows.append(torch.stack(batch_pitch_energy))
        rhythm_energy_rows.append(torch.stack(batch_rhythm_energy))
        cadence_energy_rows.append(torch.stack(batch_cadence_energy))
        resolution_energy_rows.append(torch.stack(batch_resolution_energy))
        mean_energy_rows.append(torch.stack(batch_mean_energy))
        state_rows.append(torch.stack(batch_states))

    state = torch.stack(state_rows)
    if not state.requires_grad:
        state.requires_grad_(True)
    stable_mass_values = state[..., -3]
    rhythmic_density_values = state[..., -2]
    cadence_strength_values = state[..., -1]
    pitch_energy = 1.0 - stable_mass_values
    rhythm_energy = torch.abs(rhythmic_density_values - density_target_values)
    cadence_energy = torch.abs(cadence_strength_values - cadence_target_values)
    base_energy = (
        pitch_weight * pitch_energy
        + rhythm_weight * rhythm_energy
        + cadence_weight * cadence_energy
    )
    resolution_energy = torch.abs(base_energy - tension_target_values)
    mean_energy = base_energy + resolution_weight * resolution_energy

    if valid_phrase_count > 0:
        energy_loss = mean_energy[valid_phrase_mask].mean()
        pitch_energy_mean = pitch_energy[valid_phrase_mask].mean()
        rhythm_energy_mean = rhythm_energy[valid_phrase_mask].mean()
        cadence_energy_mean = cadence_energy[valid_phrase_mask].mean()
        resolution_energy_mean = resolution_energy[valid_phrase_mask].mean()
    else:
        energy_loss = _zero(state)
        pitch_energy_mean = _zero(state)
        rhythm_energy_mean = _zero(state)
        cadence_energy_mean = _zero(state)
        resolution_energy_mean = _zero(state)

    gradient = torch.autograd.grad(
        mean_energy.sum(),
        state,
        create_graph=True,
        retain_graph=True,
    )[0]
    valid_transition_mask = valid_phrase_mask[:, :-1] & valid_phrase_mask[:, 1:]
    valid_transition_count = int(valid_transition_mask.sum().item())
    if valid_transition_count > 0:
        predicted_next_state = state[:, :-1] - descent_step_size * gradient[:, :-1]
        actual_next_state = state[:, 1:]
        descent_error = (actual_next_state - predicted_next_state).pow(2).mean(dim=-1)
        descent_loss = descent_error[valid_transition_mask].mean()
        monotonic_error = F.relu(mean_energy[:, 1:] - mean_energy[:, :-1])
        monotonic_loss = monotonic_error[valid_transition_mask].mean()
    else:
        descent_loss = _zero(state)
        monotonic_loss = _zero(state)

    total_loss = energy_loss + descent_weight * descent_loss + monotonic_weight * monotonic_loss
    return TensionLossOutput(
        total_loss=total_loss,
        energy_loss=energy_loss,
        descent_loss=descent_loss,
        monotonic_loss=monotonic_loss,
        pitch_energy=pitch_energy_mean,
        rhythm_energy=rhythm_energy_mean,
        cadence_energy=cadence_energy_mean,
        resolution_energy=resolution_energy_mean,
        mean_energy=energy_loss,
        valid_phrase_count=valid_phrase_count,
        valid_transition_count=valid_transition_count,
    )
