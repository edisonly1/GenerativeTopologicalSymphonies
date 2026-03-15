"""Tests for the Phase 4 harmonic tension regularization stage."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from losses.tension import tension_regularization_loss
from preprocessing import generate_toy_dataset, prepare_dataset
from training.conductor_targets import DEFAULT_CONDUCTOR_TARGET_VOCAB_SIZES
from training.train_conductor import run_conductor_training
from training.train_tension import run_tension_training
from training.train_torus import run_torus_training


def _toy_conductor_config(root: Path) -> dict:
    """Build a compact conductor config for staged initialization tests."""
    return {
        "experiment_name": "toy_conductor",
        "seed": 19,
        "data": {
            "raw_dir": str(root / "raw"),
            "processed_dir": str(root / "processed"),
            "splits_dir": str(root / "splits"),
            "eval_dir": str(root / "eval"),
            "quantization": "sixteenth",
            "phrase_strategy": "bars_4",
        },
        "tokenization": {
            "pitch_mode": "absolute",
            "pitch_vocab_size": 128,
            "duration_bins": 8,
            "velocity_bins": 4,
            "bar_position_bins": 16,
            "instrument_bins": 129,
            "harmony_bins": 1,
            "phrase_flag_bins": 4,
            "include_harmony": True,
            "include_phrase_flags": True,
        },
        "model": {
            "architecture": "decoder_transformer_with_conductor",
            "d_model": 64,
            "num_layers": 2,
            "num_heads": 4,
            "dim_feedforward": 128,
            "dropout": 0.1,
            "conductor_layers": 2,
            "conductor_heads": 4,
            "conductor_dim_feedforward": 128,
            "conductor_dropout": 0.1,
            "use_conductor": True,
            "use_torus": False,
            "use_tension": False,
            "use_refiner": False,
        },
        "conductor_targets": DEFAULT_CONDUCTOR_TARGET_VOCAB_SIZES.copy(),
        "losses": {
            "reconstruction_weight": 1.0,
            "conductor_weight": 0.2,
            "motif_weight": 0.1,
            "phrase_boundary_weight": 0.15,
            "phrase_boundary_class_weights": [0.0, 1.0, 2.5, 2.5, 3.0],
            "conductor_target_weights": {
                "recurrence": 2.0,
                "tension": 1.0,
                "density": 1.0,
                "cadence": 1.0,
                "harmonic_zone": 1.0,
            },
        },
        "training": {
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "max_steps": 1,
            "sequence_window": 8,
            "sequence_hop": 4,
            "min_sequence_length": 2,
            "num_workers": 0,
            "gradient_clip_norm": 1.0,
            "log_every": 1,
            "validate_every": 1,
            "checkpoint_every": 1,
            "eval_batches": 1,
            "device": "cpu",
            "cache_examples": True,
            "init_checkpoint": None,
            "output_dir": str(root / "conductor_outputs"),
        },
        "generation": {
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 0.95,
        },
        "evaluation": {
            "sample_count": 4,
            "metrics": ["recurrence", "cadence_stability"],
        },
    }


def _toy_torus_config(root: Path) -> dict:
    """Build a compact torus config for staged initialization tests."""
    return {
        "experiment_name": "toy_torus",
        "seed": 23,
        "data": {
            "raw_dir": str(root / "raw"),
            "processed_dir": str(root / "processed"),
            "splits_dir": str(root / "splits"),
            "eval_dir": str(root / "eval"),
            "quantization": "sixteenth",
            "phrase_strategy": "bars_4",
        },
        "tokenization": {
            "pitch_mode": "absolute",
            "pitch_vocab_size": 128,
            "duration_bins": 8,
            "velocity_bins": 4,
            "bar_position_bins": 16,
            "instrument_bins": 129,
            "harmony_bins": 1,
            "phrase_flag_bins": 4,
            "include_harmony": True,
            "include_phrase_flags": True,
        },
        "model": {
            "architecture": "decoder_transformer_with_conductor_torus",
            "d_model": 64,
            "num_layers": 2,
            "num_heads": 4,
            "dim_feedforward": 128,
            "dropout": 0.1,
            "conductor_layers": 2,
            "conductor_heads": 4,
            "conductor_dim_feedforward": 128,
            "conductor_dropout": 0.1,
            "latent_style_dim": 8,
            "torus_dropout": 0.1,
            "use_conductor": True,
            "use_torus": True,
            "use_tension": False,
            "use_refiner": False,
        },
        "conductor_targets": DEFAULT_CONDUCTOR_TARGET_VOCAB_SIZES.copy(),
        "losses": {
            "reconstruction_weight": 1.0,
            "conductor_weight": 0.2,
            "motif_weight": 0.1,
            "phrase_boundary_weight": 0.15,
            "phrase_boundary_class_weights": [0.0, 1.0, 2.5, 2.5, 3.0],
            "circle_weight": 0.1,
            "smooth_weight": 0.05,
            "conductor_target_weights": {
                "recurrence": 2.0,
                "tension": 1.0,
                "density": 1.0,
                "cadence": 1.0,
                "harmonic_zone": 1.0,
            },
        },
        "training": {
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "max_steps": 1,
            "sequence_window": 8,
            "sequence_hop": 4,
            "min_sequence_length": 2,
            "num_workers": 0,
            "gradient_clip_norm": 1.0,
            "log_every": 1,
            "validate_every": 1,
            "checkpoint_every": 1,
            "eval_batches": 1,
            "device": "cpu",
            "cache_examples": True,
            "init_checkpoint": None,
            "output_dir": str(root / "torus_outputs"),
        },
        "generation": {
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 0.95,
        },
        "evaluation": {
            "sample_count": 4,
            "metrics": ["recurrence", "cadence_stability"],
        },
    }


def _toy_tension_config(root: Path) -> dict:
    """Build a compact tension config for staged initialization tests."""
    return {
        "experiment_name": "toy_tension",
        "seed": 29,
        "data": {
            "raw_dir": str(root / "raw"),
            "processed_dir": str(root / "processed"),
            "splits_dir": str(root / "splits"),
            "eval_dir": str(root / "eval"),
            "quantization": "sixteenth",
            "phrase_strategy": "bars_4",
        },
        "tokenization": {
            "pitch_mode": "absolute",
            "pitch_vocab_size": 128,
            "duration_bins": 8,
            "velocity_bins": 4,
            "bar_position_bins": 16,
            "instrument_bins": 129,
            "harmony_bins": 1,
            "phrase_flag_bins": 4,
            "include_harmony": True,
            "include_phrase_flags": True,
        },
        "model": {
            "architecture": "decoder_transformer_with_conductor_torus_tension",
            "d_model": 64,
            "num_layers": 2,
            "num_heads": 4,
            "dim_feedforward": 128,
            "dropout": 0.1,
            "conductor_layers": 2,
            "conductor_heads": 4,
            "conductor_dim_feedforward": 128,
            "conductor_dropout": 0.1,
            "latent_style_dim": 8,
            "torus_dropout": 0.1,
            "use_conductor": True,
            "use_torus": True,
            "use_tension": True,
            "use_refiner": False,
        },
        "conductor_targets": DEFAULT_CONDUCTOR_TARGET_VOCAB_SIZES.copy(),
        "losses": {
            "reconstruction_weight": 1.0,
            "conductor_weight": 0.2,
            "motif_weight": 0.1,
            "phrase_boundary_weight": 0.15,
            "phrase_boundary_class_weights": [0.0, 1.0, 2.5, 2.5, 3.0],
            "circle_weight": 0.1,
            "smooth_weight": 0.05,
            "geometry_weight": 0.1,
            "dispersion_weight": 0.05,
            "min_axis_variance": 0.05,
            "tension_weight": 0.05,
            "tension_pitch_weight": 0.35,
            "tension_rhythm_weight": 0.20,
            "tension_cadence_weight": 0.25,
            "tension_resolution_weight": 0.20,
            "tension_descent_weight": 1.0,
            "tension_monotonic_weight": 0.25,
            "tension_descent_step_size": 0.15,
            "conductor_target_weights": {
                "recurrence": 2.0,
                "tension": 1.0,
                "density": 1.0,
                "cadence": 1.0,
                "harmonic_zone": 1.0,
            },
        },
        "training": {
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "max_steps": 1,
            "sequence_window": 8,
            "sequence_hop": 4,
            "min_sequence_length": 2,
            "num_workers": 0,
            "gradient_clip_norm": 1.0,
            "log_every": 1,
            "validate_every": 1,
            "checkpoint_every": 1,
            "eval_batches": 1,
            "device": "cpu",
            "cache_examples": True,
            "init_checkpoint": None,
            "output_dir": str(root / "tension_outputs"),
        },
        "generation": {
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 0.95,
        },
        "evaluation": {
            "sample_count": 4,
            "metrics": ["recurrence", "cadence_stability"],
        },
    }


class TensionTrainingTests(unittest.TestCase):
    def test_tension_energy_prefers_stable_phrase_endings(self) -> None:
        pitch_logits = torch.full((1, 2, 129), -8.0)
        duration_logits = torch.full((1, 2, 9), -8.0)
        phrase_flag_logits = torch.full((1, 2, 5), -8.0)

        stable_pitch_token = 61
        unstable_pitch_token = 67
        long_duration_token = 8
        short_duration_token = 1

        pitch_logits[0, 0, stable_pitch_token] = 8.0
        pitch_logits[0, 1, stable_pitch_token] = 8.0
        duration_logits[0, 0, long_duration_token] = 8.0
        duration_logits[0, 1, long_duration_token] = 8.0
        phrase_flag_logits[0, 0, 1] = 8.0
        phrase_flag_logits[0, 1, 3] = 8.0

        stable_loss = tension_regularization_loss(
            pitch_logits=pitch_logits,
            duration_logits=duration_logits,
            phrase_flag_logits=phrase_flag_logits,
            phrase_ids=torch.tensor([[0, 0]], dtype=torch.long),
            phrase_mask=torch.tensor([[True]], dtype=torch.bool),
            attention_mask=torch.tensor([[True, True]], dtype=torch.bool),
            conductor_targets={
                "recurrence": torch.tensor([[0]], dtype=torch.long),
                "tension": torch.tensor([[0]], dtype=torch.long),
                "density": torch.tensor([[0]], dtype=torch.long),
                "cadence": torch.tensor([[1]], dtype=torch.long),
                "harmonic_zone": torch.tensor([[0]], dtype=torch.long),
            },
            torus_pairs=torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32),
        )

        pitch_logits[0, 1, stable_pitch_token] = -8.0
        pitch_logits[0, 1, unstable_pitch_token] = 8.0
        duration_logits[0, 1, long_duration_token] = -8.0
        duration_logits[0, 1, short_duration_token] = 8.0

        unstable_loss = tension_regularization_loss(
            pitch_logits=pitch_logits,
            duration_logits=duration_logits,
            phrase_flag_logits=phrase_flag_logits,
            phrase_ids=torch.tensor([[0, 0]], dtype=torch.long),
            phrase_mask=torch.tensor([[True]], dtype=torch.bool),
            attention_mask=torch.tensor([[True, True]], dtype=torch.bool),
            conductor_targets={
                "recurrence": torch.tensor([[0]], dtype=torch.long),
                "tension": torch.tensor([[0]], dtype=torch.long),
                "density": torch.tensor([[0]], dtype=torch.long),
                "cadence": torch.tensor([[1]], dtype=torch.long),
                "harmonic_zone": torch.tensor([[0]], dtype=torch.long),
            },
            torus_pairs=torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32),
        )

        self.assertGreater(float(unstable_loss.energy_loss.item()), float(stable_loss.energy_loss.item()))

    def test_run_tension_training_dry_run_initializes_from_torus_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            generate_toy_dataset(root / "raw", num_pieces=5, seed=9)
            prepare_dataset(
                raw_dir=root / "raw",
                processed_dir=root / "processed",
                splits_dir=root / "splits",
                train_ratio=0.6,
                val_ratio=0.2,
                seed=3,
            )

            conductor_config = _toy_conductor_config(root)
            run_conductor_training(
                conductor_config,
                config_path=root / "toy_conductor.yaml",
                dry_run=False,
                max_steps_override=1,
            )

            torus_config = _toy_torus_config(root)
            torus_config["training"]["init_checkpoint"] = str(root / "conductor_outputs" / "latest.pt")
            run_torus_training(
                torus_config,
                config_path=root / "toy_torus.yaml",
                dry_run=False,
                max_steps_override=1,
            )

            tension_config = _toy_tension_config(root)
            tension_config["training"]["init_checkpoint"] = str(root / "torus_outputs" / "latest.pt")
            result = run_tension_training(
                tension_config,
                config_path=root / "toy_tension.yaml",
                dry_run=True,
            )

            self.assertEqual(result["mode"], "dry_run")
            self.assertEqual(
                result["initialization"]["initialized_from"],
                str(root / "torus_outputs" / "latest.pt"),
            )
            self.assertGreaterEqual(result["train_metrics"]["tension_loss"], 0.0)
            self.assertGreaterEqual(result["train_metrics"]["tension_energy_loss"], 0.0)
            self.assertGreaterEqual(result["train_metrics"]["tension_descent_loss"], 0.0)
            self.assertGreaterEqual(result["train_metrics"]["torus_geometry_loss"], 0.0)
            self.assertGreaterEqual(result["train_metrics"]["torus_dispersion_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
