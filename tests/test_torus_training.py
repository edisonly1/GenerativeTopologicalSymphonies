"""Tests for the Phase 3 torus latent bottleneck stage."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from losses.torus import torus_losses
from models.torus_latent import TorusLatentBottleneck, TorusLatentConfig
from preprocessing import generate_toy_dataset, prepare_dataset
from training.conductor_targets import DEFAULT_CONDUCTOR_TARGET_VOCAB_SIZES
from training.train_conductor import run_conductor_training
from training.train_baseline import build_feature_vocab_sizes
from training.train_torus import (
    build_torus_model,
    maybe_initialize_from_checkpoint,
    run_torus_training,
)


def _toy_conductor_config(root: Path) -> dict:
    """Build a compact conductor config for torus-stage initialization tests."""
    return {
        "experiment_name": "toy_conductor",
        "seed": 13,
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
    """Build a compact torus config for smoke tests."""
    return {
        "experiment_name": "toy_torus",
        "seed": 17,
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
            "geometry_weight": 0.1,
            "dispersion_weight": 0.05,
            "min_axis_variance": 0.05,
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


class TorusTrainingTests(unittest.TestCase):
    def test_torus_latent_projects_to_unit_circles(self) -> None:
        module = TorusLatentBottleneck(
            config=TorusLatentConfig(d_model=8, latent_style_dim=4, dropout=0.0),
        )
        phrase_states = torch.randn(2, 3, 8)
        phrase_mask = torch.tensor([[True, True, False], [True, True, True]])

        output = module(phrase_states, phrase_mask=phrase_mask)

        valid_norms = torch.linalg.vector_norm(output.torus_pairs, dim=-1)[phrase_mask]
        self.assertTrue(torch.allclose(valid_norms, torch.ones_like(valid_norms), atol=1e-5))
        self.assertEqual(output.global_style.shape, (2, 4))

    def test_explicit_t3_latent_uses_named_axes(self) -> None:
        module = TorusLatentBottleneck(
            config=TorusLatentConfig(
                d_model=8,
                latent_geometry="torus_t3",
                torus_axis_count=3,
                dropout=0.0,
            ),
        )
        phrase_states = torch.randn(1, 4, 8)
        phrase_mask = torch.tensor([[True, True, True, False]])

        output = module(phrase_states, phrase_mask=phrase_mask)

        self.assertEqual(output.latent_geometry, "torus_t3")
        self.assertEqual(output.axis_labels, ("pitch_cycle", "rhythm_cycle", "harmony_cycle"))
        self.assertEqual(output.latent_coordinates.shape[-1], 3)
        valid_norms = torch.linalg.vector_norm(output.torus_pairs, dim=-1)[phrase_mask]
        self.assertTrue(torch.allclose(valid_norms, torch.ones_like(valid_norms), atol=1e-5))

    def test_euclidean_r3_latent_outputs_three_coordinates(self) -> None:
        module = TorusLatentBottleneck(
            config=TorusLatentConfig(
                d_model=8,
                latent_geometry="euclidean_r3",
                euclidean_dim=3,
                dropout=0.0,
            ),
        )
        phrase_states = torch.randn(2, 3, 8)
        phrase_mask = torch.tensor([[True, True, False], [True, True, True]])

        output = module(phrase_states, phrase_mask=phrase_mask)

        self.assertEqual(output.latent_geometry, "euclidean_r3")
        self.assertEqual(output.axis_labels, ("pitch_cycle", "rhythm_cycle", "harmony_cycle"))
        self.assertEqual(output.latent_coordinates.shape, (2, 3, 3))

    def test_checkpoint_initialization_uses_first_existing_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _toy_torus_config(root)
            vocab_sizes = build_feature_vocab_sizes(config["tokenization"])
            source_model = build_torus_model(config, vocab_sizes=vocab_sizes)
            checkpoint_path = root / "checkpoint.pt"
            torch.save({"model_state": source_model.state_dict()}, checkpoint_path)

            target_model = build_torus_model(config, vocab_sizes=vocab_sizes)
            result = maybe_initialize_from_checkpoint(
                target_model,
                [root / "missing.pt", checkpoint_path],
                device=torch.device("cpu"),
            )

            self.assertEqual(result["initialized_from"], str(checkpoint_path))

    def test_torus_losses_use_wrap_aware_smoothness(self) -> None:
        torus_radii = torch.ones((1, 2, 1), dtype=torch.float32)
        torus_angles = torch.tensor([[[3.04], [-3.04]]], dtype=torch.float32)
        phrase_mask = torch.tensor([[True, True]])

        loss = torus_losses(
            torus_radii,
            torus_angles,
            phrase_mask,
            circle_weight=0.1,
            smooth_weight=0.05,
        )

        self.assertLess(float(loss.smoothness_loss.item()), 0.2)
        self.assertGreaterEqual(float(loss.total_loss.item()), 0.0)

    def test_euclidean_latent_losses_use_coordinate_smoothness(self) -> None:
        latent_coordinates = torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]], dtype=torch.float32)
        torus_radii = torch.ones((1, 2, 3), dtype=torch.float32)
        torus_angles = torch.zeros((1, 2, 3), dtype=torch.float32)
        phrase_mask = torch.tensor([[True, True]])

        loss = torus_losses(
            torus_radii,
            torus_angles,
            phrase_mask,
            geometry_kind="euclidean_r3",
            latent_coordinates=latent_coordinates,
            circle_weight=0.1,
            smooth_weight=0.05,
        )

        self.assertEqual(float(loss.circle_loss.item()), 0.0)
        self.assertGreater(float(loss.smoothness_loss.item()), 0.0)

    def test_geometry_and_dispersion_losses_activate_for_torus_latent(self) -> None:
        source_states = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]],
            dtype=torch.float32,
        )
        torus_angles = torch.tensor(
            [[[0.0], [0.1], [0.2]]],
            dtype=torch.float32,
        )
        torus_radii = torch.ones((1, 3, 1), dtype=torch.float32)
        phrase_mask = torch.tensor([[True, True, True]])

        loss = torus_losses(
            torus_radii,
            torus_angles,
            phrase_mask,
            source_states=source_states,
            latent_coordinates=torus_angles,
            circle_weight=0.0,
            smooth_weight=0.0,
            geometry_weight=0.1,
            dispersion_weight=0.1,
            min_axis_variance=0.2,
        )

        self.assertGreaterEqual(float(loss.geometry_loss.item()), 0.0)
        self.assertGreaterEqual(float(loss.dispersion_loss.item()), 0.0)
        self.assertGreaterEqual(float(loss.total_loss.item()), 0.0)

    def test_run_torus_training_dry_run_initializes_from_conductor_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            generate_toy_dataset(root / "raw", num_pieces=5, seed=8)
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
            result = run_torus_training(
                torus_config,
                config_path=root / "toy_torus.yaml",
                dry_run=True,
            )

            self.assertEqual(result["mode"], "dry_run")
            self.assertEqual(
                result["initialization"]["initialized_from"],
                str(root / "conductor_outputs" / "latest.pt"),
            )
            self.assertGreater(result["train_metrics"]["torus_loss"], 0.0)
            self.assertGreaterEqual(result["train_metrics"]["torus_circle_loss"], 0.0)
            self.assertGreaterEqual(result["train_metrics"]["torus_smoothness_loss"], 0.0)
            self.assertGreaterEqual(result["train_metrics"]["torus_geometry_loss"], 0.0)
            self.assertGreaterEqual(result["train_metrics"]["torus_dispersion_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
