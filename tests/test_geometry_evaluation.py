"""Tests for latent geometry evaluation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from evaluation.geometry import run_geometry_evaluation, score_geometry
from preprocessing import generate_toy_dataset, prepare_dataset
from training.conductor_targets import DEFAULT_CONDUCTOR_TARGET_VOCAB_SIZES
from training.train_conductor import run_conductor_training
from training.train_torus import run_torus_training


def _toy_conductor_config(root: Path) -> dict:
    return {
        "experiment_name": "toy_conductor",
        "seed": 11,
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
        "generation": {"temperature": 1.0, "top_k": 0, "top_p": 0.95},
        "evaluation": {"sample_count": 4, "metrics": ["recurrence", "cadence_stability"]},
    }


def _toy_t3_config(root: Path) -> dict:
    return {
        "experiment_name": "toy_t3",
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
            "architecture": "decoder_transformer_with_conductor_torus_t3",
            "d_model": 64,
            "num_layers": 2,
            "num_heads": 4,
            "dim_feedforward": 128,
            "dropout": 0.1,
            "conductor_layers": 2,
            "conductor_heads": 4,
            "conductor_dim_feedforward": 128,
            "conductor_dropout": 0.1,
            "latent_geometry": "torus_t3",
            "torus_axis_count": 3,
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
            "output_dir": str(root / "torus_t3_outputs"),
        },
        "generation": {"temperature": 1.0, "top_k": 0, "top_p": 0.95},
        "evaluation": {"sample_count": 4, "metrics": ["recurrence", "cadence_stability", "structural_stress"]},
    }


class GeometryEvaluationTests(unittest.TestCase):
    def test_score_geometry_recognizes_better_torus_alignment(self) -> None:
        source_states = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            dtype=torch.float32,
        )
        latent_coordinates = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [2.0, 0.1, 0.0]],
            dtype=torch.float32,
        )

        metrics = score_geometry(
            piece_id="demo",
            source_states=source_states,
            latent_coordinates=latent_coordinates,
            geometry_kind="euclidean_r3",
        )

        self.assertLess(metrics.structural_stress, 0.2)
        self.assertGreater(metrics.trustworthiness, 0.9)

    def test_run_geometry_evaluation_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            generate_toy_dataset(root / "raw", num_pieces=5, seed=5)
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

            torus_config = _toy_t3_config(root)
            torus_config["training"]["init_checkpoint"] = str(root / "conductor_outputs" / "latest.pt")
            run_torus_training(
                torus_config,
                config_path=root / "toy_t3.yaml",
                dry_run=False,
                max_steps_override=1,
            )

            result = run_geometry_evaluation(
                root / "torus_t3_outputs" / "latest.pt",
                processed_dir=root / "processed",
                splits_dir=root / "splits",
                split="val",
                limit_pieces=1,
                device="cpu",
                output_dir=root / "geometry_eval",
            )

            self.assertEqual(result["summary"]["piece_count"], 1)
            self.assertTrue((root / "geometry_eval" / "geometry_summary.json").exists())
            self.assertTrue((root / "geometry_eval" / "geometry_metrics.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
