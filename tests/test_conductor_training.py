"""Tests for the Phase 2 conductor integration stage."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from losses.motif import motif_recurrence_loss
from losses.phrase_boundary import phrase_boundary_loss
from preprocessing import generate_toy_dataset, prepare_dataset
from training.conductor_targets import DEFAULT_CONDUCTOR_TARGET_VOCAB_SIZES, derive_phrase_control_targets
from training.train_baseline import run_baseline_training
from training.train_conductor import run_conductor_training
from tokenization import load_piece_example
from tokenization.dataset import EventBlock, PieceExample


def _toy_conductor_config(root: Path) -> dict:
    """Build a compact conductor-stage config for fast smoke tests."""
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
            "max_steps": 2,
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


class ConductorTrainingTests(unittest.TestCase):
    def test_motif_recurrence_loss_rewards_matching_positive_pairs(self) -> None:
        phrase_hidden = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                ]
            ],
            dtype=torch.float32,
        )
        recurrence_targets = torch.tensor([[0, 1, 0]], dtype=torch.long)

        result = motif_recurrence_loss(phrase_hidden, recurrence_targets)

        self.assertGreater(result.positive_count, 0)
        self.assertGreater(result.negative_count, 0)
        self.assertGreaterEqual(float(result.total_loss.item()), 0.0)

    def test_phrase_boundary_loss_upweights_boundary_errors(self) -> None:
        logits = torch.tensor(
            [
                [
                    [0.0, 5.0, 0.0, 0.0, 0.0],
                    [0.0, 5.0, 0.0, 0.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor([[1, 4]], dtype=torch.long)

        unweighted = phrase_boundary_loss(logits, targets)
        weighted = phrase_boundary_loss(
            logits,
            targets,
            class_weights=[0.0, 1.0, 1.0, 1.0, 3.0],
        )

        self.assertEqual(weighted.boundary_token_count, 1)
        self.assertGreater(float(weighted.total_loss.item()), float(unweighted.total_loss.item()))

    def test_phrase_control_targets_are_derived_for_processed_piece(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            generate_toy_dataset(root / "raw", num_pieces=4, seed=5)
            prepare_dataset(
                raw_dir=root / "raw",
                processed_dir=root / "processed",
                splits_dir=root / "splits",
                train_ratio=0.5,
                val_ratio=0.25,
                seed=2,
            )

            split_ids = json.loads((root / "splits" / "train.json").read_text(encoding="utf-8"))["piece_ids"]
            example = load_piece_example(root / "processed" / f"{split_ids[0]}.json", duration_bins=8, velocity_bins=4)
            targets = derive_phrase_control_targets(example)

            self.assertEqual(len(targets.phrase_ids), max(0, len(example) - 1))
            self.assertEqual(len(targets.phrase_ranges), len(targets.targets["recurrence"]))
            self.assertTrue(all(value in (0, 1) for value in targets.targets["recurrence"]))
            self.assertTrue(all(0 <= value < 12 for value in targets.targets["harmonic_zone"]))

    def test_phrase_control_targets_require_rhythmic_match_for_recurrence(self) -> None:
        phrase_zero = [
            EventBlock(60, 1, 2, 1, 0, 0, 0, 0, 1),
            EventBlock(62, 1, 2, 1, 4, 0, 0, 0, 0),
            EventBlock(64, 2, 2, 1, 8, 0, 0, 0, 0),
            EventBlock(65, 1, 2, 1, 12, 0, 0, 0, 2),
        ]
        phrase_one = [
            EventBlock(60, 3, 2, 2, 0, 1, 0, 0, 1),
            EventBlock(62, 3, 2, 2, 1, 1, 0, 0, 0),
            EventBlock(64, 1, 2, 2, 2, 1, 0, 0, 0),
            EventBlock(65, 1, 2, 2, 3, 1, 0, 0, 2),
        ]
        phrase_two = [
            EventBlock(60, 1, 2, 3, 0, 2, 0, 0, 1),
            EventBlock(62, 1, 2, 3, 4, 2, 0, 0, 0),
            EventBlock(64, 2, 2, 3, 8, 2, 0, 0, 0),
            EventBlock(65, 1, 2, 3, 12, 2, 0, 0, 2),
        ]
        example = PieceExample(
            piece_id="synthetic_recurrence",
            event_blocks=phrase_zero + phrase_one + phrase_two,
            phrase_boundaries=[1, 2, 3],
            metadata={},
        )

        targets = derive_phrase_control_targets(example)

        self.assertEqual(targets.targets["recurrence"], [0, 0, 1])

    def test_run_conductor_training_dry_run_returns_joint_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            generate_toy_dataset(root / "raw", num_pieces=5, seed=6)
            prepare_dataset(
                raw_dir=root / "raw",
                processed_dir=root / "processed",
                splits_dir=root / "splits",
                train_ratio=0.6,
                val_ratio=0.2,
                seed=3,
            )
            config = _toy_conductor_config(root)

            result = run_conductor_training(
                config,
                config_path=root / "toy_conductor.yaml",
                dry_run=True,
            )

            self.assertEqual(result["mode"], "dry_run")
            self.assertEqual(result["objective"], "joint")
            self.assertGreater(result["train_metrics"]["loss"], 0.0)
            self.assertGreater(result["train_metrics"]["conductor_loss"], 0.0)
            self.assertGreaterEqual(result["train_metrics"]["motif_loss"], 0.0)
            self.assertGreaterEqual(result["train_metrics"]["boundary_loss"], 0.0)
            self.assertEqual(result["train_batch"]["shape"][0], 2)

    def test_conductor_training_can_initialize_from_baseline_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            generate_toy_dataset(root / "raw", num_pieces=5, seed=7)
            prepare_dataset(
                raw_dir=root / "raw",
                processed_dir=root / "processed",
                splits_dir=root / "splits",
                train_ratio=0.6,
                val_ratio=0.2,
                seed=3,
            )

            baseline_config = {
                "experiment_name": "toy_baseline",
                "seed": 7,
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
                    "architecture": "decoder_transformer",
                    "d_model": 64,
                    "num_layers": 2,
                    "num_heads": 4,
                    "dim_feedforward": 128,
                    "dropout": 0.1,
                    "use_conductor": False,
                    "use_torus": False,
                    "use_tension": False,
                    "use_refiner": False,
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
                    "output_dir": str(root / "baseline_outputs"),
                },
                "generation": {
                    "temperature": 1.0,
                    "top_k": 0,
                    "top_p": 0.95,
                },
                "evaluation": {
                    "sample_count": 4,
                    "metrics": ["perplexity"],
                },
            }
            baseline_result = run_baseline_training(
                baseline_config,
                config_path=root / "toy_baseline.yaml",
                dry_run=False,
                max_steps_override=1,
            )
            self.assertEqual(baseline_result["mode"], "train")

            conductor_config = _toy_conductor_config(root)
            conductor_config["training"]["init_checkpoint"] = str(root / "baseline_outputs" / "latest.pt")
            result = run_conductor_training(
                conductor_config,
                config_path=root / "toy_conductor.yaml",
                dry_run=True,
            )

            self.assertEqual(
                result["initialization"]["initialized_from"],
                str(root / "baseline_outputs" / "latest.pt"),
            )
            self.assertTrue(any("conductor" in key for key in result["initialization"]["missing_keys"]))


if __name__ == "__main__":
    unittest.main()
