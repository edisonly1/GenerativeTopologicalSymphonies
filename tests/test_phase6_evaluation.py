"""Tests for the Phase 6 evaluation, refiner inference, and report packaging flow."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from evaluation.evaluate_samples import evaluate_directory
from evaluation.run_ablation_suite import run_ablation_suite
from inference.refine import refine_directory
from models.refiner import ConditionalDenoisingRefiner, RefinerConfig
from preprocessing import QuantizedEvent, QuantizedPiece, generate_toy_dataset, prepare_dataset
from preprocessing import write_quantized_piece_json
from training.train_baseline import run_baseline_training


def _make_piece(piece_id: str, pitches: list[int]) -> QuantizedPiece:
    """Create a compact quantized piece for evaluation tests."""
    note_events = [
        QuantizedEvent(
            pitch=pitch,
            velocity=90,
            instrument=0,
            channel=0,
            start_step=index * 4,
            duration_steps=4,
            bar=(index // 4) + 1,
            position=(index * 4) % 16,
            track_index=0,
        )
        for index, pitch in enumerate(pitches)
    ]
    return QuantizedPiece(
        piece_id=piece_id,
        resolution="sixteenth",
        steps_per_beat=4,
        bar_steps=16,
        time_signature="4/4",
        tempo_bpm=120.0,
        note_events=note_events,
        phrase_boundaries=[1, 3],
    )


def _toy_baseline_config(root: Path) -> dict:
    """Build a compact baseline config for ablation smoke tests."""
    return {
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
            "sample_count": 2,
            "split": "val",
            "prompt_events": 4,
            "generate_events": 6,
            "metrics": [
                "recurrence",
                "cadence_stability",
                "jump_distance",
                "pitch_class_divergence",
                "perplexity",
                "playability",
                "persistence",
            ],
        },
    }


def _toy_refiner_config() -> dict:
    """Build a tiny refiner config for inference smoke tests."""
    return {
        "seed": 13,
        "tokenization": {
            "pitch_vocab_size": 128,
            "duration_bins": 8,
            "velocity_bins": 4,
            "bar_position_bins": 16,
            "instrument_bins": 129,
            "harmony_bins": 1,
            "phrase_flag_bins": 4,
        },
        "model": {
            "d_model": 32,
            "num_layers": 2,
            "num_heads": 4,
            "dim_feedforward": 64,
            "dropout": 0.1,
            "conductor_layers": 2,
            "conductor_heads": 4,
            "conductor_dim_feedforward": 64,
            "conductor_dropout": 0.1,
            "latent_style_dim": 8,
            "torus_dropout": 0.1,
            "refiner_layers": 2,
            "refiner_heads": 4,
            "refiner_dim_feedforward": 64,
            "refiner_dropout": 0.1,
            "use_conductor": True,
            "use_torus": True,
            "use_tension": True,
            "use_refiner": True,
        },
        "training": {
            "init_checkpoint": None,
        },
    }


class PhaseSixEvaluationTests(unittest.TestCase):
    def test_evaluate_directory_with_reference_adds_extended_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            generated_dir = root / "generated" / "piece_a"
            reference_dir = root / "reference" / "piece_a"
            write_quantized_piece_json(
                _make_piece("piece_a__sample", [60, 64, 67, 69, 67, 64]),
                generated_dir / "piece.json",
            )
            write_quantized_piece_json(
                _make_piece("piece_a", [60, 62, 64, 65, 67, 69]),
                reference_dir / "piece.json",
            )

            result = evaluate_directory(
                root / "generated",
                output_dir=root / "evaluated",
                reference_dir=root / "reference",
                duration_bins=8,
                velocity_bins=4,
            )

            summary = result["summary"]
            self.assertEqual(summary["piece_count"], 1)
            self.assertIsNotNone(summary["mean_pitch_class_divergence"])
            self.assertIsNotNone(summary["mean_transition_perplexity"])
            self.assertIn("mean_large_span_rate", summary)
            self.assertIn("mean_max_persistence", summary)

    def test_refine_directory_writes_refined_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "generated" / "piece_a"
            output_dir = root / "refined"
            piece = _make_piece("piece_a__sample", [60, 64, 67, 71, 67, 64])
            write_quantized_piece_json(piece, input_dir / "piece.json")

            config = _toy_refiner_config()
            vocab_sizes = {
                "pitch": config["tokenization"]["pitch_vocab_size"] + 1,
                "duration": config["tokenization"]["duration_bins"] + 1,
                "velocity": config["tokenization"]["velocity_bins"] + 1,
                "bar_position": config["tokenization"]["bar_position_bins"] + 1,
                "instrument": config["tokenization"]["instrument_bins"] + 1,
                "harmony": config["tokenization"]["harmony_bins"] + 1,
                "phrase_flag": config["tokenization"]["phrase_flag_bins"] + 1,
            }
            refiner_model = ConditionalDenoisingRefiner(
                vocab_sizes=vocab_sizes,
                config=RefinerConfig(
                    d_model=config["model"]["d_model"],
                    num_layers=config["model"]["refiner_layers"],
                    num_heads=config["model"]["refiner_heads"],
                    dropout=config["model"]["refiner_dropout"],
                    dim_feedforward=config["model"]["refiner_dim_feedforward"],
                ),
            )
            checkpoint_path = root / "refiner.pt"
            torch.save({"model_state": refiner_model.state_dict(), "config": config}, checkpoint_path)

            result = refine_directory(
                checkpoint_path,
                input_dir=root / "generated",
                output_dir=output_dir,
                preserve_prefix_events=2,
                device="cpu",
            )

            self.assertEqual(result["piece_count"], 1)
            self.assertTrue((output_dir / "piece_a" / "piece.json").exists())
            self.assertTrue((output_dir / "piece_a" / "piece.mid").exists())

    def test_run_ablation_suite_baseline_smoke(self) -> None:
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
            baseline_config = _toy_baseline_config(root)
            baseline_config_path = root / "baseline.yaml"
            baseline_config_path.write_text(
                json.dumps(baseline_config),
                encoding="utf-8",
            )
            run_baseline_training(
                baseline_config,
                config_path=baseline_config_path,
                dry_run=False,
                max_steps_override=1,
            )

            suite_config = {
                "seed": 7,
                "data": baseline_config["data"],
                "tokenization": baseline_config["tokenization"],
                "evaluation": {
                    **baseline_config["evaluation"],
                    "output_dir": str(root / "suite"),
                },
                "stages": {
                    "baseline": {
                        "mode": "generate",
                        "config": str(baseline_config_path),
                        "checkpoint": str(root / "baseline_outputs" / "latest.pt"),
                    }
                },
            }

            report = run_ablation_suite(
                suite_config,
                config_path=baseline_config_path,
                output_dir=root / "suite",
                device="cpu",
            )

            self.assertIn("baseline", report["stages"])
            self.assertTrue((root / "suite" / "report.json").exists())
            self.assertTrue((root / "suite" / "report.md").exists())
            baseline_summary = report["stages"]["baseline"]["evaluation"]["summary"]
            self.assertIsNotNone(baseline_summary["mean_pitch_class_divergence"])


if __name__ == "__main__":
    unittest.main()
