"""Tests for the baseline model and training pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from preprocessing.harmony_extract import GLOBAL_HARMONY_VOCAB
from preprocessing import generate_toy_dataset, prepare_dataset
from tokenization.dataset import EventBlock, PieceExample
from training.data import (
    WindowedAutoregressiveTokenDataset,
    _transpose_piece_example,
    create_autoregressive_dataloader,
)
from training.train_baseline import run_baseline_training


def _toy_training_config(root: Path) -> dict:
    """Build a small training config suitable for quick smoke tests."""
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
            "max_steps": 2,
            "sequence_window": 4,
            "sequence_hop": 3,
            "min_sequence_length": 2,
            "num_workers": 0,
            "gradient_clip_norm": 1.0,
            "log_every": 1,
            "validate_every": 1,
            "checkpoint_every": 1,
            "eval_batches": 1,
            "device": "cpu",
            "cache_examples": True,
            "output_dir": str(root / "outputs"),
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


class BaselineTrainingTests(unittest.TestCase):
    def test_piece_transposition_updates_pitch_and_harmony_together(self) -> None:
        example = PieceExample(
            piece_id="toy",
            event_blocks=[
                EventBlock(
                    pitch=60,
                    duration=1,
                    velocity=1,
                    bar=1,
                    bar_position=0,
                    phrase_index=0,
                    instrument=0,
                    harmony=GLOBAL_HARMONY_VOCAB["C:maj"],
                    phrase_flag=1,
                ),
                EventBlock(
                    pitch=67,
                    duration=2,
                    velocity=1,
                    bar=1,
                    bar_position=4,
                    phrase_index=0,
                    instrument=0,
                    harmony=GLOBAL_HARMONY_VOCAB["G:maj"],
                    phrase_flag=2,
                ),
            ],
            phrase_boundaries=[1],
            metadata={"key": "C:major", "chords": ["C:maj", "G:maj"]},
        )

        transposed = _transpose_piece_example(example, 2)

        self.assertEqual([block.pitch for block in transposed.event_blocks], [62, 69])
        self.assertEqual(
            [block.harmony for block in transposed.event_blocks],
            [GLOBAL_HARMONY_VOCAB["D:maj"], GLOBAL_HARMONY_VOCAB["A:maj"]],
        )
        self.assertEqual(transposed.metadata["key"], "D:major")
        self.assertEqual(transposed.metadata["chords"], ["D:maj", "A:maj"])

    def test_windowed_dataset_builds_short_training_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            processed_dir = root / "processed"
            splits_dir = root / "splits"

            generate_toy_dataset(raw_dir, num_pieces=5, seed=5)
            prepare_dataset(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                splits_dir=splits_dir,
                train_ratio=0.6,
                val_ratio=0.2,
                seed=3,
            )

            dataset = WindowedAutoregressiveTokenDataset(
                processed_dir=processed_dir,
                splits_dir=splits_dir,
                split="train",
                duration_bins=8,
                velocity_bins=4,
                sequence_window=4,
                sequence_hop=3,
                min_sequence_length=2,
                cache_examples=True,
            )
            self.assertGreater(len(dataset), 3)
            sample = dataset[0]
            self.assertLessEqual(sample["sequence_length"], 4)
            self.assertEqual(sample["window_range"][0], 0)

            batch = next(iter(create_autoregressive_dataloader(dataset, batch_size=2)))
            self.assertEqual(batch.inputs["pitch"].shape[0], 2)
            self.assertLessEqual(batch.inputs["pitch"].shape[1], 4)
            self.assertEqual(batch.targets["pitch"].shape, batch.inputs["pitch"].shape)

    def test_windowed_dataset_can_require_complete_phrase_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            processed_dir = root / "processed"
            splits_dir = root / "splits"

            generate_toy_dataset(raw_dir, num_pieces=5, seed=11)
            prepare_dataset(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                splits_dir=splits_dir,
                train_ratio=0.6,
                val_ratio=0.2,
                seed=3,
                phrase_strategy="bars_2",
            )

            dataset = WindowedAutoregressiveTokenDataset(
                processed_dir=processed_dir,
                splits_dir=splits_dir,
                split="train",
                duration_bins=8,
                velocity_bins=4,
                sequence_window=12,
                sequence_hop=12,
                min_sequence_length=2,
                phrase_aligned_windows=True,
                min_complete_phrases=1,
                min_distinct_phrases=1,
                cache_examples=True,
            )

            self.assertGreater(len(dataset), 0)
            piece_index, start, _ = dataset.window_index[0]
            full_sample = dataset.piece_dataset[piece_index]
            phrase_starts = {phrase_start for phrase_start, _ in full_sample["phrase_ranges"]}
            self.assertIn(start, phrase_starts)

            sample = dataset[0]
            self.assertGreaterEqual(sum(sample["phrase_complete"]), 1)
            self.assertGreaterEqual(len(sample["phrase_ranges"]), 1)

    def test_run_baseline_training_dry_run_returns_batch_and_loss_summary(self) -> None:
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
            config = _toy_training_config(root)

            result = run_baseline_training(config, config_path=root / "toy.yaml", dry_run=True)

            self.assertEqual(result["mode"], "dry_run")
            self.assertGreater(result["train_windows"], 0)
            self.assertGreater(result["val_windows"], 0)
            self.assertGreater(result["train_loss"], 0.0)
            self.assertEqual(result["train_batch"]["shape"][0], 2)
            self.assertLessEqual(result["train_batch"]["shape"][1], 4)

    def test_run_baseline_training_two_steps_writes_outputs(self) -> None:
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
            config = _toy_training_config(root)

            result = run_baseline_training(
                config,
                config_path=root / "toy.yaml",
                dry_run=False,
                max_steps_override=2,
            )

            self.assertEqual(result["mode"], "train")
            self.assertEqual(result["max_steps"], 2)
            self.assertTrue((root / "outputs" / "metrics.jsonl").exists())
            self.assertTrue((root / "outputs" / "latest.pt").exists())
            self.assertTrue((root / "outputs" / "checkpoints" / "best.pt").exists())
            self.assertTrue((root / "outputs" / "checkpoints" / "step_000001.pt").exists())
            self.assertTrue((root / "outputs" / "checkpoints" / "step_000002.pt").exists())

            metrics_lines = (root / "outputs" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            payloads = [json.loads(line) for line in metrics_lines]
            self.assertTrue(any(item["phase"] == "train" for item in payloads))
            self.assertTrue(any(item["phase"] == "val" for item in payloads))


if __name__ == "__main__":
    unittest.main()
