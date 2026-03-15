"""Tests for checkpoint generation and structural evaluation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation import (
    build_matched_reference_set,
    evaluate_directory,
    run_recurrence_diagnostics,
    score_cadence_stability,
    score_recurrence,
    slice_quantized_piece,
)
from inference import generate_from_checkpoint
from preprocessing import (
    QuantizedEvent,
    QuantizedPiece,
    generate_toy_dataset,
    prepare_dataset,
    write_quantized_piece_json,
)
from tokenization import encode_piece_to_blocks
from training.train_baseline import run_baseline_training


def _toy_baseline_config(root: Path) -> dict:
    """Build a compact baseline config for generation smoke tests."""
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
            "sample_count": 4,
            "metrics": ["perplexity"],
        },
    }


class GenerationEvaluationTests(unittest.TestCase):
    def test_recurrence_and_cadence_metrics_score_repeated_phrases(self) -> None:
        piece = QuantizedPiece(
            piece_id="repeat",
            resolution="sixteenth",
            steps_per_beat=4,
            bar_steps=16,
            time_signature="4/4",
            tempo_bpm=120.0,
            note_events=[
                QuantizedEvent(
                    pitch=60,
                    velocity=90,
                    instrument=0,
                    channel=0,
                    start_step=0,
                    duration_steps=4,
                    bar=1,
                    position=0,
                    track_index=0,
                ),
                QuantizedEvent(
                    pitch=64,
                    velocity=90,
                    instrument=0,
                    channel=0,
                    start_step=16,
                    duration_steps=4,
                    bar=2,
                    position=0,
                    track_index=0,
                ),
                QuantizedEvent(
                    pitch=60,
                    velocity=90,
                    instrument=0,
                    channel=0,
                    start_step=32,
                    duration_steps=4,
                    bar=3,
                    position=0,
                    track_index=0,
                ),
                QuantizedEvent(
                    pitch=64,
                    velocity=90,
                    instrument=0,
                    channel=0,
                    start_step=48,
                    duration_steps=4,
                    bar=4,
                    position=0,
                    track_index=0,
                ),
            ],
            phrase_boundaries=[1, 3],
        )
        example = encode_piece_to_blocks(piece, duration_bins=8, velocity_bins=4)

        recurrence = score_recurrence(example)
        cadence = score_cadence_stability(example)

        self.assertEqual(recurrence.phrase_count, 2)
        self.assertGreaterEqual(recurrence.recurrent_phrase_ratio, 1.0)
        self.assertGreaterEqual(cadence.cadence_rate, 0.5)

    def test_generate_from_checkpoint_and_evaluate_directory(self) -> None:
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
            config = _toy_baseline_config(root)
            run_baseline_training(
                config,
                config_path=root / "toy_baseline.yaml",
                dry_run=False,
                max_steps_override=1,
            )

            generated = generate_from_checkpoint(
                root / "baseline_outputs" / "latest.pt",
                processed_dir=root / "processed",
                splits_dir=root / "splits",
                split="val",
                limit_pieces=1,
                prompt_events=4,
                generate_events=6,
                device="cpu",
                output_dir=root / "generated",
                seed=5,
            )
            self.assertEqual(generated["piece_count"], 1)
            item = generated["items"][0]
            self.assertTrue(Path(item["output_json"]).exists())
            self.assertTrue(Path(item["output_midi"]).exists())

            evaluation = evaluate_directory(root / "generated", output_dir=root / "evaluated")
            self.assertEqual(evaluation["summary"]["piece_count"], 1)
            self.assertTrue((root / "evaluated" / "summary.json").exists())
            self.assertTrue((root / "evaluated" / "metrics.jsonl").exists())

    def test_slice_quantized_piece_rebases_segment(self) -> None:
        piece = QuantizedPiece(
            piece_id="slice_me",
            resolution="sixteenth",
            steps_per_beat=4,
            bar_steps=16,
            time_signature="4/4",
            tempo_bpm=120.0,
            note_events=[
                QuantizedEvent(
                    pitch=60,
                    velocity=90,
                    instrument=0,
                    channel=0,
                    start_step=0,
                    duration_steps=4,
                    bar=1,
                    position=0,
                    track_index=0,
                ),
                QuantizedEvent(
                    pitch=62,
                    velocity=90,
                    instrument=0,
                    channel=0,
                    start_step=16,
                    duration_steps=4,
                    bar=2,
                    position=0,
                    track_index=0,
                ),
                QuantizedEvent(
                    pitch=64,
                    velocity=90,
                    instrument=0,
                    channel=0,
                    start_step=32,
                    duration_steps=4,
                    bar=3,
                    position=0,
                    track_index=0,
                ),
            ],
            phrase_boundaries=[1, 3],
        )

        sliced = slice_quantized_piece(piece, start_event=1, event_count=2)

        self.assertEqual(len(sliced.note_events), 2)
        self.assertEqual(sliced.note_events[0].start_step, 0)
        self.assertEqual(sliced.note_events[0].bar, 1)
        self.assertEqual(sliced.phrase_boundaries[0], 1)
        self.assertEqual(sliced.metadata["slice_start_event"], 1)

    def test_build_matched_reference_set_uses_manifest_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            processed_dir = root / "processed"
            piece = QuantizedPiece(
                piece_id="reference_piece",
                resolution="sixteenth",
                steps_per_beat=4,
                bar_steps=16,
                time_signature="4/4",
                tempo_bpm=120.0,
                note_events=[
                    QuantizedEvent(
                        pitch=60 + index,
                        velocity=90,
                        instrument=0,
                        channel=0,
                        start_step=index * 4,
                        duration_steps=2,
                        bar=(index // 4) + 1,
                        position=(index * 4) % 16,
                        track_index=0,
                    )
                    for index in range(6)
                ],
                phrase_boundaries=[1, 2],
            )
            write_quantized_piece_json(piece, processed_dir / "reference_piece.json")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "piece_id": "reference_piece",
                                "generated_event_count": 4,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = build_matched_reference_set(
                manifest_path,
                processed_dir=processed_dir,
                output_dir=root / "reference_out",
            )

            self.assertEqual(result["piece_count"], 1)
            evaluation = evaluate_directory(root / "reference_out", output_dir=root / "reference_eval")
            self.assertEqual(evaluation["summary"]["piece_count"], 1)

    def test_run_recurrence_diagnostics_reports_phrase_counts_for_piece_dir(self) -> None:
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
            config = _toy_baseline_config(root)
            config_path = root / "toy_baseline.yaml"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            run_baseline_training(
                config,
                config_path=config_path,
                dry_run=False,
                max_steps_override=1,
            )
            generate_from_checkpoint(
                root / "baseline_outputs" / "latest.pt",
                processed_dir=root / "processed",
                splits_dir=root / "splits",
                split="val",
                limit_pieces=1,
                prompt_events=4,
                generate_events=6,
                device="cpu",
                output_dir=root / "generated",
                seed=5,
            )

            diagnostics = run_recurrence_diagnostics(
                config_path=config_path,
                processed_dir=root / "processed",
                splits_dir=root / "splits",
                splits=["train"],
                input_dirs=[str(root / "generated")],
                limit_pieces=1,
                device="cpu",
            )

            self.assertIn("train", diagnostics["split_summaries"])
            self.assertIn(str(root / "generated"), diagnostics["piece_set_summaries"])


if __name__ == "__main__":
    unittest.main()
