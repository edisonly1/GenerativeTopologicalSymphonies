"""Tests for benchmark baseline registries and training smoke paths."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from models.benchmarks import (
    get_benchmark_model_spec,
    list_benchmark_model_specs,
    validate_benchmark_model_config,
)
from models.diffusion_unet import DiffusionUNetConfig, DiffusionUNetDenoiser
from models.music_transformer import MusicTransformerConfig, MusicTransformerGroupedDecoder
from preprocessing import generate_toy_dataset, prepare_dataset
from training.train_baseline import build_feature_vocab_sizes, load_config, run_baseline_training
from training.train_vae import build_vae_model, run_vae_training
from training.train_diffusion_unet import run_diffusion_unet_training


def _toy_data_config(root: Path) -> dict:
    return {
        "raw_dir": str(root / "raw"),
        "processed_dir": str(root / "processed"),
        "splits_dir": str(root / "splits"),
        "eval_dir": str(root / "eval"),
        "quantization": "sixteenth",
        "phrase_strategy": "bars_4",
    }


def _toy_tokenization_config() -> dict:
    return {
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
    }


class BenchmarkModelTests(unittest.TestCase):
    def test_benchmark_registry_and_configs_validate(self) -> None:
        specs = list_benchmark_model_specs()
        self.assertEqual(
            [spec.slug for spec in specs],
            [
                "magenta_music_transformer",
                "music_transformer",
                "figaro",
                "diffusion_unet",
                "vae_decoder",
            ],
        )
        self.assertEqual(get_benchmark_model_spec("Magenta").repo_stage, "magenta_music_transformer")
        self.assertEqual(get_benchmark_model_spec("FIGARO").repo_stage, "figaro_style")

        paths = {
            "music_transformer": "configs/music_transformer_asap_score.yaml",
            "magenta_music_transformer": "configs/magenta_music_transformer_asap_score.yaml",
            "figaro": "configs/figaro_asap_score.yaml",
            "diffusion_unet": "configs/diffusion_unet_asap_score.yaml",
            "vae_decoder": "configs/vae_decoder_asap_score.yaml",
        }
        for slug, path in paths.items():
            config = load_config(path)
            self.assertEqual(validate_benchmark_model_config(config, expected_model=slug).slug, slug)

    def test_music_transformer_forward_preserves_feature_shapes(self) -> None:
        vocab_sizes = build_feature_vocab_sizes(_toy_tokenization_config())
        model = MusicTransformerGroupedDecoder(
            vocab_sizes=vocab_sizes,
            config=MusicTransformerConfig(d_model=32, num_layers=2, num_heads=4, dim_feedforward=64),
        )
        batch_size = 2
        seq_len = 5
        inputs = {
            feature: torch.ones((batch_size, seq_len), dtype=torch.long)
            for feature in vocab_sizes
        }
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        logits = model(inputs, attention_mask)
        self.assertEqual(logits["pitch"].shape[:2], (batch_size, seq_len))

    def test_vae_model_forward_returns_latent_statistics(self) -> None:
        vocab_sizes = build_feature_vocab_sizes(_toy_tokenization_config())
        config = {
            "model": {
                "d_model": 32,
                "num_layers": 2,
                "encoder_layers": 2,
                "num_heads": 4,
                "dim_feedforward": 64,
                "dropout": 0.1,
                "latent_dim": 16,
            }
        }
        model = build_vae_model(config, vocab_sizes=vocab_sizes)
        inputs = {
            feature: torch.ones((2, 6), dtype=torch.long)
            for feature in vocab_sizes
        }
        attention_mask = torch.ones((2, 6), dtype=torch.bool)
        output = model(inputs, attention_mask)
        self.assertEqual(output.token_logits["pitch"].shape[:2], (2, 6))
        self.assertEqual(output.latent_mean.shape, (2, 16))
        self.assertEqual(output.latent_logvar.shape, (2, 16))

    def test_diffusion_unet_forward_preserves_feature_shapes(self) -> None:
        vocab_sizes = build_feature_vocab_sizes(_toy_tokenization_config())
        model = DiffusionUNetDenoiser(
            vocab_sizes=vocab_sizes,
            config=DiffusionUNetConfig(d_model=32, base_channels=32, dropout=0.1),
        )
        inputs = {
            feature: torch.ones((2, 7), dtype=torch.long)
            for feature in vocab_sizes
        }
        attention_mask = torch.ones((2, 7), dtype=torch.bool)
        logits = model(inputs, attention_mask)
        self.assertEqual(logits["pitch"].shape[:2], (2, 7))

    def test_music_transformer_vae_and_diffusion_dry_runs_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            generate_toy_dataset(root / "raw", num_pieces=5, seed=19)
            prepare_dataset(
                raw_dir=root / "raw",
                processed_dir=root / "processed",
                splits_dir=root / "splits",
                train_ratio=0.6,
                val_ratio=0.2,
                seed=3,
            )

            common_training = {
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
            }

            music_config = {
                "experiment_name": "toy_music_transformer",
                "seed": 5,
                "data": _toy_data_config(root),
                "tokenization": _toy_tokenization_config(),
                "model": {
                    "architecture": "music_transformer",
                    "d_model": 32,
                    "num_layers": 2,
                    "num_heads": 4,
                    "dim_feedforward": 64,
                    "dropout": 0.1,
                    "relative_attention_buckets": 16,
                    "max_relative_distance": 32,
                },
                "training": {**common_training, "output_dir": str(root / "music_outputs")},
                "generation": {"temperature": 1.0, "top_k": 0, "top_p": 0.95},
                "evaluation": {"sample_count": 4, "metrics": ["perplexity"]},
            }
            music_result = run_baseline_training(
                music_config,
                config_path=root / "music.yaml",
                dry_run=True,
            )
            self.assertEqual(music_result["mode"], "dry_run")

            vae_config = {
                "experiment_name": "toy_vae",
                "seed": 7,
                "data": _toy_data_config(root),
                "tokenization": _toy_tokenization_config(),
                "model": {
                    "architecture": "vae_decoder",
                    "d_model": 32,
                    "num_layers": 2,
                    "encoder_layers": 2,
                    "num_heads": 4,
                    "dim_feedforward": 64,
                    "dropout": 0.1,
                    "latent_dim": 16,
                    "vae_dropout": 0.1,
                },
                "losses": {"kl_weight": 0.05},
                "training": {**common_training, "output_dir": str(root / "vae_outputs")},
                "generation": {"temperature": 1.0, "top_k": 0, "top_p": 0.95},
                "evaluation": {"sample_count": 4, "metrics": ["perplexity"]},
            }
            vae_result = run_vae_training(
                vae_config,
                config_path=root / "vae.yaml",
                dry_run=True,
            )
            self.assertEqual(vae_result["mode"], "dry_run")
            self.assertGreater(vae_result["train_metrics"]["kl_loss"], 0.0)

            diffusion_config = {
                "experiment_name": "toy_diffusion_unet",
                "seed": 11,
                "data": _toy_data_config(root),
                "tokenization": _toy_tokenization_config(),
                "model": {
                    "architecture": "diffusion_unet",
                    "d_model": 32,
                    "base_channels": 32,
                    "dropout": 0.1,
                },
                "training": {
                    **common_training,
                    "output_dir": str(root / "diffusion_outputs"),
                    "corruption": {
                        "token_mask_prob": 0.20,
                        "pitch_shift_prob": 0.10,
                        "duration_shift_prob": 0.10,
                        "phrase_flag_flip_prob": 0.10,
                        "bar_position_jitter_prob": 0.10,
                    },
                },
                "generation": {"temperature": 1.0, "top_k": 0, "top_p": 0.95},
                "evaluation": {"sample_count": 4, "metrics": ["perplexity"]},
            }
            diffusion_result = run_diffusion_unet_training(
                diffusion_config,
                config_path=root / "diffusion.yaml",
                dry_run=True,
            )
            self.assertEqual(diffusion_result["mode"], "dry_run")


if __name__ == "__main__":
    unittest.main()
