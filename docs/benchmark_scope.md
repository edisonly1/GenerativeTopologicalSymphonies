# Benchmark Scope

This repository exposes both the core project stages and the benchmark baselines used in poster-facing comparisons.

## Implemented systems

| Public name | Repo stage | Entry point | Default config |
| --- | --- | --- | --- |
| Baseline | `baseline` | `training.train_baseline` | `configs/baseline_asap_score.yaml` |
| Phrase planner | `conductor` | `training.train_conductor` | `configs/conductor_asap_score.yaml` |
| Ingram-1 | `torus_t3` | `training.train_ingram_1` | `configs/ingram_1_asap_score.yaml` |
| Ingram-2 | `tension_t3` | `training.train_ingram_2` | `configs/ingram_2_asap_score.yaml` |

## Implemented benchmark baselines

These poster-cited baselines now have repo-native implementations and entry points:

| Name | Repo status | How to read the reference |
| --- | --- | --- |
| Google Magenta / Magenta Music Transformer | Implemented | Relative-attention autoregressive baseline via `training.train_magenta_music_transformer` |
| MusicTransformer | Implemented | Relative-attention autoregressive baseline via `training.train_music_transformer` |
| FIGARO | Implemented | Control-conditioned baseline via `training.train_figaro` |
| Diffusion U-Net | Implemented | 1D symbolic denoiser via `training.train_diffusion_unet` |
| VAE decoder | Implemented | Sequence-level variational baseline via `training.train_vae` |

These are repo-native implementations aligned to the poster benchmark names. They should be read as in-repo baselines for comparison within this project.
