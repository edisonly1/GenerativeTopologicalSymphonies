# Generative Topological Symphonies

Research codebase for a symbolic music-generation system built around grouped event tokenization, phrase-level planning, torus-constrained latent structure, harmonic tension regularization, and denoising refinement.




## Public naming

The repo now exposes the poster-facing model family directly:

- Baseline -> `baseline` via `training.train_baseline`
- Phrase planner -> `conductor` via `training.train_conductor`
- Ingram-1 -> `torus_t3` via `training.train_ingram_1`
- Ingram-2 -> `tension_t3` via `training.train_ingram_2`

The repo also exposes poster-cited benchmark baselines directly:

- Google Magenta / Magenta Music Transformer -> `training.train_magenta_music_transformer`
- MusicTransformer -> `training.train_music_transformer`
- FIGARO -> `training.train_figaro`
- Diffusion U-Net -> `training.train_diffusion_unet`
- VAE decoder -> `training.train_vae`


## Repository layout

```text
configs/     Experiment configurations
data/        Raw, processed, split, and evaluation datasets
docs/        Public-facing scope notes and poster/repo mappings
src/         Project source modules
outputs/     Generated artifacts, figures, tables, and renders
```


## ASAP score-first pipeline

The repo now supports the [ASAP aligned-score dataset](https://github.com/fosfrancesco/asap-dataset)
as a score-first alternative to MAESTRO. ASAP is useful here because it provides score MIDI plus
official beat, downbeat, time-signature, and key-signature annotations.


