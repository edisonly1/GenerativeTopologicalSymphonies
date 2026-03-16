# Generative Topological Symphonies

Research codebase for a symbolic music-generation system built around grouped event tokenization, phrase-level planning, torus-constrained latent structure, harmonic tension regularization, and denoising refinement.

## Current status

Phases 0 through 5 are implemented and tested on MAESTRO-sized runs: preprocessing, grouped-token
training, the conductor, torus bottleneck, tension regularization, and the denoising refiner. Phase
6 now focuses on ablations, evaluation, and report packaging.

## Planned build order

1. Baseline symbolic generator
2. Phrase-aware conductor
3. Torus latent bottleneck
4. Tension-surface regularization
5. Denoising refiner
6. Evaluation and packaging

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

## Poster scope

The geometry and tension code is real, but the repository should be read as a topology- and tension-inspired training system, not a full theorem/proof package. A concise mapping of poster terminology, implemented scope, and geometry-family ablations is in [docs/poster_alignment.md](docs/poster_alignment.md). A direct benchmark status table is in [docs/benchmark_scope.md](docs/benchmark_scope.md).

## Repository layout

```text
configs/     Experiment configurations
data/        Raw, processed, split, and evaluation datasets
docs/        Public-facing scope notes and poster/repo mappings
notebooks/   Exploratory and validation notebooks
src/         Project source modules
outputs/     Generated artifacts, figures, tables, and renders
paper/       Research paper sources
```

## Near-term priorities

- Run the Phase 6 ablation suite in `src/evaluation/run_ablation_suite.py`
- Inspect generated artifacts under `outputs/reports/`
- Tighten evaluation metrics and human-study packaging for the paper

## ASAP score-first pipeline

The repo now supports the [ASAP aligned-score dataset](https://github.com/fosfrancesco/asap-dataset)
as a score-first alternative to MAESTRO. ASAP is useful here because it provides score MIDI plus
official beat, downbeat, time-signature, and key-signature annotations.

The GitHub repo intentionally omits raw, processed, and split dataset artifacts. Before retraining,
either prepare ASAP locally into `data/raw/asap`, `data/processed/asap_score`, and
`data/splits/asap_score`, or edit the configs to point at your local dataset paths.

Prepare ASAP score MIDI with dataset annotations:

```bash
cd /path/to/GenerativeTopologicalSymphonies

PYTHONPATH=src python3 -m preprocessing.prepare_dataset \
  --raw-dir data/raw/asap \
  --processed-dir data/processed/asap_score \
  --splits-dir data/splits/asap_score \
  --dataset-kind asap \
  --dataset-source-mode score \
  --use-dataset-annotations \
  --annotate-harmony \
  --phrase-strategy cadence_bars_4
```

This mode:

- selects unique score MIDI files instead of every performance MIDI
- injects official ASAP time/key annotation hints into processed pieces
- groups split assignment by `(composer, title)` to reduce near-duplicate leakage

Suggested training order on ASAP score data:

```bash
PYTHONPATH=src python3 -u -m training.train_baseline --config configs/baseline_asap_score.yaml
PYTHONPATH=src python3 -u -m training.train_conductor --config configs/conductor_asap_score.yaml
PYTHONPATH=src python3 -u -m training.train_torus --config configs/torus_t3_asap_score.yaml
PYTHONPATH=src python3 -u -m training.train_torus --config configs/euclidean_asap_score.yaml
PYTHONPATH=src python3 -u -m training.train_tension --config configs/tension_t3_asap_score.yaml
PYTHONPATH=src python3 -u -m evaluation.run_ablation_suite --config configs/poster_alignment_asap_score.yaml --output-dir outputs/reports/phase6_asap_score
```

Poster-facing aliases:

```bash
PYTHONPATH=src python3 -u -m training.train_ingram_1 --config configs/ingram_1_asap_score.yaml
PYTHONPATH=src python3 -u -m training.train_ingram_2 --config configs/ingram_2_asap_score.yaml
```

Benchmark baselines:

```bash
PYTHONPATH=src python3 -u -m training.train_magenta_music_transformer --config configs/magenta_music_transformer_asap_score.yaml
PYTHONPATH=src python3 -u -m training.train_music_transformer --config configs/music_transformer_asap_score.yaml
PYTHONPATH=src python3 -u -m training.train_figaro --config configs/figaro_asap_score.yaml
PYTHONPATH=src python3 -u -m training.train_diffusion_unet --config configs/diffusion_unet_asap_score.yaml
PYTHONPATH=src python3 -u -m training.train_vae --config configs/vae_decoder_asap_score.yaml
```

Geometry-family ablation package:

```bash
PYTHONPATH=src python3 -u -m evaluation.run_ablation_suite \
  --config configs/poster_geometry_ablation_asap_score.yaml \
  --output-dir outputs/reports/geometry_ablation_asap_score
```
