# Poster Alignment Guide

This repository exposes the implemented model family and geometry ablations explicitly so the codebase can be read without relying on poster-only terminology.

## Public name to repo-stage mapping

| Public / poster-facing name | Repo stage / config | Meaning |
| --- | --- | --- |
| Baseline | `baseline`, `configs/baseline_asap_score.yaml` | Plain grouped-token transformer decoder |
| Phrase planner | `conductor`, `configs/conductor_asap_score.yaml` | Baseline plus phrase-level conductor targets |
| Ingram-1 | `torus_t3`, `configs/ingram_1_asap_score.yaml` | Conductor-conditioned decoder with explicit `T^3` latent bottleneck |
| Ingram-2 | `tension_t3`, `configs/ingram_2_asap_score.yaml` | Ingram-1 plus harmonic tension regularization |

## External baselines

The following names are **not implemented in this repository**:

- Google Magenta
- FIGARO
- MusicTransformer
- diffusion / U-Net / VAE decoder baselines

If they are mentioned in external slides or posters, treat them as external comparison targets rather than shipped code paths.

## Geometry-family ablations

Implemented latent geometry families:

- `plane_r2`
- `sphere_s2`
- `hypercube_r3`
- `euclidean_r3`
- `torus_t3`

Relevant configs:

- `configs/plane_r2_asap_score.yaml`
- `configs/sphere_s2_asap_score.yaml`
- `configs/hypercube_r3_asap_score.yaml`
- `configs/euclidean_asap_score.yaml`
- `configs/torus_t3_asap_score.yaml`
- `configs/poster_geometry_ablation_asap_score.yaml`

The ablation suite will skip missing checkpoints when `skip_if_missing: true` is set, so the geometry search package can live in the repo even before all runs are materialized.

## Math scope

What the code implements:

- torus / sphere / Euclidean-family latent projections
- geometry-matching regularization on phrase-state distances
- manifold smoothness penalties
- dispersion penalties to avoid latent collapse
- tension-descent-inspired regularization over phrase trajectories

What the code does **not** currently claim as a formal proof artifact:

- Hessian-derived stability certificates
- Lyapunov convergence proofs
- a symbolic derivation tying every poster equation to a tested loss term

Use the mathematical language as intuition for the design, not as a statement that the repo contains a full theorem-backed proof package.
