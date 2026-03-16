# Benchmark Scope

This repository exposes the implemented systems directly and distinguishes them from poster-cited external baselines that are not shipped here.

## Implemented systems

| Public name | Repo stage | Entry point | Default config |
| --- | --- | --- | --- |
| Baseline | `baseline` | `training.train_baseline` | `configs/baseline_asap_score.yaml` |
| Phrase planner | `conductor` | `training.train_conductor` | `configs/conductor_asap_score.yaml` |
| Ingram-1 | `torus_t3` | `training.train_ingram_1` | `configs/ingram_1_asap_score.yaml` |
| Ingram-2 | `tension_t3` | `training.train_ingram_2` | `configs/ingram_2_asap_score.yaml` |

## External baselines referenced in poster materials

These are **not** implemented as first-party baselines in this repository:

| Name | Repo status | How to read the reference |
| --- | --- | --- |
| Google Magenta | Not implemented | External comparison target only |
| FIGARO | Not implemented | External comparison target only |
| MusicTransformer | Not implemented | External comparison target only |
| Diffusion U-Net | Not implemented | External comparison target only |
| VAE decoder | Not implemented | External comparison target only |

If a slide, poster, or summary mentions these names, that should not be read as a claim that this repository contains their training pipelines or reproducible checkpoints.
