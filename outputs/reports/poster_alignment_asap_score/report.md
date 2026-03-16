# Poster Alignment Report

## Criteria

| Criterion | Metric | Reference | Baseline | tension_t3 | euclidean | torus_t3 | Better |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Structural Coherence | mean_recurrence_similarity | 0.8488 | 0.6355 | 0.9221 | - | - | higher |
| Melodic Quality | mean_pitch_jump | 9.3413 | 20.9298 | 1.4110 | - | - | lower |
| Tonal Fidelity | mean_pitch_class_divergence | - | 0.4607 | 0.0689 | - | - | lower |
| Human Plausibility | mean_large_span_rate | 0.2236 | 0.0000 | 0.0000 | - | - | lower |
| Geometry Preservation | mean_structural_stress | - | - | - | 3.3517 | 0.8589 | lower |
| Latent Collapse | mean_collapse_score | - | - | - | 0.1218 | 0.0789 | lower |

## Claims

| Claim | Current | Target | Status |
| --- | --- | --- | --- |
| Erratic melodic jumps reduced by 68% | 93.26% | 68.0% | supported |
| Tonal distribution 85% closer to human-composed music | 85.04% | 85.0% | supported |
| Structural stress reduced by 73% versus Euclidean mapping | 74.37% | 73.0% | supported |
| Persistent patterns increased 7x over the baseline | 9.91x | 7.0x | supported |
| Unconstrained generation collapses to a low-dimensional signature | 35.19% | - | supported |


