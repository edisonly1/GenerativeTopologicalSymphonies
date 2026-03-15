from __future__ import annotations

import unittest

from evaluation.sweep_decoding import _compute_objective


class DecodingSweepTests(unittest.TestCase):
    def test_objective_penalizes_tonal_and_duration_collapse(self) -> None:
        baseline = {
            "mean_recurrence_similarity": 0.57,
            "mean_recurrent_phrase_ratio": 0.06,
            "mean_pitch_jump": 12.2,
            "mean_large_span_rate": 0.23,
            "mean_overlap_violation_rate": 0.09,
            "mean_pitch_class_entropy": 3.29,
            "mean_final_duration": 3.84,
            "mean_max_persistence": 0.056,
            "mean_cadence_rate": 0.32,
            "mean_tonal_center_strength": 0.205,
            "mean_pitch_class_divergence": 0.064,
        }
        reference = {
            "mean_recurrence_similarity": 0.72,
            "mean_recurrent_phrase_ratio": 0.19,
            "mean_pitch_jump": 11.1,
            "mean_large_span_rate": 0.15,
            "mean_overlap_violation_rate": 0.004,
            "mean_pitch_class_entropy": 3.20,
            "mean_final_duration": 2.84,
            "mean_max_persistence": 0.09,
            "mean_cadence_rate": 0.47,
            "mean_tonal_center_strength": 0.187,
        }
        balanced_candidate = {
            "mean_recurrence_similarity": 0.66,
            "mean_recurrent_phrase_ratio": 0.20,
            "mean_pitch_jump": 11.6,
            "mean_large_span_rate": 0.18,
            "mean_overlap_violation_rate": 0.05,
            "mean_pitch_class_entropy": 3.14,
            "mean_final_duration": 2.70,
            "mean_max_persistence": 0.08,
            "mean_cadence_rate": 0.45,
            "mean_tonal_center_strength": 0.20,
            "mean_pitch_class_divergence": 0.05,
        }
        collapsed_candidate = {
            "mean_recurrence_similarity": 0.83,
            "mean_recurrent_phrase_ratio": 0.62,
            "mean_pitch_jump": 8.99,
            "mean_large_span_rate": 0.11,
            "mean_overlap_violation_rate": 0.29,
            "mean_pitch_class_entropy": 2.63,
            "mean_final_duration": 1.42,
            "mean_max_persistence": 0.098,
            "mean_cadence_rate": 0.49,
            "mean_tonal_center_strength": 0.39,
            "mean_pitch_class_divergence": 0.105,
        }

        balanced_score, _ = _compute_objective(
            balanced_candidate,
            baseline_summary=baseline,
            reference_summary=reference,
        )
        collapsed_score, _ = _compute_objective(
            collapsed_candidate,
            baseline_summary=baseline,
            reference_summary=reference,
        )

        self.assertGreater(balanced_score, collapsed_score)


if __name__ == "__main__":
    unittest.main()
