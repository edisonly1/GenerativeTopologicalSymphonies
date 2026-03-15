"""Tests for poster-alignment reporting."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.poster_report import build_poster_alignment_report


class PosterReportTests(unittest.TestCase):
    def test_build_poster_alignment_report_derives_claim_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            phase6_report = {
                "reference": {
                    "summary": {
                        "mean_recurrence_similarity": 0.75,
                        "mean_pitch_jump": 10.0,
                        "mean_pitch_class_divergence": None,
                        "mean_large_span_rate": 0.12,
                    }
                },
                "stages": {
                    "baseline": {
                        "evaluation": {
                            "summary": {
                                "mean_recurrence_similarity": 0.6,
                                "mean_pitch_jump": 20.0,
                                "mean_pitch_class_divergence": 0.2,
                                "mean_large_span_rate": 0.3,
                                "mean_max_persistence": 0.1,
                            }
                        },
                    },
                    "tension": {
                        "evaluation": {
                            "summary": {
                                "mean_recurrence_similarity": 0.7,
                                "mean_pitch_jump": 10.0,
                                "mean_pitch_class_divergence": 0.05,
                                "mean_large_span_rate": 0.18,
                                "mean_max_persistence": 0.8,
                            }
                        },
                    },
                },
            }
            torus_geometry = {
                "mean_structural_stress": 0.2,
                "mean_collapse_score": 0.1,
            }
            euclidean_geometry = {
                "mean_structural_stress": 1.0,
                "mean_collapse_score": 0.4,
            }
            phase6_path = root / "phase6.json"
            torus_path = root / "torus.json"
            euclidean_path = root / "euclidean.json"
            phase6_path.write_text(json.dumps(phase6_report), encoding="utf-8")
            torus_path.write_text(json.dumps(torus_geometry), encoding="utf-8")
            euclidean_path.write_text(json.dumps(euclidean_geometry), encoding="utf-8")

            result = build_poster_alignment_report(
                phase6_path,
                output_dir=root / "poster",
                music_stage="tension",
                baseline_stage="baseline",
                geometry_summary_path=torus_path,
                euclidean_summary_path=euclidean_path,
            )

            claims = {item["claim"]: item for item in result["poster_claims"]}
            self.assertEqual(
                claims["Erratic melodic jumps reduced by 68%"]["status"],
                "partial",
            )
            self.assertEqual(
                claims["Tonal distribution 85% closer to human-composed music"]["status"],
                "partial",
            )
            self.assertEqual(
                claims["Structural stress reduced by 73% versus Euclidean mapping"]["status"],
                "supported",
            )
            self.assertEqual(
                claims["Persistent patterns increased 7x over the baseline"]["status"],
                "supported",
            )
            self.assertEqual(
                claims["Unconstrained generation collapses to a low-dimensional signature"]["status"],
                "supported",
            )
            self.assertTrue((root / "poster" / "report.json").exists())
            self.assertTrue((root / "poster" / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
