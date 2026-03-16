"""Tests for public-facing Ingram model aliases and benchmark scope."""

from __future__ import annotations

import unittest

from models.ingram import (
    get_public_model_spec,
    list_public_model_specs,
    validate_public_model_config,
)
from training.train_baseline import load_config


class IngramAliasTests(unittest.TestCase):
    def test_public_model_registry_exposes_first_class_ingram_family(self) -> None:
        specs = list_public_model_specs()
        self.assertEqual(
            [spec.slug for spec in specs],
            ["baseline", "phrase_planner", "ingram_1", "ingram_2"],
        )
        self.assertEqual(get_public_model_spec("Ingram-1").repo_stage, "torus_t3")
        self.assertEqual(get_public_model_spec("ingram_2").repo_stage, "tension_t3")

    def test_ingram_configs_validate_against_public_mapping(self) -> None:
        ingram_1 = load_config("configs/ingram_1_asap_score.yaml")
        ingram_2 = load_config("configs/ingram_2_asap_score.yaml")
        self.assertEqual(
            validate_public_model_config(ingram_1, expected_model="ingram_1").public_name,
            "Ingram-1",
        )
        self.assertEqual(
            validate_public_model_config(ingram_2, expected_model="ingram_2").public_name,
            "Ingram-2",
        )


if __name__ == "__main__":
    unittest.main()
