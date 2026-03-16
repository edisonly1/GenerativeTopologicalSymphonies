"""Public-facing model-family registry for poster and repo alignment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PublicModelSpec:
    """Public model identity mapped to an implemented repo stage."""

    slug: str
    public_name: str
    repo_stage: str
    train_module: str
    default_config: str
    description: str


PUBLIC_MODEL_SPECS: dict[str, PublicModelSpec] = {
    "baseline": PublicModelSpec(
        slug="baseline",
        public_name="Baseline",
        repo_stage="baseline",
        train_module="training.train_baseline",
        default_config="configs/baseline_asap_score.yaml",
        description="Plain grouped-token transformer decoder.",
    ),
    "phrase_planner": PublicModelSpec(
        slug="phrase_planner",
        public_name="Phrase planner",
        repo_stage="conductor",
        train_module="training.train_conductor",
        default_config="configs/conductor_asap_score.yaml",
        description="Baseline decoder plus phrase-level conductor targets.",
    ),
    "ingram_1": PublicModelSpec(
        slug="ingram_1",
        public_name="Ingram-1",
        repo_stage="torus_t3",
        train_module="training.train_ingram_1",
        default_config="configs/ingram_1_asap_score.yaml",
        description="Phrase planner plus explicit T^3 latent bottleneck.",
    ),
    "ingram_2": PublicModelSpec(
        slug="ingram_2",
        public_name="Ingram-2",
        repo_stage="tension_t3",
        train_module="training.train_ingram_2",
        default_config="configs/ingram_2_asap_score.yaml",
        description="Ingram-1 plus harmonic tension regularization.",
    ),
}

def _canonicalize_model_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def get_public_model_spec(name: str) -> PublicModelSpec:
    """Resolve a public-facing model name into an implemented repo-stage spec."""
    key = _canonicalize_model_name(name)
    aliases = {
        "phraseplanner": "phrase_planner",
        "phrase_planner": "phrase_planner",
        "ingram1": "ingram_1",
        "ingram_1": "ingram_1",
        "ingram2": "ingram_2",
        "ingram_2": "ingram_2",
    }
    key = aliases.get(key, key)
    if key not in PUBLIC_MODEL_SPECS:
        available = ", ".join(sorted(PUBLIC_MODEL_SPECS))
        raise KeyError(f"Unknown public model '{name}'. Available models: {available}.")
    return PUBLIC_MODEL_SPECS[key]


def list_public_model_specs() -> tuple[PublicModelSpec, ...]:
    """Return the public model family in stable presentation order."""
    order = ("baseline", "phrase_planner", "ingram_1", "ingram_2")
    return tuple(PUBLIC_MODEL_SPECS[key] for key in order)


def validate_public_model_config(config: dict, *, expected_model: str) -> PublicModelSpec:
    """Ensure a public-facing config carries the expected metadata mapping."""
    spec = get_public_model_spec(expected_model)
    metadata = config.get("metadata", {})
    public_name = metadata.get("public_name")
    repo_stage = metadata.get("repo_stage")
    if public_name != spec.public_name:
        raise ValueError(
            f"Config public_name mismatch: expected '{spec.public_name}', got '{public_name}'."
        )
    if repo_stage != spec.repo_stage:
        raise ValueError(
            f"Config repo_stage mismatch: expected '{spec.repo_stage}', got '{repo_stage}'."
        )
    return spec
