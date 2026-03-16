"""Benchmark-model registry for poster-cited baseline systems."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BenchmarkModelSpec:
    """Named benchmark baseline mapped to an implemented repo entrypoint."""

    slug: str
    public_name: str
    repo_stage: str
    train_module: str
    default_config: str
    description: str


BENCHMARK_MODEL_SPECS: dict[str, BenchmarkModelSpec] = {
    "music_transformer": BenchmarkModelSpec(
        slug="music_transformer",
        public_name="MusicTransformer",
        repo_stage="music_transformer",
        train_module="training.train_music_transformer",
        default_config="configs/music_transformer_asap_score.yaml",
        description="Relative-attention autoregressive transformer baseline.",
    ),
    "magenta_music_transformer": BenchmarkModelSpec(
        slug="magenta_music_transformer",
        public_name="Magenta Music Transformer",
        repo_stage="magenta_music_transformer",
        train_module="training.train_magenta_music_transformer",
        default_config="configs/magenta_music_transformer_asap_score.yaml",
        description="Magenta-branded Music Transformer baseline with the same relative-attention core.",
    ),
    "figaro": BenchmarkModelSpec(
        slug="figaro",
        public_name="FIGARO",
        repo_stage="figaro_style",
        train_module="training.train_figaro",
        default_config="configs/figaro_asap_score.yaml",
        description="Control-conditioned autoregressive baseline inspired by FIGARO.",
    ),
    "diffusion_unet": BenchmarkModelSpec(
        slug="diffusion_unet",
        public_name="Diffusion U-Net",
        repo_stage="diffusion_unet",
        train_module="training.train_diffusion_unet",
        default_config="configs/diffusion_unet_asap_score.yaml",
        description="1D U-Net denoising baseline over grouped symbolic tokens.",
    ),
    "vae_decoder": BenchmarkModelSpec(
        slug="vae_decoder",
        public_name="VAE decoder",
        repo_stage="vae_decoder",
        train_module="training.train_vae",
        default_config="configs/vae_decoder_asap_score.yaml",
        description="Sequence-level variational autoencoder baseline.",
    ),
}


def _canonicalize_model_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def get_benchmark_model_spec(name: str) -> BenchmarkModelSpec:
    """Resolve a poster-cited benchmark model name to a repo-native implementation."""
    key = _canonicalize_model_name(name)
    aliases = {
        "musictransformer": "music_transformer",
        "magenta": "magenta_music_transformer",
        "magenta_music_transformer": "magenta_music_transformer",
        "google_magenta": "magenta_music_transformer",
        "figaro_style": "figaro",
        "figaro": "figaro",
        "diffusion_unet": "diffusion_unet",
        "u_net": "diffusion_unet",
        "vae": "vae_decoder",
        "vae_decoder": "vae_decoder",
    }
    key = aliases.get(key, key)
    if key not in BENCHMARK_MODEL_SPECS:
        available = ", ".join(sorted(BENCHMARK_MODEL_SPECS))
        raise KeyError(f"Unknown benchmark model '{name}'. Available models: {available}.")
    return BENCHMARK_MODEL_SPECS[key]


def list_benchmark_model_specs() -> tuple[BenchmarkModelSpec, ...]:
    """Return benchmark models in stable presentation order."""
    order = (
        "magenta_music_transformer",
        "music_transformer",
        "figaro",
        "diffusion_unet",
        "vae_decoder",
    )
    return tuple(BENCHMARK_MODEL_SPECS[key] for key in order)


def validate_benchmark_model_config(config: dict, *, expected_model: str) -> BenchmarkModelSpec:
    """Ensure benchmark config metadata matches the intended named baseline."""
    spec = get_benchmark_model_spec(expected_model)
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
