"""Geometry-preservation diagnostics for latent musical structure."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from training.data import AutoregressiveTokenDataset, collate_autoregressive_batch
from training.train_baseline import build_feature_vocab_sizes, load_config, move_batch_to_device, resolve_device
from training.train_torus import build_torus_model

TORUS_GEOMETRIES = frozenset({"legacy_torus", "torus_t3"})
SPHERE_GEOMETRIES = frozenset({"sphere_s2"})


@dataclass(slots=True)
class GeometryMetrics:
    """Geometry-preservation summary for one piece."""

    piece_id: str
    geometry_kind: str
    phrase_count: int
    intrinsic_dim: int
    structural_stress: float
    trustworthiness: float
    continuity: float
    neighbor_overlap: float
    effective_rank: float
    collapse_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit-pieces", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _pairwise_euclidean(points: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distances."""
    return torch.cdist(points, points, p=2)


def _pairwise_torus(angles: torch.Tensor) -> torch.Tensor:
    """Compute wrap-aware torus distances from angle coordinates."""
    delta = angles[:, None, :] - angles[None, :, :]
    wrapped = torch.atan2(torch.sin(delta), torch.cos(delta))
    return torch.linalg.vector_norm(wrapped, dim=-1)


def _pairwise_sphere(coordinates: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Compute great-circle distances for a set of unit-sphere coordinates."""
    normalized = coordinates / torch.linalg.vector_norm(
        coordinates,
        dim=-1,
        keepdim=True,
    ).clamp(min=eps)
    cosine = (normalized[:, None, :] * normalized[None, :, :]).sum(dim=-1).clamp(
        min=-1.0 + eps,
        max=1.0 - eps,
    )
    return torch.arccos(cosine)


def _rank_matrix(distances: torch.Tensor) -> torch.Tensor:
    """Convert a distance matrix into 1-based neighbor ranks."""
    order = torch.argsort(distances, dim=1)
    ranks = torch.zeros_like(order)
    rank_values = torch.arange(order.shape[1], device=distances.device).expand_as(order)
    ranks.scatter_(1, order, rank_values)
    return ranks


def _trustworthiness(source_distances: torch.Tensor, latent_distances: torch.Tensor, k: int) -> float:
    """Compute trustworthiness from pairwise distances."""
    phrase_count = source_distances.shape[0]
    if phrase_count <= 2 or k <= 0:
        return 1.0
    source_ranks = _rank_matrix(source_distances)
    latent_neighbors = torch.argsort(latent_distances, dim=1)[:, 1 : k + 1]
    source_neighbors = torch.argsort(source_distances, dim=1)[:, 1 : k + 1]
    penalties = 0.0
    source_neighbor_sets = [set(row.tolist()) for row in source_neighbors]
    for index in range(phrase_count):
        for neighbor in latent_neighbors[index].tolist():
            if neighbor not in source_neighbor_sets[index]:
                penalties += float(source_ranks[index, neighbor].item() - k)
    normalizer = phrase_count * k * max(1, (2 * phrase_count) - (3 * k) - 1)
    return max(0.0, 1.0 - (2.0 * penalties / normalizer))


def _continuity(source_distances: torch.Tensor, latent_distances: torch.Tensor, k: int) -> float:
    """Compute continuity from pairwise distances."""
    phrase_count = source_distances.shape[0]
    if phrase_count <= 2 or k <= 0:
        return 1.0
    latent_ranks = _rank_matrix(latent_distances)
    latent_neighbors = torch.argsort(latent_distances, dim=1)[:, 1 : k + 1]
    source_neighbors = torch.argsort(source_distances, dim=1)[:, 1 : k + 1]
    penalties = 0.0
    latent_neighbor_sets = [set(row.tolist()) for row in latent_neighbors]
    for index in range(phrase_count):
        for neighbor in source_neighbors[index].tolist():
            if neighbor not in latent_neighbor_sets[index]:
                penalties += float(latent_ranks[index, neighbor].item() - k)
    normalizer = phrase_count * k * max(1, (2 * phrase_count) - (3 * k) - 1)
    return max(0.0, 1.0 - (2.0 * penalties / normalizer))


def _neighbor_overlap(source_distances: torch.Tensor, latent_distances: torch.Tensor, k: int) -> float:
    """Compute average top-k neighbor overlap."""
    phrase_count = source_distances.shape[0]
    if phrase_count <= 1 or k <= 0:
        return 1.0
    source_neighbors = torch.argsort(source_distances, dim=1)[:, 1 : k + 1]
    latent_neighbors = torch.argsort(latent_distances, dim=1)[:, 1 : k + 1]
    overlaps = []
    for index in range(phrase_count):
        source_set = set(source_neighbors[index].tolist())
        latent_set = set(latent_neighbors[index].tolist())
        overlaps.append(len(source_set & latent_set) / max(len(source_set | latent_set), 1))
    return sum(overlaps) / len(overlaps)


def _effective_rank(coordinates: torch.Tensor) -> float:
    """Estimate the effective rank of centered latent coordinates."""
    if coordinates.shape[0] <= 1:
        return 0.0
    centered = coordinates - coordinates.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    if not torch.isfinite(singular_values).all() or float(singular_values.sum()) <= 0.0:
        return 0.0
    probabilities = singular_values / singular_values.sum()
    entropy = -torch.sum(probabilities * torch.log(probabilities.clamp(min=1e-9)))
    return float(torch.exp(entropy).item())


def _truncate_sample(sample: dict[str, Any], max_length: int) -> dict[str, Any]:
    """Truncate a full-piece sample to the model context window."""
    if sample["sequence_length"] <= max_length:
        return sample
    start = 0
    end = max_length
    window_phrase_ids = sample["phrase_ids"][start:end]
    local_phrase_order: list[int] = []
    local_phrase_map: dict[int, int] = {}
    for phrase_id in window_phrase_ids.tolist():
        if phrase_id not in local_phrase_map:
            local_phrase_map[phrase_id] = len(local_phrase_order)
            local_phrase_order.append(phrase_id)
    remapped_phrase_ids = torch.tensor(
        [local_phrase_map[phrase_id] for phrase_id in window_phrase_ids.tolist()],
        dtype=torch.long,
    )
    phrase_ranges: list[tuple[int, int]] = []
    phrase_complete: list[bool] = []
    conductor_targets: dict[str, torch.Tensor] = {}
    for global_phrase_id in local_phrase_order:
        full_start, full_end = sample["phrase_ranges"][global_phrase_id]
        clipped_start = max(start, full_start)
        clipped_end = min(end, full_end)
        phrase_ranges.append((clipped_start - start, clipped_end - start))
        phrase_complete.append(full_start >= start and full_end <= end)
    for target_name, target_values in sample["conductor_targets"].items():
        values = [
            int(target_values[global_phrase_id].item())
            for global_phrase_id in local_phrase_order
        ]
        conductor_targets[target_name] = torch.tensor(values, dtype=torch.long)
    return {
        "piece_id": sample["piece_id"],
        "inputs": {
            feature: values[start:end]
            for feature, values in sample["inputs"].items()
        },
        "targets": {
            feature: values[start:end]
            for feature, values in sample["targets"].items()
        },
        "sequence_length": end - start,
        "phrase_boundaries": list(sample["phrase_boundaries"]),
        "metadata": dict(sample["metadata"]),
        "window_range": (start, end),
        "phrase_ids": remapped_phrase_ids,
        "phrase_ranges": phrase_ranges,
        "phrase_complete": phrase_complete,
        "conductor_targets": conductor_targets,
    }


def score_geometry(
    *,
    piece_id: str,
    source_states: torch.Tensor,
    latent_coordinates: torch.Tensor,
    geometry_kind: str,
    neighbor_k: int = 3,
) -> GeometryMetrics:
    """Score how well a latent geometry preserves phrase relationships."""
    phrase_count = int(source_states.shape[0])
    intrinsic_dim = int(latent_coordinates.shape[1]) if latent_coordinates.ndim == 2 else 0
    if phrase_count <= 1 or intrinsic_dim == 0:
        return GeometryMetrics(
            piece_id=piece_id,
            geometry_kind=geometry_kind,
            phrase_count=phrase_count,
            intrinsic_dim=intrinsic_dim,
            structural_stress=0.0,
            trustworthiness=1.0,
            continuity=1.0,
            neighbor_overlap=1.0,
            effective_rank=0.0,
            collapse_score=1.0,
        )

    source_distances = _pairwise_euclidean(source_states)
    if geometry_kind in TORUS_GEOMETRIES:
        latent_distances = _pairwise_torus(latent_coordinates)
    elif geometry_kind in SPHERE_GEOMETRIES:
        latent_distances = _pairwise_sphere(latent_coordinates)
    else:
        latent_distances = _pairwise_euclidean(latent_coordinates)

    upper_triangle = torch.triu_indices(phrase_count, phrase_count, offset=1)
    source_vector = source_distances[upper_triangle[0], upper_triangle[1]]
    latent_vector = latent_distances[upper_triangle[0], upper_triangle[1]]
    denominator = float(source_vector.square().sum().item())
    if denominator <= 0.0:
        structural_stress = 0.0
    else:
        stress = float((source_vector - latent_vector).square().sum().item())
        structural_stress = math.sqrt(stress / denominator)

    k = min(neighbor_k, max(1, phrase_count - 1))
    effective_rank = _effective_rank(latent_coordinates)
    collapse_score = max(0.0, 1.0 - min(1.0, effective_rank / max(intrinsic_dim, 1)))
    return GeometryMetrics(
        piece_id=piece_id,
        geometry_kind=geometry_kind,
        phrase_count=phrase_count,
        intrinsic_dim=intrinsic_dim,
        structural_stress=structural_stress,
        trustworthiness=_trustworthiness(source_distances, latent_distances, k),
        continuity=_continuity(source_distances, latent_distances, k),
        neighbor_overlap=_neighbor_overlap(source_distances, latent_distances, k),
        effective_rank=effective_rank,
        collapse_score=collapse_score,
    )


def summarize_geometry_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate piece-level geometry metrics."""
    count = max(1, len(results))
    return {
        "piece_count": len(results),
        "geometry_kind": results[0]["geometry_kind"] if results else "unknown",
        "mean_structural_stress": sum(item["structural_stress"] for item in results) / count,
        "mean_trustworthiness": sum(item["trustworthiness"] for item in results) / count,
        "mean_continuity": sum(item["continuity"] for item in results) / count,
        "mean_neighbor_overlap": sum(item["neighbor_overlap"] for item in results) / count,
        "mean_effective_rank": sum(item["effective_rank"] for item in results) / count,
        "mean_collapse_score": sum(item["collapse_score"] for item in results) / count,
        "mean_phrase_count": sum(item["phrase_count"] for item in results) / count,
        "mean_intrinsic_dim": sum(item["intrinsic_dim"] for item in results) / count,
    }


def run_geometry_evaluation(
    checkpoint: str | Path,
    *,
    config_path: str | Path | None = None,
    processed_dir: str | Path | None = None,
    splits_dir: str | Path | None = None,
    split: str = "val",
    limit_pieces: int = 8,
    device: str = "auto",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate latent-geometry preservation for a torus-like checkpoint."""
    checkpoint_path = Path(checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = payload.get("config")
    if config is None:
        if config_path is None:
            raise ValueError("Config is required when checkpoint does not embed one.")
        config = load_config(config_path)
    if not config["model"].get("use_torus", False):
        raise ValueError("Geometry evaluation currently requires a torus-family checkpoint.")

    processed_root = Path(processed_dir or config["data"]["processed_dir"])
    splits_root = Path(splits_dir or config["data"]["splits_dir"])
    run_device = resolve_device(device)
    vocab_sizes = build_feature_vocab_sizes(config["tokenization"])
    model = build_torus_model(config, vocab_sizes=vocab_sizes)
    model.load_state_dict(payload["model_state"])
    model.to(run_device)
    model.eval()

    dataset = AutoregressiveTokenDataset(
        processed_dir=processed_root,
        splits_dir=splits_root,
        split=split,
        duration_bins=config["tokenization"]["duration_bins"],
        velocity_bins=config["tokenization"]["velocity_bins"],
        limit=limit_pieces,
        cache_examples=False,
    )
    max_length = config.get("training", {}).get("sequence_window", 512)

    results: list[dict[str, Any]] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = _truncate_sample(dataset[index], max_length=max_length)
            batch = collate_autoregressive_batch([sample])
            batch = move_batch_to_device(batch, run_device)
            output = model(
                batch.inputs,
                batch.attention_mask,
                phrase_ids=batch.phrase_ids,
                phrase_mask=batch.phrase_mask,
            )
            valid_phrase_mask = batch.phrase_mask[0]
            source_states = output.control_state[0, valid_phrase_mask].detach().cpu()
            latent_coordinates = output.latent_coordinates[0, valid_phrase_mask].detach().cpu()
            metrics = score_geometry(
                piece_id=sample["piece_id"],
                source_states=source_states,
                latent_coordinates=latent_coordinates,
                geometry_kind=output.latent_geometry,
            )
            results.append(asdict(metrics))

    summary = summarize_geometry_results(results)
    if output_dir is None:
        return {"summary": summary, "results": results}

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "geometry_metrics.jsonl"
    summary_path = output_root / "geometry_summary.json"
    metrics_path.write_text(
        "\n".join(json.dumps(item) for item in results) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "summary": summary,
        "results": results,
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
    }


def main() -> None:
    args = parse_args()
    result = run_geometry_evaluation(
        args.checkpoint,
        config_path=args.config,
        processed_dir=args.processed_dir,
        splits_dir=args.splits_dir,
        split=args.split,
        limit_pieces=args.limit_pieces,
        device=args.device,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
