"""Persistent-homology helpers for structural evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tokenization import PieceExample

try:
    from ripser import ripser
except Exception:  # pragma: no cover - optional dependency
    ripser = None


@dataclass(slots=True)
class PersistenceSummary:
    """Compact H1 persistence summary."""

    phrase_count: int
    h1_bar_count: int
    max_persistence: float
    mean_persistence: float
    method: str


def _phrase_histograms(example: PieceExample) -> list[list[float]]:
    """Convert phrases into duration-weighted pitch-class histograms."""
    phrase_map: dict[int, list[float]] = {}
    for block in example.event_blocks:
        histogram = phrase_map.setdefault(block.phrase_index, [0.0] * 12)
        histogram[block.pitch % 12] += float(block.duration + 1)
    histograms: list[list[float]] = []
    for phrase_index in sorted(phrase_map):
        histogram = phrase_map[phrase_index]
        total = sum(histogram)
        if total > 0.0:
            histogram = [value / total for value in histogram]
        histograms.append(histogram)
    return histograms


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity between two phrase histograms."""
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    denominator = np.linalg.norm(left_array) * np.linalg.norm(right_array)
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left_array, right_array) / denominator)


def _proxy_persistence(histograms: list[list[float]]) -> PersistenceSummary:
    """Fallback persistence proxy when ripser is unavailable."""
    persistence_values: list[float] = []
    for index, histogram in enumerate(histograms):
        for other_index in range(index + 2, len(histograms)):
            similarity = _cosine_similarity(histogram, histograms[other_index])
            persistence = max(0.0, similarity - 0.75)
            if persistence > 0.0:
                persistence_values.append(persistence)
    if not persistence_values:
        return PersistenceSummary(
            phrase_count=len(histograms),
            h1_bar_count=0,
            max_persistence=0.0,
            mean_persistence=0.0,
            method="similarity_proxy",
        )
    return PersistenceSummary(
        phrase_count=len(histograms),
        h1_bar_count=len(persistence_values),
        max_persistence=max(persistence_values),
        mean_persistence=sum(persistence_values) / len(persistence_values),
        method="similarity_proxy",
    )


def compute_persistence_summary(example: PieceExample) -> PersistenceSummary:
    """Compute an H1 persistence summary from phrase trajectories."""
    histograms = _phrase_histograms(example)
    if len(histograms) <= 2:
        return PersistenceSummary(
            phrase_count=len(histograms),
            h1_bar_count=0,
            max_persistence=0.0,
            mean_persistence=0.0,
            method="degenerate",
        )
    if ripser is None:
        return _proxy_persistence(histograms)

    point_cloud = np.asarray(
        [
            histogram + [index / max(len(histograms) - 1, 1)]
            for index, histogram in enumerate(histograms)
        ],
        dtype=float,
    )
    diagrams = ripser(point_cloud, maxdim=1)["dgms"]
    persistence_values = [
        float(death - birth)
        for birth, death in diagrams[1]
        if np.isfinite(death)
    ]
    if not persistence_values:
        return PersistenceSummary(
            phrase_count=len(histograms),
            h1_bar_count=0,
            max_persistence=0.0,
            mean_persistence=0.0,
            method="ripser",
        )
    return PersistenceSummary(
        phrase_count=len(histograms),
        h1_bar_count=len(persistence_values),
        max_persistence=max(persistence_values),
        mean_persistence=sum(persistence_values) / len(persistence_values),
        method="ripser",
    )

