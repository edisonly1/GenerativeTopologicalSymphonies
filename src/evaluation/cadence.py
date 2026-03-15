"""Cadence and phrase-resolution metrics."""

from __future__ import annotations

from dataclasses import dataclass

from tokenization import PieceExample


@dataclass(slots=True)
class CadenceMetrics:
    """Phrase-ending stability summary for one piece."""

    phrase_count: int
    cadence_rate: float
    mean_final_duration: float


def _group_phrase_blocks(example: PieceExample) -> list[list]:
    """Collect event blocks by phrase index."""
    phrase_map: dict[int, list] = {}
    for block in example.event_blocks:
        phrase_map.setdefault(block.phrase_index, []).append(block)
    return [phrase_map[index] for index in sorted(phrase_map)]


def _harmonic_zone(blocks) -> int:
    """Approximate a local harmonic zone from weighted pitch-class counts."""
    histogram = [0.0] * 12
    for block in blocks:
        histogram[block.pitch % 12] += float(block.duration + 1)
    return max(range(12), key=histogram.__getitem__)


def _is_stable_phrase_ending(blocks) -> bool:
    """Heuristic cadence detector based on phrase-final stability."""
    if not blocks:
        return False
    final_block = blocks[-1]
    harmonic_zone = _harmonic_zone(blocks)
    stable_pitch_classes = {
        harmonic_zone,
        (harmonic_zone + 4) % 12,
        (harmonic_zone + 7) % 12,
    }
    durations = sorted(block.duration for block in blocks)
    median_duration = durations[len(durations) // 2]
    return (
        final_block.pitch % 12 in stable_pitch_classes
        and final_block.duration >= median_duration
    )


def score_cadence_stability(example: PieceExample) -> CadenceMetrics:
    """Measure the rate of phrase endings that resolve to stable local targets."""
    phrase_blocks = _group_phrase_blocks(example)
    phrase_count = len(phrase_blocks)
    if phrase_count == 0:
        return CadenceMetrics(phrase_count=0, cadence_rate=0.0, mean_final_duration=0.0)

    cadence_hits = sum(1 for blocks in phrase_blocks if _is_stable_phrase_ending(blocks))
    mean_final_duration = sum((blocks[-1].duration + 1) for blocks in phrase_blocks if blocks) / phrase_count
    return CadenceMetrics(
        phrase_count=phrase_count,
        cadence_rate=cadence_hits / phrase_count,
        mean_final_duration=mean_final_duration,
    )
