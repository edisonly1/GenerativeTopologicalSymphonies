"""Heuristic phrase-level supervision targets for the conductor stage."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean

from tokenization import PieceExample


CONDUCTOR_TARGET_NAMES = (
    "recurrence",
    "tension",
    "density",
    "cadence",
    "harmonic_zone",
)

DEFAULT_CONDUCTOR_TARGET_VOCAB_SIZES = {
    "recurrence": 2,
    "tension": 4,
    "density": 4,
    "cadence": 2,
    "harmonic_zone": 12,
}


@dataclass(slots=True)
class PhraseControlTargets:
    """Phrase-aligned supervision targets for one tokenized piece."""

    phrase_ids: list[int]
    phrase_ranges: list[tuple[int, int]]
    targets: dict[str, list[int]]


def _pitch_class_histogram(blocks) -> list[float]:
    """Build a duration-weighted pitch-class histogram."""
    histogram = [0.0] * 12
    for block in blocks:
        histogram[block.pitch % 12] += float(block.duration + 1)
    total = sum(histogram)
    if total <= 0:
        return histogram
    return [value / total for value in histogram]


def _duration_histogram(blocks) -> list[float]:
    """Build a normalized duration-bucket histogram."""
    if not blocks:
        return [0.0]
    bucket_count = max(block.duration for block in blocks) + 1
    histogram = [0.0] * bucket_count
    for block in blocks:
        histogram[block.duration] += 1.0
    total = sum(histogram)
    return [value / total for value in histogram]


def _position_histogram(blocks) -> list[float]:
    """Build a normalized bar-position histogram."""
    if not blocks:
        return [0.0]
    bucket_count = max(block.bar_position for block in blocks) + 1
    histogram = [0.0] * bucket_count
    for block in blocks:
        histogram[block.bar_position] += 1.0
    total = sum(histogram)
    return [value / total for value in histogram]


def _interval_histogram(blocks) -> list[float]:
    """Build a clipped melodic-interval histogram for transposition-aware matching."""
    bucket_count = 25
    if len(blocks) < 2:
        return [0.0] * bucket_count
    histogram = [0.0] * bucket_count
    for index in range(1, len(blocks)):
        interval = blocks[index].pitch - blocks[index - 1].pitch
        clipped = max(-12, min(12, interval))
        histogram[clipped + 12] += 1.0
    total = sum(histogram)
    return [value / total for value in histogram]


def _interval_profile(blocks, *, profile_length: int = 8) -> list[int]:
    """Capture the leading melodic contour as clipped pitch intervals."""
    if not blocks:
        return []
    profile = [0]
    for index in range(1, len(blocks)):
        interval = blocks[index].pitch - blocks[index - 1].pitch
        profile.append(max(-12, min(12, interval)))
        if len(profile) >= profile_length:
            break
    return profile


def _rhythm_profile(blocks, *, profile_length: int = 8) -> list[tuple[int, int]]:
    """Capture the leading rhythmic profile as duration/onset pairs."""
    return [
        (int(block.duration), int(block.bar_position))
        for block in blocks[:profile_length]
    ]


def _bar_span(blocks) -> int:
    """Measure how many bars a phrase spans."""
    if not blocks:
        return 0
    return max(block.bar for block in blocks) - min(block.bar for block in blocks) + 1


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity between two histograms."""
    target_size = max(len(left), len(right))
    if len(left) < target_size:
        left = left + [0.0] * (target_size - len(left))
    if len(right) < target_size:
        right = right + [0.0] * (target_size - len(right))
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _sign(value: int) -> int:
    """Return the sign of an integer in {-1, 0, 1}."""
    if value == 0:
        return 0
    return 1 if value > 0 else -1


def _interval_profile_similarity(left: list[int], right: list[int]) -> float:
    """Compare melodic contour profiles with tolerance for small interval drift."""
    compare_length = min(len(left), len(right))
    if compare_length == 0:
        return 0.0
    score = 0.0
    for left_value, right_value in zip(left[:compare_length], right[:compare_length], strict=True):
        if abs(left_value - right_value) <= 1:
            score += 1.0
        elif _sign(left_value) == _sign(right_value) and abs(left_value - right_value) <= 3:
            score += 0.5
    return score / compare_length


def _rhythm_profile_similarity(
    left: list[tuple[int, int]],
    right: list[tuple[int, int]],
) -> float:
    """Compare rhythmic profiles with tolerance for near-aligned durations/onsets."""
    compare_length = min(len(left), len(right))
    if compare_length == 0:
        return 0.0
    score = 0.0
    for (left_duration, left_position), (right_duration, right_position) in zip(
        left[:compare_length],
        right[:compare_length],
        strict=True,
    ):
        duration_match = abs(left_duration - right_duration) <= 1
        position_match = abs(left_position - right_position) <= 2
        if duration_match and position_match:
            score += 1.0
        elif duration_match or position_match:
            score += 0.5
    return score / compare_length


def _harmonic_zone(blocks) -> int:
    """Approximate a local key region with the dominant pitch class."""
    histogram = _pitch_class_histogram(blocks)
    return max(range(12), key=histogram.__getitem__)


def _density_bucket(blocks) -> int:
    """Bucket phrase density based on notes per bar."""
    if not blocks:
        return 0
    bars_covered = max(1, max(block.bar for block in blocks) - min(block.bar for block in blocks) + 1)
    notes_per_bar = len(blocks) / bars_covered
    if notes_per_bar <= 4.0:
        return 0
    if notes_per_bar <= 8.0:
        return 1
    if notes_per_bar <= 16.0:
        return 2
    return 3


def _cadence_target(blocks, harmonic_zone: int) -> int:
    """Approximate whether a phrase ends with a stable cadence-like gesture."""
    if not blocks:
        return 0
    stable_pitch_classes = {
        harmonic_zone,
        (harmonic_zone + 4) % 12,
        (harmonic_zone + 7) % 12,
    }
    durations = sorted(block.duration for block in blocks)
    median_duration = durations[len(durations) // 2]
    last_block = blocks[-1]
    return int(
        last_block.pitch % 12 in stable_pitch_classes
        and last_block.duration >= median_duration
    )


def _tension_bucket(blocks, *, density_bucket: int, cadence_target: int) -> int:
    """Approximate phrase tension from diversity, leapiness, density, and closure."""
    if not blocks:
        return 0
    unique_pitch_classes = len({block.pitch % 12 for block in blocks})
    pitch_class_score = min(1.0, unique_pitch_classes / 7.0)
    if len(blocks) > 1:
        mean_leap = mean(
            abs(blocks[index].pitch - blocks[index - 1].pitch)
            for index in range(1, len(blocks))
        )
    else:
        mean_leap = 0.0
    leap_score = min(1.0, mean_leap / 12.0)
    density_score = density_bucket / 3.0
    unresolved_bonus = 0.2 if cadence_target == 0 else 0.0
    score = 0.4 * pitch_class_score + 0.3 * leap_score + 0.2 * density_score + 0.1 * unresolved_bonus
    return min(3, max(0, int(score * 4.0)))


def _recurrence_target(
    phrase_index: int,
    phrase_features: dict[int, dict[str, object]],
) -> int:
    """Mark phrases that closely resemble an earlier phrase in melody and rhythm."""
    current = phrase_features[phrase_index]
    for previous_index in range(phrase_index):
        previous = phrase_features.get(previous_index)
        if previous is None:
            continue
        pitch_similarity = _cosine_similarity(
            current["pitch_histogram"],
            previous["pitch_histogram"],
        )
        duration_similarity = _cosine_similarity(
            current["duration_histogram"],
            previous["duration_histogram"],
        )
        position_similarity = _cosine_similarity(
            current["position_histogram"],
            previous["position_histogram"],
        )
        interval_similarity = _cosine_similarity(
            current["interval_histogram"],
            previous["interval_histogram"],
        )
        interval_profile_similarity = _interval_profile_similarity(
            current["interval_profile"],
            previous["interval_profile"],
        )
        rhythm_profile_similarity = _rhythm_profile_similarity(
            current["rhythm_profile"],
            previous["rhythm_profile"],
        )
        event_count_ratio = min(current["event_count"], previous["event_count"]) / max(
            current["event_count"],
            previous["event_count"],
        )
        bar_span_ratio = min(current["bar_span"], previous["bar_span"]) / max(
            current["bar_span"],
            previous["bar_span"],
        )
        structural_similarity = (
            0.28 * pitch_similarity
            + 0.14 * duration_similarity
            + 0.14 * position_similarity
            + 0.14 * interval_similarity
            + 0.12 * interval_profile_similarity
            + 0.12 * rhythm_profile_similarity
            + 0.03 * event_count_ratio
            + 0.03 * bar_span_ratio
        )
        if (
            structural_similarity >= 0.86
            and pitch_similarity >= 0.90
            and interval_profile_similarity >= 0.50
            and rhythm_profile_similarity >= 0.45
            and event_count_ratio >= 0.70
            and bar_span_ratio >= 0.75
        ):
            return 1
    return 0


def _compute_phrase_ranges(phrase_ids: list[int]) -> list[tuple[int, int]]:
    """Convert a phrase-id stream into contiguous [start, end) spans."""
    if not phrase_ids:
        return []
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(phrase_ids)):
        if phrase_ids[index] != phrase_ids[index - 1]:
            ranges.append((start, index))
            start = index
    ranges.append((start, len(phrase_ids)))
    return ranges


def derive_phrase_control_targets(example: PieceExample) -> PhraseControlTargets:
    """Derive phrase-level heuristic targets aligned to autoregressive positions."""
    if len(example.event_blocks) <= 1:
        empty_targets = {name: [] for name in CONDUCTOR_TARGET_NAMES}
        return PhraseControlTargets(phrase_ids=[], phrase_ranges=[], targets=empty_targets)

    full_phrase_blocks: dict[int, list] = {}
    for block in example.event_blocks:
        full_phrase_blocks.setdefault(block.phrase_index, []).append(block)

    shifted_blocks = example.event_blocks[1:]
    visible_phrase_indices: list[int] = []
    local_phrase_by_global: dict[int, int] = {}
    for block in shifted_blocks:
        if block.phrase_index not in local_phrase_by_global:
            local_phrase_by_global[block.phrase_index] = len(visible_phrase_indices)
            visible_phrase_indices.append(block.phrase_index)

    phrase_ids = [local_phrase_by_global[block.phrase_index] for block in shifted_blocks]
    phrase_ranges = _compute_phrase_ranges(phrase_ids)

    phrase_features = {
        phrase_index: {
            "pitch_histogram": _pitch_class_histogram(blocks),
            "duration_histogram": _duration_histogram(blocks),
            "position_histogram": _position_histogram(blocks),
            "interval_histogram": _interval_histogram(blocks),
            "interval_profile": _interval_profile(blocks),
            "rhythm_profile": _rhythm_profile(blocks),
            "event_count": len(blocks),
            "bar_span": _bar_span(blocks),
        }
        for phrase_index, blocks in full_phrase_blocks.items()
    }
    targets = {name: [] for name in CONDUCTOR_TARGET_NAMES}
    for phrase_index in visible_phrase_indices:
        blocks = full_phrase_blocks[phrase_index]
        harmonic_zone = _harmonic_zone(blocks)
        density_bucket = _density_bucket(blocks)
        cadence_target = _cadence_target(blocks, harmonic_zone)
        targets["recurrence"].append(_recurrence_target(phrase_index, phrase_features))
        targets["tension"].append(
            _tension_bucket(
                blocks,
                density_bucket=density_bucket,
                cadence_target=cadence_target,
            )
        )
        targets["density"].append(density_bucket)
        targets["cadence"].append(cadence_target)
        targets["harmonic_zone"].append(harmonic_zone)

    return PhraseControlTargets(
        phrase_ids=phrase_ids,
        phrase_ranges=phrase_ranges,
        targets=targets,
    )
