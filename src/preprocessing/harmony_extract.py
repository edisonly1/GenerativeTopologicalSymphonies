"""Harmonic-context extraction helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any

from preprocessing.schema import QuantizedEvent, QuantizedPiece
from preprocessing.serialization import quantized_piece_from_dict


PITCH_CLASS_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
GLOBAL_HARMONY_LABELS = (
    ["unknown"]
    + [f"{name}:maj" for name in PITCH_CLASS_NAMES]
    + [f"{name}:min" for name in PITCH_CLASS_NAMES]
)
GLOBAL_HARMONY_VOCAB = {
    label: index
    for index, label in enumerate(GLOBAL_HARMONY_LABELS)
}
GLOBAL_HARMONY_ID_TO_LABEL = {
    index: label
    for label, index in GLOBAL_HARMONY_VOCAB.items()
}
PITCH_CLASS_BY_NAME = {
    name: index
    for index, name in enumerate(PITCH_CLASS_NAMES)
}


def transpose_pitch_class_name(name: str, semitones: int) -> str:
    """Transpose a pitch-class label by the given number of semitones."""
    pitch_class = PITCH_CLASS_BY_NAME.get(name)
    if pitch_class is None:
        return name
    return PITCH_CLASS_NAMES[(pitch_class + semitones) % 12]


def transpose_chord_label(label: str, semitones: int) -> str:
    """Transpose a simple triadic chord label such as C:maj or A:min."""
    if label == "unknown" or ":" not in label:
        return label
    root, quality = label.split(":", 1)
    return f"{transpose_pitch_class_name(root, semitones)}:{quality}"


def transpose_key_label(label: str, semitones: int) -> str:
    """Transpose a key label such as C:major or A:minor."""
    if label == "unknown" or ":" not in label:
        return label
    root, quality = label.split(":", 1)
    return f"{transpose_pitch_class_name(root, semitones)}:{quality}"


def _piece_pitch_histogram(piece: QuantizedPiece) -> list[float]:
    """Build a duration-weighted piece-level pitch-class histogram."""
    histogram = [0.0] * 12
    for event in piece.note_events:
        histogram[event.pitch % 12] += float(event.duration_steps)
    return histogram


def _rotate(values: list[float], offset: int) -> list[float]:
    """Rotate a pitch-class profile by scale degree."""
    return values[-offset:] + values[:-offset] if offset else list(values)


def _correlation(left: list[float], right: list[float]) -> float:
    """Compute a simple centered correlation score."""
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_norm = sum((a - left_mean) ** 2 for a in left) ** 0.5
    right_norm = sum((b - right_mean) ** 2 for b in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _estimate_key(piece: QuantizedPiece) -> str:
    """Estimate a global major/minor key from weighted pitch classes."""
    histogram = _piece_pitch_histogram(piece)
    best_label = "unknown"
    best_score = float("-inf")
    for pitch_class, name in enumerate(PITCH_CLASS_NAMES):
        major_score = _correlation(histogram, _rotate(MAJOR_PROFILE, pitch_class))
        if major_score > best_score:
            best_score = major_score
            best_label = f"{name}:major"
        minor_score = _correlation(histogram, _rotate(MINOR_PROFILE, pitch_class))
        if minor_score > best_score:
            best_score = minor_score
            best_label = f"{name}:minor"
    return best_label


def _estimate_bar_chords(piece: QuantizedPiece) -> list[str]:
    """Estimate one simple triadic label per bar."""
    bar_histograms: dict[int, list[float]] = defaultdict(lambda: [0.0] * 12)
    for event in piece.note_events:
        bar_histograms[event.bar][event.pitch % 12] += float(event.duration_steps)

    labels: list[str] = []
    total_bars = max(piece.total_bars, 1)
    for bar in range(1, total_bars + 1):
        histogram = bar_histograms.get(bar)
        if histogram is None or max(histogram) <= 0.0:
            labels.append("unknown")
            continue
        root = max(range(12), key=histogram.__getitem__)
        major_score = histogram[(root + 4) % 12] + histogram[(root + 7) % 12]
        minor_score = histogram[(root + 3) % 12] + histogram[(root + 7) % 12]
        quality = "maj" if major_score >= minor_score else "min"
        labels.append(f"{PITCH_CLASS_NAMES[root]}:{quality}")
    return labels


def extract_harmony(piece: QuantizedPiece | dict[str, Any]) -> dict[str, Any]:
    """Estimate lightweight key and bar-level chord annotations."""
    if isinstance(piece, dict):
        quantized_piece = quantized_piece_from_dict(piece)
    else:
        quantized_piece = piece
    chords = _estimate_bar_chords(quantized_piece)
    tonic = chords[0] if chords else "unknown"
    cadence_bars = [
        index + 1
        for index, chord in enumerate(chords)
        if chord == tonic and index > 0
    ]
    return {
        "key": _estimate_key(quantized_piece),
        "chords": chords,
        "cadence_bars": cadence_bars,
    }


def annotate_quantized_piece_harmony(piece: QuantizedPiece | dict[str, Any]) -> QuantizedPiece:
    """Attach estimated harmony labels directly to each quantized event."""
    if isinstance(piece, dict):
        quantized_piece = quantized_piece_from_dict(piece)
    else:
        quantized_piece = piece

    harmony = extract_harmony(quantized_piece)
    chords = harmony["chords"]
    annotated_events: list[QuantizedEvent] = []
    for event in quantized_piece.note_events:
        chord_index = max(0, min(len(chords) - 1, event.bar - 1))
        chord_label = chords[chord_index] if chords else "unknown"
        annotated_events.append(replace(event, harmony=chord_label))

    metadata = dict(quantized_piece.metadata)
    metadata["cadence_bars"] = list(harmony["cadence_bars"])
    metadata["global_harmony_vocab"] = GLOBAL_HARMONY_VOCAB
    return replace(
        quantized_piece,
        note_events=annotated_events,
        metadata=metadata,
        key=harmony["key"],
        chords=list(chords),
    )
