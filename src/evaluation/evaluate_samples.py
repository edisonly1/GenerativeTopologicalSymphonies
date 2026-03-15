"""Evaluate generated or reference symbolic samples with structure metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evaluation.cadence import score_cadence_stability
from evaluation.fluency import build_interval_language_model, score_fluency
from evaluation.playability import score_playability
from evaluation.recurrence import score_recurrence
from evaluation.tonal import score_tonal_alignment
from preprocessing import load_quantized_piece_json
from tda.persistence import compute_persistence_summary
from tokenization import encode_piece_to_blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--reference-dir", default=None)
    parser.add_argument("--duration-bins", type=int, default=16)
    parser.add_argument("--velocity-bins", type=int, default=8)
    parser.add_argument("--bar-position-bins", type=int, default=16)
    return parser.parse_args()


def _discover_piece_paths(input_dir: Path) -> list[Path]:
    """Find quantized-piece JSON files inside a directory."""
    ignored_names = {"manifest.json", "summary.json", "metrics.jsonl"}
    paths = sorted(
        path
        for path in input_dir.rglob("*.json")
        if path.name not in ignored_names
    )
    if not paths:
        raise FileNotFoundError(f"No JSON pieces found in {input_dir}")
    return paths


def evaluate_piece(
    path: Path,
    *,
    reference_path: Path | None = None,
    interval_model: dict[int, float] | None = None,
    duration_bins: int,
    velocity_bins: int,
    bar_position_bins: int,
) -> dict[str, Any]:
    """Evaluate one quantized-piece JSON artifact."""
    quantized_piece = load_quantized_piece_json(path)
    example = encode_piece_to_blocks(
        quantized_piece,
        duration_bins=duration_bins,
        velocity_bins=velocity_bins,
        bar_position_bins=bar_position_bins,
    )
    reference_example = None
    if reference_path is not None:
        reference_piece = load_quantized_piece_json(reference_path)
        reference_example = encode_piece_to_blocks(
            reference_piece,
            duration_bins=duration_bins,
            velocity_bins=velocity_bins,
            bar_position_bins=bar_position_bins,
        )
    recurrence = score_recurrence(example)
    cadence = score_cadence_stability(example)
    fluency = score_fluency(quantized_piece, interval_model=interval_model)
    tonal = score_tonal_alignment(example, reference_example=reference_example)
    playability = score_playability(quantized_piece)
    persistence = compute_persistence_summary(example)
    return {
        "piece_id": quantized_piece.piece_id,
        "recurrence": asdict(recurrence),
        "cadence": asdict(cadence),
        "fluency": asdict(fluency),
        "tonal": asdict(tonal),
        "playability": asdict(playability),
        "persistence": asdict(persistence),
        "event_count": len(quantized_piece.note_events),
        "reference_piece_id": reference_example.piece_id if reference_example is not None else None,
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate piece-level metrics into a summary payload."""
    count = max(1, len(results))
    transition_perplexities = [
        item["fluency"]["transition_perplexity"]
        for item in results
        if item["fluency"]["transition_perplexity"] is not None
    ]
    pitch_class_divergences = [
        item["tonal"]["pitch_class_divergence"]
        for item in results
        if item["tonal"]["pitch_class_divergence"] is not None
    ]
    return {
        "piece_count": len(results),
        "mean_recurrent_phrase_ratio": sum(
            item["recurrence"]["recurrent_phrase_ratio"] for item in results
        ) / count,
        "mean_recurrence_similarity": sum(
            item["recurrence"]["mean_max_similarity"] for item in results
        ) / count,
        "mean_cadence_rate": sum(
            item["cadence"]["cadence_rate"] for item in results
        ) / count,
        "mean_final_duration": sum(
            item["cadence"]["mean_final_duration"] for item in results
        ) / count,
        "mean_event_count": sum(item["event_count"] for item in results) / count,
        "mean_pitch_jump": sum(
            item["fluency"]["mean_pitch_jump"] for item in results
        ) / count,
        "mean_large_jump_rate": sum(
            item["fluency"]["large_jump_rate"] for item in results
        ) / count,
        "mean_transition_perplexity": (
            sum(transition_perplexities) / len(transition_perplexities)
            if transition_perplexities
            else None
        ),
        "mean_pitch_class_entropy": sum(
            item["tonal"]["pitch_class_entropy"] for item in results
        ) / count,
        "mean_tonal_center_strength": sum(
            item["tonal"]["tonal_center_strength"] for item in results
        ) / count,
        "mean_pitch_class_divergence": (
            sum(pitch_class_divergences) / len(pitch_class_divergences)
            if pitch_class_divergences
            else None
        ),
        "mean_large_span_rate": sum(
            item["playability"]["large_span_rate"] for item in results
        ) / count,
        "mean_overlap_violation_rate": sum(
            item["playability"]["overlap_violation_rate"] for item in results
        ) / count,
        "mean_polyphony_peak": sum(
            item["playability"]["polyphony_peak"] for item in results
        ) / count,
        "mean_persistence_bar_count": sum(
            item["persistence"]["h1_bar_count"] for item in results
        ) / count,
        "mean_max_persistence": sum(
            item["persistence"]["max_persistence"] for item in results
        ) / count,
    }


def evaluate_directory(
    input_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    reference_dir: str | Path | None = None,
    duration_bins: int = 16,
    velocity_bins: int = 8,
    bar_position_bins: int = 16,
) -> dict[str, Any]:
    """Evaluate all quantized-piece JSON files inside a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir is not None else input_path
    output_path.mkdir(parents=True, exist_ok=True)
    piece_paths = _discover_piece_paths(input_path)
    reference_map: dict[str, Path] = {}
    interval_model = None
    if reference_dir is not None:
        reference_root = Path(reference_dir)
        reference_paths = _discover_piece_paths(reference_root)
        reference_map = {
            str(path.relative_to(reference_root).parent): path
            for path in reference_paths
        }
        reference_pieces = [load_quantized_piece_json(path) for path in reference_paths]
        interval_model = build_interval_language_model(reference_pieces)
    results = [
        evaluate_piece(
            piece_path,
            reference_path=reference_map.get(str(piece_path.relative_to(input_path).parent)),
            interval_model=interval_model,
            duration_bins=duration_bins,
            velocity_bins=velocity_bins,
            bar_position_bins=bar_position_bins,
        )
        for piece_path in piece_paths
    ]
    summary = summarize_results(results)
    (output_path / "metrics.jsonl").write_text(
        "\n".join(json.dumps(item) for item in results) + "\n",
        encoding="utf-8",
    )
    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "summary": summary,
        "metrics_path": str(output_path / "metrics.jsonl"),
        "summary_path": str(output_path / "summary.json"),
    }


def main() -> None:
    args = parse_args()
    result = evaluate_directory(
        args.input_dir,
        output_dir=args.output_dir,
        reference_dir=args.reference_dir,
        duration_bins=args.duration_bins,
        velocity_bins=args.velocity_bins,
        bar_position_bins=args.bar_position_bins,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
