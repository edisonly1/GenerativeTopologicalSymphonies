"""Sweep decoding settings for one checkpoint against poster-aligned music metrics."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import torch

from evaluation.evaluate_samples import evaluate_directory
from evaluation.matched_reference import build_matched_reference_set
from inference.generate import generate_from_checkpoint
from training.train_baseline import load_config


OBJECTIVE_METRICS: dict[str, dict[str, Any]] = {
    "mean_recurrence_similarity": {"mode": "reference_closeness", "weight": 1.0},
    "mean_recurrent_phrase_ratio": {"mode": "reference_closeness", "weight": 0.8},
    "mean_pitch_jump": {"mode": "reference_closeness", "weight": 1.1},
    "mean_large_span_rate": {"mode": "reference_closeness", "weight": 1.0},
    "mean_overlap_violation_rate": {"mode": "reference_closeness", "weight": 1.2},
    "mean_pitch_class_entropy": {"mode": "reference_closeness", "weight": 1.0},
    "mean_final_duration": {"mode": "reference_closeness", "weight": 0.9},
    "mean_max_persistence": {"mode": "reference_closeness", "weight": 0.6},
    "mean_cadence_rate": {"mode": "reference_closeness", "weight": 0.5},
    "mean_tonal_center_strength": {"mode": "reference_closeness", "weight": 0.4},
    "mean_pitch_class_divergence": {"mode": "baseline_reduction", "weight": 1.4},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--phase6-report", required=True)
    parser.add_argument("--baseline-stage", default="baseline")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit-pieces", type=int, default=8)
    parser.add_argument("--prompt-events", type=int, default=64)
    parser.add_argument("--generate-events", type=int, default=96)
    parser.add_argument("--temperatures", default="0.78,0.85,0.92")
    parser.add_argument("--top-ks", default="0,4,8")
    parser.add_argument("--top-ps", default="0.9,0.95")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _parse_float_grid(raw_value: str) -> list[float]:
    return [float(item.strip()) for item in raw_value.split(",") if item.strip()]


def _parse_int_grid(raw_value: str) -> list[int]:
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _slugify_combo(temperature: float, top_k: int, top_p: float) -> str:
    return (
        f"temp_{temperature:.2f}".replace(".", "p")
        + f"__topk_{top_k}"
        + f"__topp_{top_p:.2f}".replace(".", "p")
    )


def _reference_closeness_gain(
    baseline_value: float | None,
    candidate_value: float | None,
    reference_value: float | None,
) -> float | None:
    if baseline_value is None or candidate_value is None or reference_value is None:
        return None
    baseline_error = abs(baseline_value - reference_value)
    candidate_error = abs(candidate_value - reference_value)
    scale = max(baseline_error, 1e-8)
    return (baseline_error - candidate_error) / scale


def _baseline_reduction(
    baseline_value: float | None,
    candidate_value: float | None,
) -> float | None:
    if baseline_value is None or candidate_value is None:
        return None
    scale = max(abs(baseline_value), 1e-8)
    return (baseline_value - candidate_value) / scale


def _compute_objective(
    candidate_summary: dict[str, Any],
    *,
    baseline_summary: dict[str, Any],
    reference_summary: dict[str, Any],
) -> tuple[float, dict[str, float | None]]:
    contributions: dict[str, float | None] = {}
    total = 0.0
    total_weight = 0.0
    for metric_name, metric_config in OBJECTIVE_METRICS.items():
        if metric_config["mode"] == "reference_closeness":
            score = _reference_closeness_gain(
                baseline_summary.get(metric_name),
                candidate_summary.get(metric_name),
                reference_summary.get(metric_name),
            )
        else:
            score = _baseline_reduction(
                baseline_summary.get(metric_name),
                candidate_summary.get(metric_name),
            )
        contributions[metric_name] = score
        if score is None:
            continue
        weight = float(metric_config["weight"])
        total += score * weight
        total_weight += weight
    normalized = total / total_weight if total_weight > 0.0 else 0.0
    return normalized, contributions


def _percent_reduction(baseline_value: float | None, candidate_value: float | None) -> float | None:
    if baseline_value is None or candidate_value is None or baseline_value == 0:
        return None
    return ((baseline_value - candidate_value) / baseline_value) * 100.0


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Decoding Sweep",
        "",
        f"Checkpoint: `{report['checkpoint']}`",
        "",
        "## Best Setting",
        "",
        f"- temperature: `{report['best']['temperature']}`",
        f"- top_k: `{report['best']['top_k']}`",
        f"- top_p: `{report['best']['top_p']}`",
        f"- objective_score: `{report['best']['objective_score']:.4f}`",
        "",
        "## Top Results",
        "",
        "| temperature | top_k | top_p | objective | recurrence | jump | tonal_div | span | overlap | entropy | duration |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in report["results"][: min(10, len(report["results"]))]:
        summary = result["summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{result['temperature']:.2f}",
                    str(result["top_k"]),
                    f"{result['top_p']:.2f}",
                    f"{result['objective_score']:.4f}",
                    f"{summary['mean_recurrence_similarity']:.4f}",
                    f"{summary['mean_pitch_jump']:.4f}",
                    f"{summary['mean_pitch_class_divergence']:.4f}",
                    f"{summary['mean_large_span_rate']:.4f}",
                    f"{summary['mean_overlap_violation_rate']:.4f}",
                    f"{summary['mean_pitch_class_entropy']:.4f}",
                    f"{summary['mean_final_duration']:.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def run_decoding_sweep(
    *,
    checkpoint: str | Path,
    phase6_report_path: str | Path,
    config_path: str | Path | None,
    processed_dir: str | Path | None,
    splits_dir: str | Path | None,
    split: str,
    limit_pieces: int,
    prompt_events: int,
    generate_events: int,
    temperatures: list[float],
    top_ks: list[int],
    top_ps: list[float],
    device: str,
    seed: int,
    output_dir: str | Path,
    baseline_stage: str,
    skip_existing: bool,
) -> dict[str, Any]:
    phase6_report = _load_json(phase6_report_path)
    baseline_summary = phase6_report["stages"][baseline_stage]["evaluation"]["summary"]
    reference_summary = phase6_report["reference"]["summary"]
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    config = None
    if config_path is not None:
        config = load_config(config_path)
    else:
        payload = torch.load(Path(checkpoint), map_location="cpu")
        config = payload.get("config")
        if config is None:
            raise ValueError("Config path is required when the checkpoint does not store its config.")

    processed_root = Path(processed_dir or config["data"]["processed_dir"])
    reference_dir = output_root / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    reference_built = reference_dir.joinpath("manifest.json").exists()

    for temperature, top_k, top_p in itertools.product(temperatures, top_ks, top_ps):
        combo_slug = _slugify_combo(temperature, top_k, top_p)
        combo_root = output_root / combo_slug
        samples_dir = combo_root / "samples"
        evaluation_dir = combo_root / "evaluation"
        summary_path = evaluation_dir / "summary.json"

        if skip_existing and summary_path.exists():
            evaluation = {
                "summary": _load_json(summary_path),
                "summary_path": str(summary_path),
                "metrics_path": str(evaluation_dir / "metrics.jsonl"),
            }
            manifest_path = samples_dir / "manifest.json"
        else:
            manifest = generate_from_checkpoint(
                checkpoint,
                config_path=config_path,
                processed_dir=processed_root,
                splits_dir=splits_dir,
                split=split,
                limit_pieces=limit_pieces,
                prompt_events=prompt_events,
                generate_events=generate_events,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=device,
                output_dir=samples_dir,
                seed=seed,
            )
            manifest_path = Path(manifest["manifest_path"])
            if not reference_built:
                build_matched_reference_set(
                    manifest_path,
                    processed_dir=processed_root,
                    output_dir=reference_dir,
                )
                evaluate_directory(
                    reference_dir,
                    output_dir=reference_dir,
                    duration_bins=config["tokenization"]["duration_bins"],
                    velocity_bins=config["tokenization"]["velocity_bins"],
                )
                reference_built = True
            evaluation = evaluate_directory(
                samples_dir,
                output_dir=evaluation_dir,
                reference_dir=reference_dir,
                duration_bins=config["tokenization"]["duration_bins"],
                velocity_bins=config["tokenization"]["velocity_bins"],
            )

        objective_score, objective_breakdown = _compute_objective(
            evaluation["summary"],
            baseline_summary=baseline_summary,
            reference_summary=reference_summary,
        )
        results.append(
            {
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "objective_score": objective_score,
                "objective_breakdown": objective_breakdown,
                "summary": evaluation["summary"],
                "summary_path": evaluation["summary_path"],
                "metrics_path": evaluation["metrics_path"],
                "manifest_path": str(manifest_path),
                "jump_reduction_percent": _percent_reduction(
                    baseline_summary.get("mean_pitch_jump"),
                    evaluation["summary"].get("mean_pitch_jump"),
                ),
                "tonal_gain_percent": _percent_reduction(
                    baseline_summary.get("mean_pitch_class_divergence"),
                    evaluation["summary"].get("mean_pitch_class_divergence"),
                ),
                "persistence_ratio": _ratio(
                    evaluation["summary"].get("mean_max_persistence"),
                    baseline_summary.get("mean_max_persistence"),
                ),
            }
        )

    results.sort(key=lambda item: item["objective_score"], reverse=True)
    best = results[0] if results else None
    report = {
        "checkpoint": str(checkpoint),
        "phase6_report_path": str(phase6_report_path),
        "output_dir": str(output_root),
        "baseline_stage": baseline_stage,
        "baseline_summary": baseline_summary,
        "reference_summary": reference_summary,
        "results": results,
        "best": best,
    }
    _write_json(output_root / "report.json", report)
    (output_root / "report.md").write_text(_render_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    result = run_decoding_sweep(
        checkpoint=args.checkpoint,
        phase6_report_path=args.phase6_report,
        config_path=args.config,
        processed_dir=args.processed_dir,
        splits_dir=args.splits_dir,
        split=args.split,
        limit_pieces=args.limit_pieces,
        prompt_events=args.prompt_events,
        generate_events=args.generate_events,
        temperatures=_parse_float_grid(args.temperatures),
        top_ks=_parse_int_grid(args.top_ks),
        top_ps=_parse_float_grid(args.top_ps),
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
        baseline_stage=args.baseline_stage,
        skip_existing=args.skip_existing,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
