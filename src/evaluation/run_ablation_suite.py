"""Run the Phase 6 ablation and packaging workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from evaluation.evaluate_samples import evaluate_directory
from evaluation.geometry import run_geometry_evaluation
from evaluation.matched_reference import build_matched_reference_set
from inference.cleanup import cleanup_directory
from inference.generate import generate_from_checkpoint
from inference.refine import refine_directory
from training.train_baseline import load_config


METRIC_SUMMARY_KEYS = {
    "recurrence": "mean_recurrence_similarity",
    "cadence_stability": "mean_cadence_rate",
    "jump_distance": "mean_pitch_jump",
    "pitch_class_divergence": "mean_pitch_class_divergence",
    "perplexity": "mean_transition_perplexity",
    "playability": "mean_large_span_rate",
    "persistence": "mean_max_persistence",
    "structural_stress": "mean_structural_stress",
    "latent_trustworthiness": "mean_trustworthiness",
    "latent_continuity": "mean_continuity",
    "latent_neighbor_overlap": "mean_neighbor_overlap",
    "latent_collapse": "mean_collapse_score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/full_model.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit-pieces", type=int, default=None)
    parser.add_argument("--prompt-events", type=int, default=None)
    parser.add_argument("--generate-events", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _maybe_load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON artifact if it already exists."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_generation_stage(
    stage_name: str,
    stage_config: dict[str, Any],
    *,
    suite_config: dict[str, Any],
    output_root: Path,
    limit_pieces: int,
    prompt_events: int,
    generate_events: int,
    device: str,
    seed: int,
    skip_existing: bool,
) -> dict[str, Any]:
    """Run one generation stage or reuse an existing manifest."""
    samples_dir = output_root / "samples" / stage_name
    manifest_path = samples_dir / "manifest.json"
    if skip_existing:
        existing = _maybe_load_json(manifest_path)
        if existing is not None:
            if "manifest_path" not in existing:
                existing["manifest_path"] = str(manifest_path)
            if "piece_count" not in existing:
                existing["piece_count"] = len(existing.get("items", []))
            return existing

    stage_mode = stage_config.get("mode", "generate")
    shared_kwargs = {
        "processed_dir": suite_config["data"]["processed_dir"],
        "splits_dir": suite_config["data"]["splits_dir"],
        "split": suite_config["evaluation"].get("split", "val"),
        "limit_pieces": limit_pieces,
        "prompt_events": prompt_events,
        "generate_events": generate_events,
        "device": device,
        "seed": seed,
    }
    if stage_mode == "cleanup":
        source_stage = stage_config.get("draft_stage") or stage_config.get("source_stage")
        if not source_stage:
            raise ValueError(f"Cleanup stage '{stage_name}' requires draft_stage or source_stage.")
        source_dir = output_root / "samples" / source_stage
        if not source_dir.exists():
            if stage_config.get("skip_if_missing", False):
                return {"skipped": True, "reason": f"missing source stage directory: {source_dir}"}
            raise FileNotFoundError(f"Cleanup source stage directory not found for {stage_name}: {source_dir}")
        return cleanup_directory(
            source_dir,
            output_dir=samples_dir,
            preserve_prefix_events=stage_config.get("preserve_prefix_events", prompt_events),
            max_notes_per_onset=stage_config.get("max_notes_per_onset", 6),
            max_simultaneous_span=stage_config.get("max_simultaneous_span", 24),
            trim_same_pitch_overlaps=stage_config.get("trim_same_pitch_overlaps", True),
        )
    checkpoint_path = Path(stage_config["checkpoint"])
    if not checkpoint_path.exists():
        if stage_config.get("skip_if_missing", False):
            return {"skipped": True, "reason": f"missing checkpoint: {checkpoint_path}"}
        raise FileNotFoundError(f"Stage checkpoint not found for {stage_name}: {checkpoint_path}")

    if stage_mode == "refine":
        draft_stage = stage_config.get("draft_stage")
        if draft_stage:
            draft_dir = output_root / "samples" / draft_stage
        else:
            draft_dir = output_root / "drafts" / stage_name
            draft_manifest_path = draft_dir / "manifest.json"
            if not (skip_existing and draft_manifest_path.exists()):
                generate_from_checkpoint(
                    stage_config["draft_checkpoint"],
                    config_path=stage_config.get("draft_config"),
                    output_dir=draft_dir,
                    **shared_kwargs,
                )
        return refine_directory(
            stage_config["checkpoint"],
            config_path=stage_config.get("config"),
            primary_checkpoint=stage_config.get("primary_checkpoint"),
            input_dir=draft_dir,
            output_dir=samples_dir,
            preserve_prefix_events=stage_config.get("preserve_prefix_events", prompt_events),
            device=device,
        )

    return generate_from_checkpoint(
        checkpoint_path,
        config_path=stage_config.get("config"),
        temperature=stage_config.get("temperature"),
        top_k=stage_config.get("top_k"),
        top_p=stage_config.get("top_p"),
        output_dir=samples_dir,
        **shared_kwargs,
    )


def _run_geometry_stage(
    stage_name: str,
    stage_config: dict[str, Any],
    *,
    suite_config: dict[str, Any],
    output_root: Path,
    limit_pieces: int,
    device: str,
    skip_existing: bool,
) -> dict[str, Any] | None:
    """Run geometry diagnostics for one stage when requested."""
    if not stage_config.get("run_geometry", False):
        return None
    checkpoint_path = Path(stage_config["checkpoint"])
    if not checkpoint_path.exists():
        if stage_config.get("skip_if_missing", False):
            return None
        raise FileNotFoundError(f"Geometry checkpoint not found for {stage_name}: {checkpoint_path}")
    geometry_dir = output_root / "geometry" / stage_name
    summary_path = geometry_dir / "geometry_summary.json"
    metrics_path = geometry_dir / "geometry_metrics.jsonl"
    if skip_existing and summary_path.exists() and metrics_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return {
            "summary": summary,
            "summary_path": str(summary_path),
            "metrics_path": str(metrics_path),
        }
    return run_geometry_evaluation(
        checkpoint_path,
        config_path=stage_config.get("config"),
        processed_dir=suite_config["data"]["processed_dir"],
        splits_dir=suite_config["data"]["splits_dir"],
        split=suite_config["evaluation"].get("split", "val"),
        limit_pieces=limit_pieces,
        device=device,
        output_dir=geometry_dir,
    )


def _evaluate_stage(
    stage_name: str,
    *,
    samples_dir: Path,
    reference_dir: Path,
    suite_config: dict[str, Any],
    output_root: Path,
    skip_existing: bool,
) -> dict[str, Any]:
    """Evaluate one stage and reuse summaries when available."""
    evaluation_dir = output_root / "evaluations" / stage_name
    summary_path = evaluation_dir / "summary.json"
    metrics_path = evaluation_dir / "metrics.jsonl"
    if skip_existing and summary_path.exists() and metrics_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return {
            "summary": summary,
            "summary_path": str(summary_path),
            "metrics_path": str(metrics_path),
        }
    return evaluate_directory(
        samples_dir,
        output_dir=evaluation_dir,
        reference_dir=reference_dir,
        duration_bins=suite_config["tokenization"]["duration_bins"],
        velocity_bins=suite_config["tokenization"]["velocity_bins"],
        bar_position_bins=suite_config["tokenization"].get("bar_position_bins", 16),
    )


def _format_metric(value: float | None) -> str:
    """Render a scalar metric for markdown tables."""
    if value is None:
        return "-"
    return f"{value:.4f}"


def _lookup_metric(stage_payload: dict[str, Any], metric_name: str) -> float | None:
    """Look up a metric from either sample evaluation or geometry diagnostics."""
    if stage_payload is None:
        return None
    evaluation = stage_payload.get("evaluation", {})
    geometry = stage_payload.get("geometry") or {}
    if METRIC_SUMMARY_KEYS[metric_name] in evaluation.get("summary", {}):
        return evaluation["summary"].get(METRIC_SUMMARY_KEYS[metric_name])
    if METRIC_SUMMARY_KEYS[metric_name] in geometry.get("summary", {}):
        return geometry["summary"].get(METRIC_SUMMARY_KEYS[metric_name])
    return None


def _render_markdown_report(report: dict[str, Any], *, metrics: list[str]) -> str:
    """Render a compact markdown report for the ablation suite."""
    lines = [
        "# Phase 6 Ablation Report",
        "",
        f"Pieces: {report['piece_count']}",
        "",
        "## Summary",
        "",
        "| Stage | " + " | ".join(metrics) + " |",
        "| --- | " + " | ".join("---" for _ in metrics) + " |",
    ]
    reference_summary = report["reference"]["summary"]
    lines.append(
        "| reference | "
        + " | ".join(
            _format_metric(reference_summary.get(METRIC_SUMMARY_KEYS[metric]))
            for metric in metrics
        )
        + " |"
    )
    for stage_name, stage_payload in report["stages"].items():
        lines.append(
            f"| {stage_name} | "
            + " | ".join(
                _format_metric(_lookup_metric(stage_payload, metric))
                for metric in metrics
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def run_ablation_suite(
    config: dict[str, Any],
    *,
    config_path: str | Path = "configs/full_model.yaml",
    output_dir: str | Path | None = None,
    limit_pieces_override: int | None = None,
    prompt_events_override: int | None = None,
    generate_events_override: int | None = None,
    device: str = "auto",
    seed_override: int | None = None,
    skip_existing: bool = False,
) -> dict[str, Any]:
    """Run the Phase 6 evaluation matrix and package the results."""
    evaluation_config = config["evaluation"]
    stages_config = config["stages"]
    output_root = Path(output_dir or evaluation_config.get("output_dir", "outputs/reports/phase6"))
    output_root.mkdir(parents=True, exist_ok=True)

    limit_pieces = limit_pieces_override or evaluation_config.get("sample_count", 8)
    prompt_events = prompt_events_override or evaluation_config.get("prompt_events", 64)
    generate_events = generate_events_override or evaluation_config.get("generate_events", 96)
    seed = seed_override or config.get("seed", 42)

    stage_results: dict[str, Any] = {}
    skipped_stages: dict[str, str] = {}
    reference_dir = output_root / "reference"
    reference_manifest = None
    reference_evaluation = None
    for stage_name, stage_config in stages_config.items():
        manifest = _run_generation_stage(
            stage_name,
            stage_config,
            suite_config=config,
            output_root=output_root,
            limit_pieces=limit_pieces,
            prompt_events=prompt_events,
            generate_events=generate_events,
            device=device,
            seed=seed,
            skip_existing=skip_existing,
        )
        if manifest.get("skipped"):
            skipped_stages[stage_name] = manifest["reason"]
            continue
        stage_samples_dir = output_root / "samples" / stage_name
        if reference_manifest is None:
            reference_manifest = build_matched_reference_set(
                manifest["manifest_path"],
                processed_dir=config["data"]["processed_dir"],
                output_dir=reference_dir,
            )
            reference_evaluation = evaluate_directory(
                reference_dir,
                output_dir=reference_dir,
                duration_bins=config["tokenization"]["duration_bins"],
                velocity_bins=config["tokenization"]["velocity_bins"],
            )
        evaluation = _evaluate_stage(
            stage_name,
            samples_dir=stage_samples_dir,
            reference_dir=reference_dir,
            suite_config=config,
            output_root=output_root,
            skip_existing=skip_existing,
        )
        geometry = _run_geometry_stage(
            stage_name,
            stage_config,
            suite_config=config,
            output_root=output_root,
            limit_pieces=limit_pieces,
            device=device,
            skip_existing=skip_existing,
        )
        stage_results[stage_name] = {
            "manifest": manifest,
            "samples_dir": str(stage_samples_dir),
            "evaluation": evaluation,
            "geometry": geometry,
        }

    if reference_manifest is None or reference_evaluation is None:
        raise ValueError("Ablation suite requires at least one stage.")

    report = {
        "config_path": str(config_path),
        "output_dir": str(output_root),
        "piece_count": limit_pieces,
        "prompt_events": prompt_events,
        "generate_events": generate_events,
        "metrics": evaluation_config.get("metrics", []),
        "skipped_stages": skipped_stages,
        "reference": {
            "manifest": reference_manifest,
            "summary": reference_evaluation["summary"],
            "summary_path": reference_evaluation["summary_path"],
            "metrics_path": reference_evaluation["metrics_path"],
        },
        "stages": stage_results,
    }
    _write_json(output_root / "report.json", report)
    (output_root / "report.md").write_text(
        _render_markdown_report(report, metrics=evaluation_config.get("metrics", [])),
        encoding="utf-8",
    )
    (output_root / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result = run_ablation_suite(
        config,
        config_path=args.config,
        output_dir=args.output_dir,
        limit_pieces_override=args.limit_pieces,
        prompt_events_override=args.prompt_events,
        generate_events_override=args.generate_events,
        device=args.device,
        seed_override=args.seed,
        skip_existing=args.skip_existing,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
