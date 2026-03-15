"""Build a poster-alignment report from existing ablation and geometry outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase6-report", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--music-stage", default="tension")
    parser.add_argument("--baseline-stage", default="baseline")
    parser.add_argument("--geometry-summary", default=None)
    parser.add_argument("--euclidean-summary", default=None)
    parser.add_argument("--geometry-stage-label", default="torus_t3")
    parser.add_argument("--euclidean-stage-label", default="euclidean_r3")
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _maybe_load_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        return None
    return _load_json(resolved)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _stage_summary(report: dict[str, Any], stage_name: str) -> dict[str, Any]:
    return report["stages"][stage_name]["evaluation"]["summary"]


def _resolve_stage_name(
    report: dict[str, Any],
    requested: str,
    *,
    fallbacks: tuple[str, ...] = (),
) -> str:
    stage_names = report.get("stages", {})
    if requested in stage_names:
        return requested
    for candidate in fallbacks:
        if candidate in stage_names:
            return candidate
    available = ", ".join(sorted(stage_names))
    raise KeyError(
        f"Stage '{requested}' not found. Available stages: {available or '(none)'}"
    )


def _maybe_resolve_stage_name(
    report: dict[str, Any],
    requested: str,
    *,
    fallbacks: tuple[str, ...] = (),
) -> str | None:
    stage_names = report.get("stages", {})
    if requested in stage_names:
        return requested
    for candidate in fallbacks:
        if candidate in stage_names:
            return candidate
    return None


def _geometry_summary(
    report: dict[str, Any],
    stage_name: str | None,
    fallback_path: str | Path | None,
) -> dict[str, Any] | None:
    if stage_name is not None and stage_name in report.get("stages", {}):
        stage_geometry = report["stages"][stage_name].get("geometry")
        if stage_geometry is not None:
            return stage_geometry["summary"]
    return _maybe_load_json(fallback_path)


def _percent_reduction(baseline_value: float | None, candidate_value: float | None) -> float | None:
    if baseline_value is None or candidate_value is None or baseline_value == 0:
        return None
    return ((baseline_value - candidate_value) / baseline_value) * 100.0


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _claim_status(
    value: float | None,
    *,
    target: float | None = None,
    positive_is_good: bool = True,
    threshold: float = 0.0,
) -> str:
    if value is None:
        return "missing_data"
    direction_value = value if positive_is_good else -value
    if target is not None:
        if direction_value >= target:
            return "supported"
        if direction_value > threshold:
            return "partial"
        return "not_supported"
    if direction_value > threshold:
        return "supported"
    return "not_supported"


def _format_value(value: float | None, *, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def build_poster_alignment_report(
    phase6_report_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    music_stage: str = "tension",
    baseline_stage: str = "baseline",
    geometry_summary_path: str | Path | None = None,
    euclidean_summary_path: str | Path | None = None,
    geometry_stage_label: str = "torus_t3",
    euclidean_stage_label: str = "euclidean_r3",
) -> dict[str, Any]:
    report = _load_json(phase6_report_path)
    output_root = Path(output_dir or "outputs/reports/poster_alignment")
    output_root.mkdir(parents=True, exist_ok=True)

    music_stage = _resolve_stage_name(
        report,
        music_stage,
        fallbacks=("tension_t3", "tension", "conductor", "torus_t3", "torus"),
    )
    baseline_stage = _resolve_stage_name(report, baseline_stage)
    resolved_geometry_stage_label = _maybe_resolve_stage_name(
        report,
        geometry_stage_label,
        fallbacks=("torus_t3", "tension_t3", "torus"),
    )
    resolved_euclidean_stage_label = _maybe_resolve_stage_name(
        report,
        euclidean_stage_label,
        fallbacks=("euclidean", "euclidean_r3"),
    )

    reference = report["reference"]["summary"]
    baseline = _stage_summary(report, baseline_stage)
    music = _stage_summary(report, music_stage)
    torus_geometry = _geometry_summary(
        report,
        resolved_geometry_stage_label,
        geometry_summary_path,
    )
    euclidean_geometry = _geometry_summary(
        report,
        resolved_euclidean_stage_label,
        euclidean_summary_path,
    )

    jump_reduction = _percent_reduction(
        baseline.get("mean_pitch_jump"),
        music.get("mean_pitch_jump"),
    )
    tonal_gain = _percent_reduction(
        baseline.get("mean_pitch_class_divergence"),
        music.get("mean_pitch_class_divergence"),
    )
    stress_reduction = None
    collapse_reduction = None
    if torus_geometry is not None and euclidean_geometry is not None:
        stress_reduction = _percent_reduction(
            euclidean_geometry.get("mean_structural_stress"),
            torus_geometry.get("mean_structural_stress"),
        )
        collapse_reduction = _percent_reduction(
            euclidean_geometry.get("mean_collapse_score"),
            torus_geometry.get("mean_collapse_score"),
        )
    persistence_ratio = _ratio(
        music.get("mean_max_persistence"),
        baseline.get("mean_max_persistence"),
    )

    criteria = [
        {
            "criterion": "Structural Coherence",
            "metric": "mean_recurrence_similarity",
            "better": "higher",
            "reference": reference.get("mean_recurrence_similarity"),
            "baseline": baseline.get("mean_recurrence_similarity"),
            music_stage: music.get("mean_recurrence_similarity"),
        },
        {
            "criterion": "Melodic Quality",
            "metric": "mean_pitch_jump",
            "better": "lower",
            "reference": reference.get("mean_pitch_jump"),
            "baseline": baseline.get("mean_pitch_jump"),
            music_stage: music.get("mean_pitch_jump"),
        },
        {
            "criterion": "Tonal Fidelity",
            "metric": "mean_pitch_class_divergence",
            "better": "lower",
            "reference": reference.get("mean_pitch_class_divergence"),
            "baseline": baseline.get("mean_pitch_class_divergence"),
            music_stage: music.get("mean_pitch_class_divergence"),
        },
        {
            "criterion": "Human Plausibility",
            "metric": "mean_large_span_rate",
            "better": "lower",
            "reference": reference.get("mean_large_span_rate"),
            "baseline": baseline.get("mean_large_span_rate"),
            music_stage: music.get("mean_large_span_rate"),
        },
    ]
    if torus_geometry is not None and euclidean_geometry is not None:
        criteria.append(
            {
                "criterion": "Geometry Preservation",
                "metric": "mean_structural_stress",
                "better": "lower",
                (resolved_euclidean_stage_label or euclidean_stage_label): euclidean_geometry.get(
                    "mean_structural_stress"
                ),
                (resolved_geometry_stage_label or geometry_stage_label): torus_geometry.get(
                    "mean_structural_stress"
                ),
            }
        )
        criteria.append(
            {
                "criterion": "Latent Collapse",
                "metric": "mean_collapse_score",
                "better": "lower",
                (resolved_euclidean_stage_label or euclidean_stage_label): euclidean_geometry.get(
                    "mean_collapse_score"
                ),
                (resolved_geometry_stage_label or geometry_stage_label): torus_geometry.get(
                    "mean_collapse_score"
                ),
            }
        )

    poster_claims = [
        {
            "claim": "Erratic melodic jumps reduced by 68%",
            "current_value": jump_reduction,
            "target_value": 68.0,
            "unit": "percent",
            "status": _claim_status(jump_reduction, target=68.0),
        },
        {
            "claim": "Tonal distribution 85% closer to human-composed music",
            "current_value": tonal_gain,
            "target_value": 85.0,
            "unit": "percent",
            "status": _claim_status(tonal_gain, target=85.0),
        },
        {
            "claim": "Structural stress reduced by 73% versus Euclidean mapping",
            "current_value": stress_reduction,
            "target_value": 73.0,
            "unit": "percent",
            "status": _claim_status(stress_reduction, target=73.0),
        },
        {
            "claim": "Persistent patterns increased 7x over the baseline",
            "current_value": persistence_ratio,
            "target_value": 7.0,
            "unit": "ratio",
            "status": _claim_status(persistence_ratio, target=7.0, threshold=1.0),
        },
        {
            "claim": "Unconstrained generation collapses to a low-dimensional signature",
            "current_value": collapse_reduction,
            "target_value": None,
            "unit": "percent",
            "status": _claim_status(collapse_reduction),
        },
    ]

    matched_claims = [claim["claim"] for claim in poster_claims if claim["status"] == "supported"]
    partial_claims = [claim["claim"] for claim in poster_claims if claim["status"] == "partial"]
    unsupported_claims = [
        claim["claim"]
        for claim in poster_claims
        if claim["status"] not in {"supported", "partial"}
    ]
    summary = {
        "poster_match_score": len(matched_claims),
        "poster_partial_score": len(partial_claims),
        "matched_claims": matched_claims,
        "partial_claims": partial_claims,
        "unsupported_claims": unsupported_claims,
    }

    payload = {
        "phase6_report_path": str(phase6_report_path),
        "music_stage": music_stage,
        "baseline_stage": baseline_stage,
        "geometry_stage_label": resolved_geometry_stage_label or geometry_stage_label,
        "euclidean_stage_label": resolved_euclidean_stage_label or euclidean_stage_label,
        "criteria": criteria,
        "poster_claims": poster_claims,
        "summary": summary,
        "reference_summary": reference,
        "baseline_summary": baseline,
        "music_summary": music,
        "geometry_summary": torus_geometry,
        "euclidean_summary": euclidean_geometry,
    }
    _write_json(output_root / "report.json", payload)
    (output_root / "report.md").write_text(
        render_poster_alignment_markdown(payload),
        encoding="utf-8",
    )
    return payload


def render_poster_alignment_markdown(payload: dict[str, Any]) -> str:
    music_stage = payload["music_stage"]
    geometry_stage_label = payload["geometry_stage_label"]
    euclidean_stage_label = payload["euclidean_stage_label"]
    lines = [
        "# Poster Alignment Report",
        "",
        "## Criteria",
        "",
        "| Criterion | Metric | Reference | Baseline | "
        f"{music_stage} | {euclidean_stage_label} | {geometry_stage_label} | Better |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["criteria"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["criterion"],
                    row["metric"],
                    _format_value(row.get("reference")),
                    _format_value(row.get("baseline")),
                    _format_value(row.get(music_stage)),
                    _format_value(row.get(euclidean_stage_label)),
                    _format_value(row.get(geometry_stage_label)),
                    row["better"],
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Poster Claims",
            "",
            "| Claim | Current | Target | Status |",
            "| --- | --- | --- | --- |",
        ]
    )
    for claim in payload["poster_claims"]:
        target_text = "-" if claim["target_value"] is None else _format_value(claim["target_value"], digits=1)
        current_text = _format_value(claim["current_value"], digits=2)
        if claim["unit"] == "percent" and current_text != "-":
            current_text = f"{current_text}%"
        elif claim["unit"] == "ratio" and current_text != "-":
            current_text = f"{current_text}x"
        if claim["target_value"] is not None and claim["unit"] == "percent":
            target_text = f"{target_text}%"
        elif claim["target_value"] is not None and claim["unit"] == "ratio":
            target_text = f"{target_text}x"
        lines.append(
            f"| {claim['claim']} | {current_text} | {target_text} | {claim['status']} |"
        )

    summary = payload["summary"]
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- Supported claims: {summary['poster_match_score']}",
            f"- Partial claims: {summary['poster_partial_score']}",
            f"- Unsupported claims: {len(summary['unsupported_claims'])}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    result = build_poster_alignment_report(
        args.phase6_report,
        output_dir=args.output_dir,
        music_stage=args.music_stage,
        baseline_stage=args.baseline_stage,
        geometry_summary_path=args.geometry_summary,
        euclidean_summary_path=args.euclidean_summary,
        geometry_stage_label=args.geometry_stage_label,
        euclidean_stage_label=args.euclidean_stage_label,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
