"""Evaluation utilities for musical quality and structure."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "CadenceMetrics": ("cadence", "CadenceMetrics"),
    "FluencyMetrics": ("fluency", "FluencyMetrics"),
    "GeometryMetrics": ("geometry", "GeometryMetrics"),
    "PersistenceSummary": ("tda.persistence", "PersistenceSummary"),
    "PlayabilityMetrics": ("playability", "PlayabilityMetrics"),
    "RecurrenceMetrics": ("recurrence", "RecurrenceMetrics"),
    "TonalMetrics": ("tonal", "TonalMetrics"),
    "build_poster_alignment_report": ("poster_report", "build_poster_alignment_report"),
    "build_matched_reference_set": ("matched_reference", "build_matched_reference_set"),
    "evaluate_directory": ("evaluate_samples", "evaluate_directory"),
    "run_geometry_evaluation": ("geometry", "run_geometry_evaluation"),
    "evaluate_piece": ("evaluate_samples", "evaluate_piece"),
    "run_ablation_suite": ("run_ablation_suite", "run_ablation_suite"),
    "run_recurrence_diagnostics": ("diagnose_recurrence", "run_recurrence_diagnostics"),
    "score_cadence_stability": ("cadence", "score_cadence_stability"),
    "score_fluency": ("fluency", "score_fluency"),
    "score_geometry": ("geometry", "score_geometry"),
    "score_playability": ("playability", "score_playability"),
    "score_recurrence": ("recurrence", "score_recurrence"),
    "score_tonal_alignment": ("tonal", "score_tonal_alignment"),
    "slice_quantized_piece": ("matched_reference", "slice_quantized_piece"),
    "summarize_results": ("evaluate_samples", "summarize_results"),
    "render_poster_alignment_markdown": ("poster_report", "render_poster_alignment_markdown"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """Lazily expose evaluation helpers without importing CLI modules eagerly."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    if module_name.startswith("tda."):
        module = import_module(module_name)
    else:
        module = import_module(f".{module_name}", __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
