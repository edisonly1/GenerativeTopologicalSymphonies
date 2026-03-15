"""Inference and rendering utilities."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "generate_from_checkpoint": ("generate", "generate_from_checkpoint"),
    "generate_piece_continuation": ("generate", "generate_piece_continuation"),
    "refine_directory": ("refine", "refine_directory"),
    "refine_piece": ("refine", "refine_piece"),
    "render_piece_to_midi": ("render_midi", "render_piece_to_midi"),
    "render_piece_to_midi_bytes": ("render_midi", "render_piece_to_midi_bytes"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """Lazily expose inference helpers without importing CLI modules eagerly."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
