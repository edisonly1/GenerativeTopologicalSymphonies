"""Preprocessing utilities for symbolic music data."""

from .asap import detect_asap_dataset
from .harmony_extract import annotate_quantized_piece_harmony, extract_harmony
from .midi_parser import parse_midi_bytes, parse_midi_file
from .phrase_segment import segment_phrases
from .quantize import quantize_piece, resolve_steps_per_beat
from .schema import (
    NoteEvent,
    ParsedPiece,
    QuantizedEvent,
    QuantizedPiece,
    TempoChange,
    TimeSignatureChange,
)
from .serialization import (
    load_quantized_piece_json,
    quantized_piece_from_dict,
    quantized_piece_to_dict,
    write_quantized_piece_json,
)


def assign_splits(*args, **kwargs):
    """Lazily import split assignment helpers to keep module CLIs clean."""
    from .prepare_dataset import assign_splits as _assign_splits

    return _assign_splits(*args, **kwargs)


def detect_maestro_official_splits(*args, **kwargs):
    """Lazily import MAESTRO split detection helpers to keep module CLIs clean."""
    from .prepare_dataset import detect_maestro_official_splits as _detect_maestro_official_splits

    return _detect_maestro_official_splits(*args, **kwargs)


def detect_dataset_kind(*args, **kwargs):
    """Lazily import dataset-family detection helpers to keep module CLIs clean."""
    from .prepare_dataset import detect_dataset_kind as _detect_dataset_kind

    return _detect_dataset_kind(*args, **kwargs)


def discover_midi_files(*args, **kwargs):
    """Lazily import MIDI discovery helpers to keep module CLIs clean."""
    from .prepare_dataset import discover_midi_files as _discover_midi_files

    return _discover_midi_files(*args, **kwargs)


def generate_toy_dataset(*args, **kwargs):
    """Lazily import the toy-data generator to avoid package import cycles."""
    from .generate_toy_data import generate_toy_dataset as _generate_toy_dataset

    return _generate_toy_dataset(*args, **kwargs)


def prepare_dataset(*args, **kwargs):
    """Lazily import dataset preparation helpers to keep module CLIs clean."""
    from .prepare_dataset import prepare_dataset as _prepare_dataset

    return _prepare_dataset(*args, **kwargs)

__all__ = [
    "NoteEvent",
    "ParsedPiece",
    "QuantizedEvent",
    "QuantizedPiece",
    "TempoChange",
    "TimeSignatureChange",
    "assign_splits",
    "annotate_quantized_piece_harmony",
    "detect_asap_dataset",
    "detect_dataset_kind",
    "detect_maestro_official_splits",
    "discover_midi_files",
    "extract_harmony",
    "generate_toy_dataset",
    "load_quantized_piece_json",
    "parse_midi_bytes",
    "parse_midi_file",
    "prepare_dataset",
    "quantize_piece",
    "quantized_piece_from_dict",
    "quantized_piece_to_dict",
    "resolve_steps_per_beat",
    "segment_phrases",
    "write_quantized_piece_json",
]
