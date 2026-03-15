"""Vocabulary specifications for grouped symbolic features."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class VocabularySpec:
    """Compact description of a token family."""

    name: str
    size: int
