"""Typed schemas for research publications."""

from dataclasses import dataclass
from typing import List


@dataclass
class PaperInfo:
    id: str
    file: str
    title: str
    authors: List[str]
    year: int
    venue: str
    category: str
    key_contribution: str
