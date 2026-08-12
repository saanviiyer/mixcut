"""mixcut - automatically shorten a song for DJ mixes by cutting the second
verse + chorus and splicing the ends together with a beat-aligned, equal-power
crossfade. Detection is approximate: always review/preview before export."""

from .analysis import analyze, Section, Analysis
from .render import render_cut, CutResult

__version__ = "0.1.0"

__all__ = [
    "analyze",
    "Section",
    "Analysis",
    "render_cut",
    "CutResult",
    "__version__",
]
