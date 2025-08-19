# file: eval/__init__.py
"""
Evaluation helpers: metrics and quick plots.
"""
from .metrics import compute_basic_metrics
from .plots import quick_diagnostics

__all__ = ["compute_basic_metrics", "quick_diagnostics"]
