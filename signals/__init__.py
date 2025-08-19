# file: signals/__init__.py
"""
Synthetic signal generators (e.g., OU and regime-switching paths).
"""
from .path import OUPath, OUParams, RegimePath, RegimeParams

__all__ = ["OUPath", "OUParams", "RegimePath", "RegimeParams"]
