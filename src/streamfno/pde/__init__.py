"""Reflected Fokker-Planck solver on [0,1] with regulated boundary at x=1."""

from .solver import FPResult, solve_fp, stationary_exponential

__all__ = ["FPResult", "solve_fp", "stationary_exponential"]
