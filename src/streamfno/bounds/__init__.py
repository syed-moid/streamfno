"""Bound instruments for the predictability horizon (T3)."""

from .divergence import DivergenceRecord, bump_perturbation, divergence_rate
from .lecam import (
    PairBound,
    ensemble_outcome_probs,
    kl_gaussian_obs,
    pde_clean_observables,
    pinsker_bound,
    two_point_bound,
)
from .margins import nearest_opposite_margins

__all__ = [
    "DivergenceRecord",
    "PairBound",
    "bump_perturbation",
    "divergence_rate",
    "ensemble_outcome_probs",
    "kl_gaussian_obs",
    "nearest_opposite_margins",
    "pde_clean_observables",
    "pinsker_bound",
    "two_point_bound",
]
