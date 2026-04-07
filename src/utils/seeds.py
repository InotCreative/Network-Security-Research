"""Deterministic seed management.

A single global seed fans out to per-component seeds so that:
  - Changing the global seed changes everything consistently.
  - Each component has an independent, reproducible random state.
  - Seeds are serialised into every artifact for full auditability.
"""

from __future__ import annotations

import hashlib
import random
from typing import Dict

import numpy as np

_GLOBAL_SEED: int = 42
_REGISTRY: Dict[str, int] = {}


def set_global_seed(seed: int) -> None:
    """Set the global seed and reset the component registry."""
    global _GLOBAL_SEED, _REGISTRY
    _GLOBAL_SEED = int(seed)
    _REGISTRY = {}
    random.seed(_GLOBAL_SEED)
    np.random.seed(_GLOBAL_SEED)


def get_seed(component: str) -> int:
    """Return a deterministic integer seed for *component*.

    The seed is derived from the global seed + component name, so identical
    component names always produce identical seeds across runs with the same
    global seed, and different components never share a seed.
    """
    if component not in _REGISTRY:
        h = hashlib.sha256(f"{_GLOBAL_SEED}::{component}".encode()).digest()
        _REGISTRY[component] = int.from_bytes(h[:4], "big") % (2**31)
    return _REGISTRY[component]


def get_global_seed() -> int:
    return _GLOBAL_SEED


def seed_summary() -> Dict[str, int]:
    """Return a copy of the seed registry for artifact logging."""
    return dict(_REGISTRY)
