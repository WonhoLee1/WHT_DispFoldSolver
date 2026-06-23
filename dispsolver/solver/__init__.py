"""Solver module — Implicit dynamic FEM (Newmark + Newton-Raphson)."""

import jax
jax.config.update("jax_enable_x64", True)

from .dynamic import MaterialAdapter, DynamicSolver

__all__ = [
    "MaterialAdapter",
    "DynamicSolver",
]
