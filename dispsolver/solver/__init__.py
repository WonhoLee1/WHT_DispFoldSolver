"""Solver module — Implicit dynamic FEM (Newmark + Newton-Raphson)."""

import jax
jax.config.update("jax_enable_x64", True)

from .dynamic import MaterialAdapter, DynamicSolver as LegacyDynamicSolver
from .unified import DynamicSolver
from .time_integrator_presets import resolve_time_integrator, TimeIntegratorConfig, PRESETS

__all__ = [
    "MaterialAdapter",
    "DynamicSolver",
    "LegacyDynamicSolver",
    "resolve_time_integrator",
    "TimeIntegratorConfig",
    "PRESETS",
]

