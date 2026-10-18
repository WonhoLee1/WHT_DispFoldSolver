from __future__ import annotations

from dispsolver.io.keyword_registry import GLOBAL_KEYWORD_REGISTRY
from dispsolver.io.keywords import interaction, loads, material, mesh, p0_2, scope, sections, steps, solution, springs

_SUPPORT_OVERRIDES = {
    "MPC": "STUB",
    "COUPLING": "STUB",
    "KINEMATIC": "STUB",
    "DISTRIBUTING": "STUB",
    "DLOAD": "PARTIAL",
    "ORIENTATION": "PARTIAL",
    "EQUATION": "FULL",
    "TRANSFORM": "FULL",
}

for _kw, _sup in _SUPPORT_OVERRIDES.items():
    _spec = GLOBAL_KEYWORD_REGISTRY.get_handler(_kw)
    if _spec is not None:
        _spec.support = _sup
del _kw, _sup, _spec

__all__ = ["interaction", "loads", "material", "mesh", "p0_2", "scope", "sections", "steps", "solution", "springs"]

