"""Step, Boundary Condition, and Load definitions for CAE model hierarchy."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Sequence, Any


@dataclass
class Amplitude:
    """Time-dependent amplitude curve for BCs and Loads."""
    name: str
    data: List[Tuple[float, float]] = field(default_factory=list)  # [(time, value), ...]
    smooth: bool = True

    def evaluate(self, t: float) -> float:
        """Evaluate amplitude value at time t."""
        if not self.data:
            return 1.0
        if self.smooth:
            # Smoothstep interpolation between (0, 0) and (1, 1) by default
            t_clamped = max(0.0, min(1.0, float(t)))
            return 3.0 * (t_clamped ** 2) - 2.0 * (t_clamped ** 3)
        # Linear interpolation
        times = [pt[0] for pt in self.data]
        values = [pt[1] for pt in self.data]
        return float(np.interp(t, times, values))


@dataclass
class DisplacementBC:
    """Prescribed displacement boundary condition."""
    name: str
    region: str  # Qualified Set name (e.g. "PART_INST.NSET_NAME")
    u1: Optional[float] = None
    u2: Optional[float] = None
    u3: Optional[float] = None
    ur1: Optional[float] = None
    ur2: Optional[float] = None
    ur3: Optional[float] = None
    amplitude: Optional[str] = None


@dataclass
class ConcentratedLoad:
    """Concentrated nodal force load."""
    name: str
    region: str  # Qualified NodeSet name
    f1: float = 0.0
    f2: float = 0.0
    f3: float = 0.0
    amplitude: Optional[str] = None


@dataclass
class Step:
    """Analysis Step definition (Static, Dynamic, etc.)."""
    name: str
    procedure: str = "STATIC"  # "STATIC" or "DYNAMIC"
    time_period: float = 1.0
    nlgeom: bool = True
    dt_init: float = 0.02
    dt_min: float = 1e-5
    dt_max: float = 0.05
    boundary_conditions: Dict[str, DisplacementBC] = field(default_factory=dict)
    loads: Dict[str, Any] = field(default_factory=dict)

    def add_boundary_condition(self, bc: DisplacementBC) -> None:
        self.boundary_conditions[bc.name] = bc

    def add_load(self, load: Any) -> None:
        self.loads[load.name] = load
