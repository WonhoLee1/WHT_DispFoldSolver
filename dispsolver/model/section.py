"""Section definitions for CAE model hierarchy."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Section:
    """Abstract base Section definition."""
    name: str
    material_name: str


@dataclass
class SolidSection(Section):
    """Solid continuum section (2D Plane Strain/Stress or 3D Continuum Solid)."""
    thickness: float = 1.0  # Used for 2D plane stress/strain thickness
    orientation: Optional[np.ndarray] = None


@dataclass
class ShellSection(Section):
    """Homogeneous Shell section definition."""
    thickness: float = 1.0
    num_int_pts: int = 5
    offset: float = 0.0
