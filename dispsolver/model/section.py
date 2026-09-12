"""Section definitions for CAE model hierarchy."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np


@dataclass
class SectionControls:
    """Controls for continuum elements (*SECTION CONTROLS in Abaqus).

    Attributes:
        name: Name of the control definition.
        distortion_control: Whether to enable anti-inversion/distortion barrier.
        length_ratio: Critical volume/thickness ratio r (default 0.1, Abaqus standard).
        hourglass_control: Hourglass formulation ('ENHANCED', 'STIFFNESS', 'VISCOUS').
        viscous_damping: Element-level viscous damping coefficient beta (default 0.0).
        anti_inversion_barrier: Whether to activate infinite logarithmic energy barrier as J -> 0+.
        min_det_f: Minimum safe volume ratio for line-search safeguard (default 0.02).
    """
    name: str
    distortion_control: bool = True
    length_ratio: float = 0.1
    hourglass_control: str = "ENHANCED"
    viscous_damping: float = 0.0
    anti_inversion_barrier: bool = True
    min_det_f: float = 0.02

    def to_control_array(self) -> np.ndarray:
        """Convert controls to an 8-float array for Numba element kernels."""
        arr = np.zeros(8, dtype=np.float64)
        arr[0] = 1.0 if self.distortion_control else 0.0
        arr[1] = float(self.length_ratio)
        arr[2] = float(self.viscous_damping)
        arr[3] = 1.0 if self.anti_inversion_barrier else 0.0
        arr[4] = float(self.min_det_f)
        # Hourglass mode: 0 = ENHANCED, 1 = STIFFNESS, 2 = VISCOUS
        hg_map = {"ENHANCED": 0.0, "STIFFNESS": 1.0, "VISCOUS": 2.0}
        arr[5] = hg_map.get(self.hourglass_control.upper(), 0.0)
        return arr


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
    controls: Optional[Union[str, SectionControls]] = None


@dataclass
class ShellSection(Section):
    """Homogeneous Shell section definition."""
    thickness: float = 1.0
    num_int_pts: int = 5
    offset: float = 0.0
