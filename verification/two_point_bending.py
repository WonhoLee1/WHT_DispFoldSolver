"""
two_point_bending.py
====================
Analytical closed-form Elastica solutions and benchmark verification for
Two-Point Bending of Thin Glass Substrates.

Theoretical Reference:
----------------------
Suresh T. Gulati, Jamie Westbrook, Stephen Carley, Hemanth Vepakomma,
and Toshihiko Ono (Corning Incorporated),
"45.2: Two Point Bending of Thin Glass Substrate",
SID 2004 Digest, ISSN 0004-0966X/04/3502-0001, pp. 1-2.

Key Analytical Results:
-----------------------
1. Peak Bending Stress (Apex, theta = 90 deg):
   sigma_max = 1.19814 * [E' * t / (D - t)] * sqrt(cos(psi))
   where E' = E / (1 - nu^2) for plane strain (2D FEM).

2. Variation of Bend Stress along Elastica:
   sigma_bend(theta) = sigma_max * sqrt(sin(theta))

3. Elastica Coordinates (Closed-form elliptic integral solution):
   x(theta) = ((D - t) / 1.19814) * sqrt(sin(theta))
   y(theta) = ((D - t) / 2.39628) * int_0^theta sqrt(sin(phi)) dphi
"""

from __future__ import annotations

import numpy as np
import scipy.integrate as integrate
from typing import Dict, List, Optional, Tuple

from dispsolver.mesh.mesh import Element, Mesh, Node


class GulatiTwoPointBendingTheory:
    """Closed-form analytical solution for two-point bending of thin glass plates."""

    K_GULATI: float = 1.1981402347355916  # int_0^(pi/2) sqrt(sin(phi)) dphi

    def __init__(
        self,
        E: float = 72300.0,      # MPa (72.3 GPa, Corning flat glass)
        nu: float = 0.22,        # Poisson's ratio for Corning glass
        t: float = 0.40,         # mm (substrate thickness)
        D: float = 20.0,         # mm (parallel plate spacing)
        plane_strain: bool = True,
    ):
        """Initialize theoretical problem parameters.

        Parameters
        ----------
        E : float
            Young's modulus in MPa (default: 72,300 MPa).
        nu : float
            Poisson's ratio (default: 0.22).
        t : float
            Substrate thickness in mm (default: 0.4 mm).
        D : float
            Distance between the parallel plates in mm (default: 20.0 mm).
        plane_strain : bool
            Whether to use plane strain effective modulus E / (1 - nu^2).
        """
        self.E = float(E)
        self.nu = float(nu)
        self.t = float(t)
        self.D = float(D)
        self.plane_strain = bool(plane_strain)

    @property
    def E_eff(self) -> float:
        """Effective modulus in MPa."""
        if self.plane_strain:
            return self.E / (1.0 - self.nu ** 2)
        return self.E

    @property
    def D_eff(self) -> float:
        """Effective plate spacing between neutral axes (D - t) in mm."""
        return max(self.D - self.t, 1e-12)

    def peak_stress(self, psi_deg: float = 0.0) -> float:
        """Compute maximum bending stress at mid-length (apex) in MPa.

        Parameters
        ----------
        psi_deg : float
            Contact angle with parallel plates in degrees (0 for long plates).

        Returns
        -------
        float
            Maximum bending stress sigma_max in MPa.
        """
        psi_rad = np.radians(psi_deg)
        stress = self.K_GULATI * (self.E_eff * self.t / self.D_eff)
        if psi_deg != 0.0:
            stress *= np.sqrt(np.cos(psi_rad))
        return float(stress)

    def stress_distribution(self, thetas_rad: np.ndarray, psi_deg: float = 0.0) -> np.ndarray:
        """Compute bending stress distribution along the elastica profile.

        Parameters
        ----------
        thetas_rad : np.ndarray
            Angle array in radians from 0 to pi.
        psi_deg : float
            Contact angle in degrees.

        Returns
        -------
        np.ndarray
            Bending stress at each angle in MPa.
        """
        s_max = self.peak_stress(psi_deg=psi_deg)
        return s_max * np.sqrt(np.maximum(np.sin(thetas_rad), 0.0))

    def exact_profile(self, n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Compute exact (x, y) coordinates of the full symmetric elastica loop.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (xs, ys) coordinate arrays in mm.
            Apex is located at (x_max, 0.0).
            The two ends touch the parallel plates at y = -D_eff/2 and y = +D_eff/2.
        """
        # Half loop from theta = 0 (bottom plate) to pi/2 (apex)
        th_half = np.linspace(0.0, np.pi / 2.0, n_points // 2)
        xs_half = (self.D_eff / self.K_GULATI) * np.sqrt(np.maximum(np.sin(th_half), 0.0))

        ys_half = np.zeros_like(th_half)
        for i, th in enumerate(th_half):
            if th > 0:
                val, _ = integrate.quad(lambda p: np.sqrt(np.sin(p)), 0, th)
                ys_half[i] = (self.D_eff / (2.0 * self.K_GULATI)) * val

        # Full symmetric loop: bottom to top
        # Bottom half: y from -D_eff/2 to 0
        xs_bot = xs_half
        ys_bot = ys_half - 0.5 * self.D_eff

        # Top half: y from 0 to +D_eff/2
        xs_top = xs_half[::-1]
        ys_top = 0.5 * self.D_eff - ys_half[::-1]

        xs_full = np.concatenate([xs_bot, xs_top[1:]])
        ys_full = np.concatenate([ys_bot, ys_top[1:]])
        return xs_full, ys_full

    def loop_arc_length(self) -> float:
        """Total arc length of the free bent loop between parallel plates in mm."""
        # int_0^(pi/2) dphi / sqrt(sin(phi)) = sqrt(2) * K(1/sqrt(2)) ~= 2.622057554
        k_ellip = 2.6220575542921196
        return float(2.0 * k_ellip * self.D_eff / (2.0 * self.K_GULATI))


def build_two_point_bending_mesh(
    L: float = 100.0,
    t: float = 0.4,
    nx: int = 100,
    ny: int = 2,
    pid: int = 1,
) -> Mesh:
    """Build a 2D plane-strain beam mesh for two-point bending simulation.

    Parameters
    ----------
    L : float
        Length of the beam in mm (default: 100 mm).
    t : float
        Thickness of the beam in mm (default: 0.4 mm).
    nx : int
        Number of elements along length.
    ny : int
        Number of elements through thickness.
    pid : int
        Property ID.

    Returns
    -------
    Mesh
        Structured 2D FEM Mesh.
    """
    mesh = Mesh()
    xs = np.linspace(-L / 2.0, L / 2.0, nx + 1)
    ys = np.linspace(-t / 2.0, t / 2.0, ny + 1)

    node_grid = np.zeros((nx + 1, ny + 1), dtype=int)
    nid = 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            mesh.add_node(nid, float(xs[i]), float(ys[j]))
            node_grid[i, j] = nid
            nid += 1

    eid = 1
    for j in range(ny):
        for i in range(nx):
            n1 = int(node_grid[i, j])
            n2 = int(node_grid[i + 1, j])
            n3 = int(node_grid[i + 1, j + 1])
            n4 = int(node_grid[i, j + 1])
            mesh.elements[eid] = Element(eid, [n1, n2, n3, n4], "QUAD4", pid=pid)
            eid += 1

    return mesh
