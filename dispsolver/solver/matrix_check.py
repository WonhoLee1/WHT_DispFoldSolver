"""Rigid-body-mode check of an assembled stiffness matrix (Abaqus ``*MATRIX CHECK``).

A free-floating body stores no strain energy under a rigid-body motion, so its stiffness matrix
(before any boundary condition or constraint is applied) must annihilate the translations and the
infinitesimal rotations: ``K @ r = 0``.  A nonzero product means a spurious stiffness path (a
constraint left in the matrix, a kernel whose strain measure is not rotation invariant, a tangent
assembled with the wrong sign).  This is the global counterpart of the element rank tests.

DOF layout is ``dof = dim * node_index + axis`` (the layout of the 2D and 3D solvers).

Scope: valid for the tangent at an unstressed reference state (``u = 0`` and no prestress).  A
tangent at a stressed or deformed state legitimately carries geometric stiffness: the rotation
modes then leave a residual equal to the rotated internal force, not zero (translations still
pass).  Pass ``rotations=False`` there.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class ModeResult:
    name: str
    relative_residual: float      # |K r| / (|K|_F |r|)
    passed: bool


@dataclass
class MatrixCheckReport:
    tol: float
    modes: List[ModeResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(m.passed for m in self.modes)

    def __str__(self) -> str:
        lines = [f"*MATRIX CHECK (tol {self.tol:g}): {'PASS' if self.passed else 'FAIL'}"]
        for m in self.modes:
            lines.append(f"  {m.name:<14s} |K r|/(|K||r|) = {m.relative_residual:.3e}  {'ok' if m.passed else 'NOT NULL'}")
        return "\n".join(lines)


def rigid_body_modes(coords: np.ndarray, rotations: bool = True):
    """(names, modes) with one column per mode, length ``dim * n_nodes``.

    Translations are unit shifts; rotations are infinitesimal rotations about the centroid
    (1 in 2D, 3 in 3D). Columns are not normalised."""
    X = np.asarray(coords, dtype=np.float64)
    n, dim = X.shape
    if dim not in (2, 3):
        raise ValueError("coords must be (n_nodes, 2) or (n_nodes, 3)")
    c = X.mean(axis=0)
    names, cols = [], []
    for a in range(dim):
        v = np.zeros((n, dim))
        v[:, a] = 1.0
        names.append(("UX", "UY", "UZ")[a])
        cols.append(v.ravel())
    if rotations:
        d = X - c
        if dim == 2:
            v = np.stack([-d[:, 1], d[:, 0]], axis=1)
            names.append("ROT")
            cols.append(v.ravel())
        else:
            for a, nm in enumerate(("ROTX", "ROTY", "ROTZ")):
                axis = np.zeros(3)
                axis[a] = 1.0
                names.append(nm)
                cols.append(np.cross(axis, d).ravel())
    return names, np.stack(cols, axis=1)


def matrix_check(K, coords: np.ndarray, tol: float = 1e-8, rotations: bool = True) -> MatrixCheckReport:
    """Project ``K`` on the rigid-body modes of ``coords``; report ``|K r| / (|K|_F |r|)`` per mode.

    ``K`` is the assembled matrix BEFORE boundary conditions / constraints (dense or scipy sparse)."""
    names, R = rigid_body_modes(coords, rotations=rotations)
    if K.shape[0] != R.shape[0]:
        raise ValueError(f"K has {K.shape[0]} DOFs but coords give {R.shape[0]}; DOF layout is dim*node+axis")
    if hasattr(K, "tocsr"):
        knorm = float(np.sqrt(K.multiply(K).sum()))
    else:
        knorm = float(np.linalg.norm(K))
    report = MatrixCheckReport(tol=tol)
    for j, nm in enumerate(names):
        r = R[:, j]
        rn = float(np.linalg.norm(r))
        res = float(np.linalg.norm(K @ r)) / max(knorm * rn, 1e-300)
        report.modes.append(ModeResult(nm, res, res <= tol))
    return report
