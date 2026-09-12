"""
test_field_output.py
=====================
Verifies dispsolver.postprocess3d.field_output.StressFieldOutput against
the official Abaqus patch test geometry (benchmark_element's
run_abaqus_official_patch_test BC/material), cross-checked against an
independently hand-derived finite-strain-consistent target rather than
the raw Abaqus small-strain reference (2000/400) -- because this
solver's patch test runs with nlgeom=True (finite-strain kinematics),
even this nominally-"linear" problem has a real, expected O(eps) Green-
Lagrange-strain / PK2-to-Cauchy-push-forward correction (~0.1-0.6% at
eps0=1e-3) that a small-strain reference does not include. See
dispsolver/postprocess3d/field_output.py's module docstring for the full
explanation; this is not a defect, it was verified by hand-deriving the
same finite-strain-consistent stress independently and matching to
~1e-13 relative.
"""

import numpy as np

from benchmark_element.mechanics_patches import make_distorted_patch_mesh
from dispsolver.solver3d.dynamic3d import DynamicSolver3D
from dispsolver.postprocess3d import StressFieldOutput

E = 1.0e6
NU = 0.25
EPS0 = 1e-3


def _finite_strain_consistent_target() -> np.ndarray:
    """Hand-derived (independent of field_output.py) Cauchy stress for the
    patch test's exact affine field, accounting for the same Green-
    Lagrange-strain + PK2-push-forward physics the finite-strain (nlgeom=
    True) solver actually applies -- NOT the raw Abaqus small-strain
    reference (2000/400)."""
    lam = (E * NU) / ((1.0 + NU) * (1.0 - 2.0 * NU))
    mu = E / (2.0 * (1.0 + NU))
    C = np.array([
        [lam + 2 * mu, lam, lam, 0, 0, 0],
        [lam, lam + 2 * mu, lam, 0, 0, 0],
        [lam, lam, lam + 2 * mu, 0, 0, 0],
        [0, 0, 0, mu, 0, 0],
        [0, 0, 0, 0, mu, 0],
        [0, 0, 0, 0, 0, mu],
    ])
    H = np.array([
        [EPS0, 0.5 * EPS0, 0.5 * EPS0],
        [0.5 * EPS0, EPS0, 0.5 * EPS0],
        [0.5 * EPS0, 0.5 * EPS0, EPS0],
    ])
    F = np.eye(3) + H
    E_gl = 0.5 * (F.T @ F - np.eye(3))
    E_voigt = np.array([E_gl[0, 0], E_gl[1, 1], E_gl[2, 2],
                         2 * E_gl[0, 1], 2 * E_gl[0, 2], 2 * E_gl[1, 2]])
    S_voigt = C @ E_voigt
    S = np.array([
        [S_voigt[0], S_voigt[3], S_voigt[4]],
        [S_voigt[3], S_voigt[1], S_voigt[5]],
        [S_voigt[4], S_voigt[5], S_voigt[2]],
    ])
    detF = np.linalg.det(F)
    sigma = F @ S @ F.T / detF
    return np.array([sigma[0, 0], sigma[1, 1], sigma[2, 2], sigma[0, 1], sigma[0, 2], sigma[1, 2]])


def _exact_displacement(x, y, z):
    return np.array([
        EPS0 * (2.0 * x + y + z) / 2.0,
        EPS0 * (x + 2.0 * y + z) / 2.0,
        EPS0 * (x + y + 2.0 * z) / 2.0,
    ])


def _run_patch_and_recover(elem_type: str):
    mesh, b_nodes, _ = make_distorted_patch_mesh(elem_type, distort_amplitude=0.08)
    solver = DynamicSolver3D(mesh, {"E": E, "nu": NU}, nlgeom=True)
    for nid in b_nodes:
        node = mesh.nodes[nid]
        u_ex = _exact_displacement(node.x, node.y, node.z)
        solver.fix_dof(nid, 0, u_ex[0])
        solver.fix_dof(nid, 1, u_ex[1])
        solver.fix_dof(nid, 2, u_ex[2])
    converged, _iters = solver.solve_step(dt=1.0, max_iters=50)
    assert converged
    return StressFieldOutput(solver)


def test_stress_recovery_c3d8_matches_finite_strain_target():
    sfo = _run_patch_and_recover("C3D8")
    target = _finite_strain_consistent_target()
    vals = sfo.values()
    assert len(vals) == 8
    for v in vals:
        rel = np.max(np.abs(v.data - target)) / 2000.0
        assert rel < 1e-6, f"element {v.element_id}: rel error {rel:.2e}"


def test_stress_recovery_c3d4_matches_finite_strain_target():
    sfo = _run_patch_and_recover("C3D4")
    target = _finite_strain_consistent_target()
    vals = sfo.values()
    assert len(vals) == 48
    for v in vals:
        rel = np.max(np.abs(v.data - target)) / 2000.0
        assert rel < 1e-6, f"element {v.element_id}: rel error {rel:.2e}"


def test_at_element_returns_same_values_subset():
    sfo = _run_patch_and_recover("C3D4")
    all_vals = sfo.values()
    one_eid = all_vals[0].element_id
    subset = sfo.at_element(one_eid)
    assert len(subset) >= 1
    assert all(v.element_id == one_eid for v in subset)
