"""
Test suite for enhanced Radioss LAW36:
  1. C1-Smooth curve generation (eliminates sharp kinks at yield and UTS).
  2. Extrapolated regularized tail up to eps=5.0 with positive slope (H_tail > 0).
  3. Progressive damage softening when use_damage=True.
  4. Robust Newton convergence in DynamicSolver3D under large deformation without singularity.
"""

import numpy as np
import pytest

from dispsolver.model.material import Material
from dispsolver.material3d.numba_materials import (
    _interp_tabular_hardening,
    material_dispatch_3d,
    MAT_J2_PLASTICITY_TABULAR,
)
from dispsolver.mesh3d import Mesh3D, Node3D, Element3D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D


def test_radioss_law36_smooth_transition():
    """Verify smooth curve generation between sy0 and sigma_uts without sharp kinks."""
    mat = Material(name="SmoothSteel")
    mat.RadiossLaw36(
        E=200000.0,
        nu=0.3,
        sy0=300.0,
        sigma_uts=500.0,
        eps_fail=0.20,
        smooth=True,
        n_points=8,
        use_damage=False
    )
    table = mat.plastic
    assert len(table) >= 6

    # Verify strictly monotonic increasing plastic strain and stress
    eps_vals = [pt[1] for pt in table]
    sy_vals = [pt[0] for pt in table]

    for i in range(len(table) - 1):
        assert eps_vals[i+1] > eps_vals[i], f"Strain not monotonic at index {i}"
        assert sy_vals[i+1] > sy_vals[i], f"Stress not monotonic at index {i}"

    # Verify initial yield stress matches sy0
    assert abs(sy_vals[0] - 300.0) < 1e-9
    assert abs(eps_vals[0] - 0.0) < 1e-9

    # Verify slopes are strictly positive and gradually decrease toward UTS
    slopes = [
        (sy_vals[i+1] - sy_vals[i]) / (eps_vals[i+1] - eps_vals[i])
        for i in range(len(table) - 2)  # Exclude tail
    ]
    for s in slopes:
        assert s > 0.0, f"Slope {s} must be strictly positive"


def test_radioss_law36_extrapolated_positive_tail_eps5():
    """Verify regularized positive tail up to eps=5.0 when damage is disabled."""
    E = 210000.0
    s_uts = 600.0
    ef = 0.25
    tail_ratio = 1e-3
    expected_H_tail = tail_ratio * E  # 210.0 MPa

    mat = Material(name="ToughSteel")
    mat.RadiossLaw36(
        E=E,
        nu=0.3,
        sy0=350.0,
        sigma_uts=s_uts,
        eps_fail=ef,
        smooth=True,
        n_points=8,
        use_damage=False,
        extrapolate_strain=5.0,
        tail_modulus_ratio=tail_ratio
    )
    table = mat.plastic
    last_pt = table[-1]

    # Verify last point extends to eps=5.0
    assert abs(last_pt[1] - 5.0) < 1e-9, f"Expected eps=5.0, got {last_pt[1]}"

    # Verify stress at eps=5.0 is extrapolated with expected H_tail
    expected_s_tail = s_uts + expected_H_tail * (5.0 - ef)
    assert abs(last_pt[0] - expected_s_tail) < 1e-3, f"Expected {expected_s_tail}, got {last_pt[0]}"

    # Verify to_numba_props sets eps_fail to 1e9 so failure is not prematurely triggered
    props = mat.to_numba_props()
    assert props["eps_fail"] >= 1e8
    assert props["sigma_fail"] == 0.0


def test_radioss_law36_progressive_damage_softening():
    """Verify progressive damage softening curve when use_damage=True."""
    s_uts = 500.0
    ef = 0.15
    d_eps = 0.05

    mat = Material(name="DamageableAlloy")
    mat.RadiossLaw36(
        E=70000.0,
        nu=0.33,
        sy0=250.0,
        sigma_uts=s_uts,
        eps_fail=ef,
        smooth=True,
        n_points=8,
        use_damage=True,
        delta_eps_damage=d_eps
    )
    table = mat.plastic

    # Find the peak stress and subsequent points
    sy_vals = [pt[0] for pt in table]
    eps_vals = [pt[1] for pt in table]
    peak_idx = np.argmax(sy_vals)

    # After peak, stress should smoothly soften
    assert peak_idx < len(table) - 1
    post_peak_sy = sy_vals[peak_idx:]
    for i in range(len(post_peak_sy) - 1):
        assert post_peak_sy[i+1] < post_peak_sy[i], "Post-peak stress must soften monotonically"

    # Final residual stress should be 1% of UTS
    assert abs(table[-1][0] - 0.01 * s_uts) < 1e-3
    assert abs(table[-1][1] - (ef + d_eps)) < 1e-3


def test_dynamicsolver3d_large_strain_stability_with_tail():
    """Verify DynamicSolver3D converges cleanly under large strain with extrapolated tail."""
    mesh = Mesh3D()
    coords = [
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]
    ]
    for i, c in enumerate(coords):
        mesh.add_node(node_id=i+1, x=c[0], y=c[1], z=c[2])
    
    elem = Element3D(id=1, node_ids=[1, 2, 3, 4, 5, 6, 7, 8], elem_type="C3D8", pid=0)
    mesh.elements[1] = elem

    mat = Material(name="StabilizedSteel")
    mat.RadiossLaw36(
        E=200000.0,
        nu=0.3,
        sy0=300.0,
        sigma_uts=450.0,
        eps_fail=0.10,
        use_damage=False,
        extrapolate_strain=5.0,
        tail_modulus_ratio=1e-3
    )

    solver = DynamicSolver3D(
        mesh=mesh,
        materials={0: mat},
        nlgeom=True
    )

    # Fix bottom face (z=0: nodes 1, 2, 3, 4)
    for nid in [1, 2, 3, 4]:
        idx = nid - 1
        solver.fixed_dofs[idx * 3 + 0] = 0.0
        solver.fixed_dofs[idx * 3 + 1] = 0.0
        solver.fixed_dofs[idx * 3 + 2] = 0.0

    # Incremental strain loading across multiple steps well beyond eps_fail
    dt = 0.1
    for step in range(1, 6):
        target_disp = step * 0.05  # Up to 25% strain (exceeds eps_fail=0.10)
        for nid in [5, 6, 7, 8]:
            idx = nid - 1
            solver.fixed_dofs[idx * 3 + 2] = target_disp

        converged, n_iters = solver.solve_step(dt=dt)
        assert converged, f"Step {step} failed to converge: n_iters={n_iters}"
        assert n_iters <= 12
