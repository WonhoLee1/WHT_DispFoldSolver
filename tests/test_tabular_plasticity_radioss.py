"""
Test suite for Radioss LAW36 / Abaqus Tabular Plasticity with Ductile Damage.
Verifies:
  1. Material.RadiossLaw36 helper sets up piecewise linear table and failure parameters.
  2. Piecewise linear interpolation and hardening modulus _interp_tabular_hardening.
  3. Local Newton return mapping and consistent elastoplastic tangent C_ep.
  4. Progressive softening post eps_fail.
  5. Integration with DynamicSolver3D in a 3D single element tension test.
"""

import numpy as np
import pytest

from dispsolver.material3d.numba_materials import (
    _interp_tabular_hardening,
    material_dispatch_3d,
    MAT_J2_PLASTICITY_TABULAR,
)
from dispsolver.model.material import Material
from dispsolver.mesh3d import Mesh3D, Node3D, Element3D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D


def _call_dispatch(mat_type, props, sdv_prev, E_voigt, dt=0.01):
    F = np.eye(3, dtype=np.float64)
    detF = 1.0
    return material_dispatch_3d(
        mat_type, props, sdv_prev, E_voigt, F, detF, dt
    )


def test_radioss_law36_helper():
    """Verify Material.RadiossLaw36 creates proper tabular data for damage and tail modes."""
    # 1. Damage mode
    mat_dam = Material(name="Steel_LAW36_Damage")
    mat_dam.RadiossLaw36(
        E=210000.0,
        nu=0.3,
        sy0=350.0,
        sigma_uts=600.0,
        eps_fail=0.25,
        density=7.85e-9,
        use_damage=True,
        smooth=False
    )
    assert mat_dam.mat_type == "TABULAR_PLASTIC"
    assert mat_dam.density == 7.85e-9
    assert mat_dam.extra_params["eps_fail"] == 0.25
    assert mat_dam.extra_params["sigma_fail"] == 6.0  # 1% residual stress
    assert len(mat_dam.plastic) >= 3

    # 2. Extrapolated Tail mode (use_damage=False)
    mat_tail = Material(name="Steel_LAW36_Tail")
    mat_tail.RadiossLaw36(
        E=210000.0,
        nu=0.3,
        sy0=350.0,
        sigma_uts=600.0,
        eps_fail=0.25,
        use_damage=False,
        smooth=False
    )
    assert mat_tail.extra_params["eps_fail"] >= 1e8
    assert mat_tail.plastic[-1][1] == 5.0  # Extrapolated to 5.0


def test_tabular_interpolation_kernel():
    """Verify _interp_tabular_hardening at knots and within segments."""
    table_sy = np.array([300.0, 400.0, 500.0], dtype=np.float64)
    table_ep = np.array([0.0, 0.05, 0.20], dtype=np.float64)
    n_pts = 3

    # 1. At knots
    sy0, H0 = _interp_tabular_hardening(0.0, n_pts, table_sy, table_ep)
    assert abs(sy0 - 300.0) < 1e-12
    assert abs(H0 - (400.0 - 300.0) / 0.05) < 1e-12  # H = 2000.0

    sy1, H1 = _interp_tabular_hardening(0.05, n_pts, table_sy, table_ep)
    assert abs(sy1 - 400.0) < 1e-12

    # 2. Midpoint in segment 2
    ep_mid = 0.125
    sy_mid, H_mid = _interp_tabular_hardening(ep_mid, n_pts, table_sy, table_ep)
    expected_H = (500.0 - 400.0) / (0.20 - 0.05)  # 666.666...
    expected_sy = 400.0 + expected_H * (ep_mid - 0.05)
    assert abs(sy_mid - expected_sy) < 1e-10
    assert abs(H_mid - expected_H) < 1e-10

    # 3. Beyond last point (saturation/perfect plasticity)
    sy_ext, H_ext = _interp_tabular_hardening(0.50, n_pts, table_sy, table_ep)
    assert abs(sy_ext - 500.0) < 1e-12
    assert abs(H_ext - 0.0) < 1e-12


def test_tabular_plasticity_return_mapping_and_softening():
    """Verify return mapping, tangent consistency, and damage softening."""
    E = 200000.0
    nu = 0.3
    table_sy = [300.0, 450.0, 500.0]
    table_ep = [0.0, 0.05, 0.15]
    eps_fail = 0.10
    sigma_fail = 5.0

    props = np.zeros(64, dtype=np.float64)
    props[0] = E
    props[1] = nu
    props[2] = 3.0
    props[3] = eps_fail
    props[4] = sigma_fail
    for p in range(3):
        props[5 + 2 * p] = table_sy[p]
        props[6 + 2 * p] = table_ep[p]

    sdv = np.zeros(8, dtype=np.float64)

    # 1. Elastic step: strain below yield (sy0/E = 300 / 200000 = 0.0015)
    E_elastic = np.array([0.001, -0.0003, -0.0003, 0.0, 0.0, 0.0], dtype=np.float64)
    S_el, C_el, sdv_next, err1 = _call_dispatch(
        mat_type=MAT_J2_PLASTICITY_TABULAR,
        props=props,
        sdv_prev=sdv,
        E_voigt=E_elastic,
        dt=0.01
    )
    assert err1 == 0
    assert sdv_next[6] == 0.0  # eq_p == 0
    assert abs(S_el[0] - E * 0.001) < 1.0  # 1D uniaxial approx

    # 2. Plastic loading step: strain well into plastic zone (eps = 0.03)
    E_plastic = np.array([0.03, -0.01, -0.01, 0.0, 0.0, 0.0], dtype=np.float64)
    sdv_cur = sdv_next.copy()
    S_pl, C_pl, sdv_out, err2 = _call_dispatch(
        mat_type=MAT_J2_PLASTICITY_TABULAR,
        props=props,
        sdv_prev=sdv_cur,
        E_voigt=E_plastic,
        dt=0.01
    )
    assert err2 == 0
    assert sdv_out[6] > 0.01  # plastic strain accumulated
    assert sdv_out[7] == 0.0  # not failed yet since eq_p < eps_fail (0.10)
    # Tangent symmetry
    diff_sym = np.max(np.abs(C_pl - C_pl.T))
    assert diff_sym < 1e-10

    # 3. Softening step: strain exceeding eps_fail (eps = 0.15)
    E_fail = np.array([0.15, -0.05, -0.05, 0.0, 0.0, 0.0], dtype=np.float64)
    S_fail, C_fail, sdv_fail_out, err3 = _call_dispatch(
        mat_type=MAT_J2_PLASTICITY_TABULAR,
        props=props,
        sdv_prev=sdv_out,
        E_voigt=E_fail,
        dt=0.01
    )
    assert err3 == 0
    assert sdv_fail_out[7] == 1.0  # is_failed flag set
    # Stress should be softened dramatically toward residual sigma_fail
    assert np.linalg.norm(S_fail) < np.linalg.norm(S_pl) * 0.2


def test_dynamicsolver3d_with_radioss_law36():
    """Verify DynamicSolver3D runs with Radioss LAW36 material."""
    mesh = Mesh3D()
    # 8-node unit cube: [0, 1] x [0, 1] x [0, 1]
    coords = [
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]
    ]
    for i, c in enumerate(coords):
        mesh.add_node(node_id=i+1, x=c[0], y=c[1], z=c[2])
    
    elem = Element3D(id=1, node_ids=[1, 2, 3, 4, 5, 6, 7, 8], elem_type="C3D8", pid=0)
    mesh.elements[1] = elem

    mat = Material(name="LAW36_Mat")
    mat.RadiossLaw36(
        E=200000.0,
        nu=0.3,
        sy0=300.0,
        sigma_uts=450.0,
        eps_fail=0.20,
        density=7.85e-9,
        smooth=False,
        use_damage=False
    )

    solver = DynamicSolver3D(
        mesh=mesh,
        materials={0: mat},
        nlgeom=True
    )

    # Check that elem_props and elem_mat_types are properly assigned
    assert solver.elem_mat_types[0] == MAT_J2_PLASTICITY_TABULAR
    assert solver.elem_props[0, 0] == 200000.0
    assert solver.elem_props[0, 1] == 0.3
    assert solver.elem_props[0, 2] >= 2.0  # At least 2 points (sy0 and tail)

    # Fix bottom face (z=0: nodes 1, 2, 3, 4)
    for nid in [1, 2, 3, 4]:
        idx = nid - 1
        solver.fixed_dofs[idx * 3 + 0] = 0.0
        solver.fixed_dofs[idx * 3 + 1] = 0.0
        solver.fixed_dofs[idx * 3 + 2] = 0.0

    # Pull top face in z-direction (z=1: nodes 5, 6, 7, 8)
    for nid in [5, 6, 7, 8]:
        idx = nid - 1
        solver.fixed_dofs[idx * 3 + 2] = 0.002  # 0.2% strain

    converged, n_iters = solver.solve_step(dt=0.01)
    assert converged, f"Solve failed to converge in {n_iters} iterations"
    assert n_iters <= 10
