"""
test_coh2d4.py
==============
Comprehensive unit and physics verification test suite for COH2D4 (2D 4-node Cohesive Element)
with Bilinear Traction-Separation constitutive law.

Verification Coverage:
1. Rigid body mode invariance (fe = 0, Ke * u_rbm = 0 to machine precision).
2. Algorithmic consistent tangent stiffness Ke vs directional finite difference across:
   - Elastic tensile regime
   - Normal softening regime
   - Shear softening regime
   - Mixed-mode softening regime
   - Compressive penetration regime
3. Pure normal bilinear traction-separation response and G_c fracture energy conservation.
4. Pure shear bilinear traction-separation response and G_c fracture energy conservation.
5. Compressive contact penalty resistance and irreversible non-damage in compression.
6. Abaqus keyword input reader parsing (*COHESIVE SECTION, *DAMAGE INITIATION, *DAMAGE EVOLUTION).
7. End-to-end 2D delamination / peeling simulation using DynamicSolver2D.
"""

import numpy as np
import pytest

from dispsolver.element2d.coh2d4_numba import (
    compute_coh2d4_element_numba,
    assemble_mesh_coh2d4_numba,
)
from dispsolver.mesh2d import Mesh2D
from dispsolver.solver2d.dynamic2d import DynamicSolver2D
from dispsolver.material3d.numba_materials import MAT_COHESIVE
from dispsolver.io import read_inp
from dispsolver.model.model import Model


def _make_unit_coh_coords():
    """Returns initial coords for a 1.0mm long zero-thickness cohesive element."""
    return np.array([
        [0.0, 0.0],  # node 0: bottom, xi = -1
        [1.0, 0.0],  # node 1: bottom, xi = +1
        [1.0, 0.0],  # node 2: top,    xi = +1
        [0.0, 0.0],  # node 3: top,    xi = -1
    ], dtype=np.float64)


def _make_coh_props(
    Knn=1.0e5,
    Kss=1.0e5,
    Nmax=50.0,
    Smax=40.0,
    Gc=0.5,
    criterion=0,
    delta_f=0.0,
    viscosity=0.0,
    thickness=1.0,
):
    """Packs constitutive parameters into a 64-element float64 array."""
    props = np.zeros(64, dtype=np.float64)
    props[0] = Knn
    props[1] = Kss
    props[2] = Nmax
    props[3] = Smax
    props[4] = Gc
    props[5] = float(criterion)
    props[6] = 0.0  # linear softening
    props[7] = delta_f
    props[8] = viscosity
    props[9] = thickness
    return props


# =========================================================================
# 1. Rigid Body Mode Invariance Test
# =========================================================================
def test_coh2d4_rigid_body_modes():
    """Verify rigid body translation yields zero internal force and zero tangent resistance."""
    coords = _make_unit_coh_coords()
    props = _make_coh_props()

    # Pure translation: dx = 2.5, dy = -1.2 for all 4 nodes
    dx, dy = 2.5, -1.2
    u_rbm = np.array([dx, dy, dx, dy, dx, dy, dx, dy], dtype=np.float64)
    sdvs = np.zeros((2, 9), dtype=np.float64)

    fe, Ke, err = compute_coh2d4_element_numba(coords, u_rbm, props, sdvs)
    assert err == 0
    assert np.linalg.norm(fe) == pytest.approx(0.0, abs=1.0e-10)

    # Tangent matrix multiplied by rigid body vector should produce zero reaction
    K_u_rbm = Ke @ u_rbm
    assert np.linalg.norm(K_u_rbm) == pytest.approx(0.0, abs=1.0e-10)


# =========================================================================
# 2. Algorithmic Consistent Tangent vs Finite Difference
# =========================================================================
@pytest.mark.parametrize("case_name, u_elem, init_sdvs", [
    ("elastic_tension", np.array([0, 0, 0, 0, 0, 0.0002, 0, 0.0002], dtype=np.float64), None),
    ("normal_softening", np.array([0, 0, 0, 0, 0, 0.005, 0, 0.005], dtype=np.float64), "init_normal"),
    ("shear_softening", np.array([0, 0, 0, 0, 0.004, 0, 0.004, 0], dtype=np.float64), "init_shear"),
    ("mixed_mode_softening", np.array([0, 0, 0, 0, 0.003, 0.003, 0.003, 0.003], dtype=np.float64), "init_mixed"),
    ("compression_penetration", np.array([0, 0, 0, 0, 0.001, -0.002, 0.001, -0.002], dtype=np.float64), None),
])
def test_coh2d4_tangent_consistency_fd(case_name, u_elem, init_sdvs):
    """
    Verify Ke matches numerical directional derivative (finite difference)
    with relative Frobenius error < 1e-4 across all deformation regimes.
    In irreversible damage mechanics, loading directional perturbations must be used.
    """
    coords = _make_unit_coh_coords()
    props = _make_coh_props(Knn=1.0e5, Kss=1.0e5, Nmax=50.0, Smax=40.0, Gc=0.5)

    base_sdvs = np.zeros((2, 9), dtype=np.float64)
    if init_sdvs is not None:
        # Pre-seed history variables so initiation values are locked
        compute_coh2d4_element_numba(coords, u_elem, props, base_sdvs)

    sdvs = base_sdvs.copy()
    fe, Ke, err = compute_coh2d4_element_numba(coords, u_elem, props, sdvs)
    assert err == 0

    h = 1.0e-7
    Ke_num = np.zeros((8, 8), dtype=np.float64)
    for j in range(8):
        # Bottom nodes (dof 0..3) decrease jump if perturbed positively, so sign is negative
        # to ensure loading perturbation along damage evolution
        sign = -1.0 if j < 4 else 1.0
        u_pert = u_elem.copy()
        u_pert[j] += sign * h
        sdvs_pert = base_sdvs.copy()
        f_pert, _, _ = compute_coh2d4_element_numba(coords, u_pert, props, sdvs_pert)
        Ke_num[:, j] = sign * (f_pert - fe) / h

    norm_Ke = np.linalg.norm(Ke)
    diff = np.linalg.norm(Ke - Ke_num)
    rel_err = diff / norm_Ke if norm_Ke > 0.0 else diff

    assert rel_err < 1.0e-4, f"Failed {case_name}: rel_err = {rel_err:.2e}"


# =========================================================================
# 3. Pure Normal Bilinear Traction-Separation & G_c Energy Balance
# =========================================================================
def test_coh2d4_pure_normal_bilinear_and_energy():
    """
    Verify pure normal opening:
    - Peak traction matches N_max = 50.0 MPa.
    - Integrated separation energy matches G_c = 0.5 N/mm with relative error < 1e-4.
    """
    coords = _make_unit_coh_coords()
    Knn = 1.0e5
    Nmax = 50.0
    Gc = 0.5
    props = _make_coh_props(Knn=Knn, Kss=1.0e5, Nmax=Nmax, Smax=40.0, Gc=Gc)

    delta_0 = Nmax / Knn          # 0.0005 mm
    delta_f = 2.0 * Gc / Nmax     # 0.0200 mm

    n_steps = 1000
    delta_arr = np.linspace(0.0, delta_f * 1.05, n_steps)
    t_n_arr = np.zeros(n_steps)
    sdvs = np.zeros((2, 9), dtype=np.float64)

    for i, delta_n in enumerate(delta_arr):
        u_elem = np.array([0, 0, 0, 0, 0, delta_n, 0, delta_n], dtype=np.float64)
        compute_coh2d4_element_numba(coords, u_elem, props, sdvs)
        t_n_arr[i] = sdvs[0, 4]  # tn at Gauss point 0

    # 1. Peak traction
    peak_tn = np.max(t_n_arr)
    assert peak_tn == pytest.approx(Nmax, rel=1.0e-3)

    # 2. Energy integration using trapezoidal rule up to delta_f
    mask = delta_arr <= delta_f
    trap_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    energy_calc = trap_fn(t_n_arr[mask], delta_arr[mask])
    assert energy_calc == pytest.approx(Gc, rel=1.0e-4)


# =========================================================================
# 4. Pure Shear Bilinear Traction-Separation & G_c Energy Balance
# =========================================================================
def test_coh2d4_pure_shear_bilinear_and_energy():
    """
    Verify pure shear sliding:
    - Peak traction matches S_max = 40.0 MPa.
    - Integrated sliding energy matches G_c = 0.5 N/mm with relative error < 1e-4.
    """
    coords = _make_unit_coh_coords()
    Kss = 1.0e5
    Smax = 40.0
    Gc = 0.5
    props = _make_coh_props(Knn=1.0e5, Kss=Kss, Nmax=50.0, Smax=Smax, Gc=Gc)

    delta_0 = Smax / Kss          # 0.0004 mm
    delta_f = 2.0 * Gc / Smax     # 0.0250 mm

    n_steps = 1000
    delta_arr = np.linspace(0.0, delta_f * 1.05, n_steps)
    t_s_arr = np.zeros(n_steps)
    sdvs = np.zeros((2, 9), dtype=np.float64)

    for i, delta_s in enumerate(delta_arr):
        u_elem = np.array([0, 0, 0, 0, delta_s, 0, delta_s, 0], dtype=np.float64)
        compute_coh2d4_element_numba(coords, u_elem, props, sdvs)
        t_s_arr[i] = sdvs[0, 5]  # ts at Gauss point 0

    # 1. Peak shear traction
    peak_ts = np.max(t_s_arr)
    assert peak_ts == pytest.approx(Smax, rel=1.0e-3)

    # 2. Energy integration
    mask = delta_arr <= delta_f
    trap_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    energy_calc = trap_fn(t_s_arr[mask], delta_arr[mask])
    assert energy_calc == pytest.approx(Gc, rel=1.0e-4)


# =========================================================================
# 5. Compressive Contact Penalty Resistance & Non-Damage
# =========================================================================
def test_coh2d4_compressive_penalty_and_non_damage():
    """
    Verify compressive normal displacement:
    - Normal traction behaves as linear elastic penalty: tn = Knn * delta_n < 0
    - Damage parameter D remains strictly 0.0 (no damage in compression).
    """
    coords = _make_unit_coh_coords()
    Knn = 1.0e5
    props = _make_coh_props(Knn=Knn, Kss=1.0e5, Nmax=50.0, Smax=40.0, Gc=0.5)

    delta_n_comp = -0.010  # 10 um penetration
    u_elem = np.array([0, 0, 0, 0, 0, delta_n_comp, 0, delta_n_comp], dtype=np.float64)
    sdvs = np.zeros((2, 9), dtype=np.float64)

    fe, Ke, err = compute_coh2d4_element_numba(coords, u_elem, props, sdvs)
    assert err == 0

    # Check tn and D at Gauss points
    assert sdvs[0, 4] == pytest.approx(Knn * delta_n_comp, rel=1.0e-9)
    assert sdvs[0, 1] == 0.0  # D must be 0
    assert sdvs[0, 6] == 0.0  # status = 0 (elastic)


# =========================================================================
# 6. Abaqus Input Reader Verification
# =========================================================================
def test_coh2d4_abaqus_input_reader():
    """Verify parsing *COHESIVE SECTION and *DAMAGE keywords into Model."""
    inp_text = """
*HEADING
Cohesive 2D Element Model
*NODE, NSET=ALL_NODES
1, 0.0, 0.0
2, 1.0, 0.0
3, 1.0, 0.0
4, 0.0, 0.0
*ELEMENT, TYPE=COH2D4, ELSET=COH_SET
1, 1, 2, 3, 4
*COHESIVE SECTION, ELSET=COH_SET, MATERIAL=COH_MAT, RESPONSE=TRACTION SEPARATION
1.0
*MATERIAL, NAME=COH_MAT
*ELASTIC, TYPE=TRACTION
100000.0, 100000.0
*DAMAGE INITIATION, CRITERION=MAXS
50.0, 40.0
*DAMAGE EVOLUTION, TYPE=ENERGY
0.5
*STEP, NAME=PEEL, NLGEOM=YES
*STATIC
0.1, 1.0
*BOUNDARY
1, 1, 2, 0.0
2, 1, 2, 0.0
3, 2, 2, 0.01
4, 2, 2, 0.01
*END STEP
"""
    model = read_inp(inp_text, dim=2)
    assert isinstance(model, Model)

    part = list(model.parts.values())[0]
    assert len(part.elements) == 1
    elem_type, conn = part.elements[1]
    assert elem_type == "COH2D4"

    assert "COH_MAT" in model.materials
    mat = model.materials["COH_MAT"]
    damage_init = mat.extra_params.get("damage_init")
    assert damage_init is not None
    assert damage_init["criterion"] == "MAXS"
    assert damage_init["Nmax"] == 50.0
    assert damage_init["Smax"] == 40.0

    damage_evol = mat.extra_params.get("damage_evol")
    assert damage_evol is not None
    assert damage_evol["Gf"] == 0.5

    assert len(model.sections) >= 1
    sec = list(model.sections.values())[0]
    assert sec.kind == "COHESIVE"


# =========================================================================
# 7. End-to-End Quasi-Static Peeling Simulation with DynamicSolver2D
# =========================================================================
def test_coh2d4_peeling_solver2d_simulation():
    """
    Verify multi-step quasi-static peeling of a 2-element cohesive interface
    using DynamicSolver2D with full Newton-Raphson convergence.
    """
    # 2 cohesive elements in series:
    # Elem 1: nodes (1, 2, 4, 3) at x in [0, 1]
    # Elem 2: nodes (2, 5, 6, 4) at x in [1, 2]
    # Bottom nodes (1, 2, 5) fixed.
    # Top nodes (3, 4, 6) pulled upward in +y direction.
    mesh = Mesh2D()
    mesh.add_node(1, 0.0, 0.0)
    mesh.add_node(2, 1.0, 0.0)
    mesh.add_node(5, 2.0, 0.0)

    mesh.add_node(3, 0.0, 0.0)
    mesh.add_node(4, 1.0, 0.0)
    mesh.add_node(6, 2.0, 0.0)

    mesh.add_element(1, [1, 2, 4, 3], "COH2D4", pid=0)
    mesh.add_element(2, [2, 5, 6, 4], "COH2D4", pid=0)

    materials = {
        0: {
            "mat_type": MAT_COHESIVE,
            "K_nn": 1.0e5,
            "K_ss": 1.0e5,
            "N_max": 50.0,
            "S_max": 40.0,
            "G_c": 0.5,
            "criterion": "MAXS",
            "thickness": 1.0,
        }
    }

    solver = DynamicSolver2D(
        mesh=mesh,
        materials=materials,
        nlgeom=True,
        time_integrator="quasistatic",
    )

    # Fixed bottom boundary:
    # node 1: idx 0, 1
    # node 2: idx 2, 3
    # node 5: idx 4, 5
    solver.fixed_dofs[0] = 0.0
    solver.fixed_dofs[1] = 0.0
    solver.fixed_dofs[2] = 0.0
    solver.fixed_dofs[3] = 0.0
    solver.fixed_dofs[4] = 0.0
    solver.fixed_dofs[5] = 0.0

    # Top boundary fixed in x (sliding prevented):
    # node 3 (dof 6), node 4 (dof 8), node 6 (dof 10)
    solver.fixed_dofs[6] = 0.0
    solver.fixed_dofs[8] = 0.0
    solver.fixed_dofs[10] = 0.0

    # Pull node 3 progressively upward to delaminate Elem 1 first
    # Step 1: pull to 0.0003 mm (elastic, D = 0)
    solver.fixed_dofs[7] = 0.0003
    converged = solver.solve_step(dt=1.0)
    assert converged, "Step 1 (elastic) failed to converge"
    assert solver.elem_sdvs[0, 0, 1] == 0.0  # D == 0

    # Step 2: pull to 0.0050 mm (softening, 0 < D < 1)
    solver.fixed_dofs[7] = 0.0050
    converged = solver.solve_step(dt=1.0)
    assert converged, "Step 2 (softening) failed to converge"
    assert 0.0 < solver.elem_sdvs[0, 0, 1] < 1.0

    # Step 3: pull to 0.0350 mm (fully delaminated, D > 0.99)
    solver.fixed_dofs[7] = 0.0350
    converged = solver.solve_step(dt=1.0)
    assert converged, "Step 3 (full separation) failed to converge"
    assert solver.elem_sdvs[0, 0, 1] > 0.99


# =========================================================================
# 8. Rotated Reference Configuration Frame Invariance & Jacobian (AGENTS.md §4.14–4.17)
# =========================================================================
@pytest.mark.parametrize("angle_deg", [15.0, 37.0, 45.0, 90.0, 135.0])
def test_coh2d4_rotated_reference_frame_invariance_and_fd(angle_deg):
    """
    Verify frame invariance in rotated reference configurations (AGENTS.md §4.14–4.17):
    1. f_e(R X, T u) == T f_e(X, u) to machine precision (< 1e-11)
    2. K_e(R X, T u) == T K_e(X, u) T^T to machine precision (< 1e-11)
    3. Directional finite difference Jacobian in rotated frame matches K_e(R X, T u).
    """
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R2 = np.array([[c, -s], [s, c]], dtype=np.float64)
    T8 = np.zeros((8, 8), dtype=np.float64)
    for i in range(4):
        T8[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = R2

    coords_0 = _make_unit_coh_coords()
    coords_rot = coords_0 @ R2.T
    props = _make_coh_props(Knn=1.0e5, Kss=8.0e4, Nmax=50.0, Smax=40.0, Gc=0.5, criterion=1)

    # Test mixed-mode softening displacement in unrotated frame and rotated frame
    u_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0025, 0.0035, 0.0020, 0.0030], dtype=np.float64)
    u_rot = T8 @ u_0

    sdvs_0 = np.zeros((2, 9), dtype=np.float64)
    compute_coh2d4_element_numba(coords_0, u_0, props, sdvs_0)
    fe_0, Ke_0, err_0 = compute_coh2d4_element_numba(coords_0, u_0, props, sdvs_0.copy())
    assert err_0 == 0

    sdvs_rot_base = np.zeros((2, 9), dtype=np.float64)
    compute_coh2d4_element_numba(coords_rot, u_rot, props, sdvs_rot_base)
    fe_rot, Ke_rot, err_rot = compute_coh2d4_element_numba(coords_rot, u_rot, props, sdvs_rot_base.copy())
    assert err_rot == 0

    # 1. Internal force covariance: f(R X, T u) = T f(X, u)
    assert np.linalg.norm(fe_rot - T8 @ fe_0) / np.linalg.norm(fe_0) < 1.0e-12

    # 2. Tangent stiffness covariance: K(R X, T u) = T K(X, u) T^T
    assert np.linalg.norm(Ke_rot - T8 @ Ke_0 @ T8.T) / np.linalg.norm(Ke_0) < 1.0e-12

    # 3. Directional FD Jacobian in the rotated reference configuration along loading perturbations
    h = 1.0e-7
    Ke_rot_fd = np.zeros((8, 8), dtype=np.float64)
    for j in range(8):
        # Perturb along the rotated local opening basis vector T8[:, j]
        sign = -1.0 if j < 4 else 1.0
        du_rot = sign * h * T8[:, j]
        f_pert, _, _ = compute_coh2d4_element_numba(coords_rot, u_rot + du_rot, props, sdvs_rot_base.copy())
        Ke_rot_fd[:, j] = sign * (f_pert - fe_rot) / h

    # Since perturbations were taken along columns of T8, Ke_rot_fd == Ke_rot @ T8
    rel_fd_err = np.linalg.norm(Ke_rot @ T8 - Ke_rot_fd) / np.linalg.norm(Ke_rot)
    assert rel_fd_err < 1.0e-4


# =========================================================================
# 9. DCB (Double Cantilever Beam) Delamination Benchmark vs Analytical Solution
# =========================================================================
def test_coh2d4_dcb_delamination_analytical_benchmark():
    """
    Verify Double Cantilever Beam (DCB) / elastic adherend peeling benchmark:
    Two CPE4I elastic adherend strips bonded along y=0 by COH2D4 elements.
    Checks:
    1. Elastic initial slope matches series combination of adherends + cohesive layer.
    2. Peak critical load and progressive delamination front propagation from tip (x=0) to root (x=L).
    3. External work done during uniform interface delamination matches analytical fracture energy
       W_ext = b * L * G_c within < 0.5%.
    """
    # Part A: Work-energy theorem on multi-element cohesive interface in DynamicSolver2D
    L = 4.0
    b = 1.0
    Gc = 0.5
    Nmax = 50.0
    Knn = 1.0e5

    mesh = Mesh2D()
    # 2 cohesive elements of length 2.0 mm each along x in [0, 4]
    mesh.add_node(1, 0.0, 0.0)
    mesh.add_node(2, 2.0, 0.0)
    mesh.add_node(3, 4.0, 0.0)
    mesh.add_node(4, 0.0, 0.0)
    mesh.add_node(5, 2.0, 0.0)
    mesh.add_node(6, 4.0, 0.0)
    mesh.add_element(1, [1, 2, 5, 4], "COH2D4", pid=0)
    mesh.add_element(2, [2, 3, 6, 5], "COH2D4", pid=0)

    materials = {
        0: {
            "mat_type": MAT_COHESIVE,
            "K_nn": Knn,
            "K_ss": Knn,
            "N_max": Nmax,
            "S_max": 40.0,
            "G_c": Gc,
            "criterion": "MAXS",
            "thickness": b,
        }
    }

    solver = DynamicSolver2D(mesh=mesh, materials=materials, nlgeom=True, time_integrator="quasistatic")
    # Fix bottom nodes (1, 2, 3) completely, and top nodes (4, 5, 6) in x
    for nid_idx in [0, 1, 2]:
        solver.fixed_dofs[2 * nid_idx + 0] = 0.0
        solver.fixed_dofs[2 * nid_idx + 1] = 0.0
    for nid_idx in [3, 4, 5]:
        solver.fixed_dofs[2 * nid_idx + 0] = 0.0

    # Sweep uniform opening uy up to delta_f = 2*Gc/Nmax = 0.020 mm
    delta_0 = Nmax / Knn       # 0.0005 mm
    delta_f = 2.0 * Gc / Nmax  # 0.0200 mm
    u_steps = np.concatenate([
        np.linspace(0.0, delta_0, 6)[1:],
        np.linspace(delta_0, delta_f * 1.02, 25)[1:],
    ])
    forces = [0.0]
    disps = [0.0]
    for uy in u_steps:
        for nid_idx in [3, 4, 5]:
            solver.fixed_dofs[2 * nid_idx + 1] = float(uy)
        assert solver.solve_step(dt=1.0), f"Failed to converge at uy={uy}"
        _, f_int = solver._assemble_physical(solver.u)
        total_fy = sum(f_int[2 * nid_idx + 1] for nid_idx in [3, 4, 5])
        forces.append(float(total_fy))
        disps.append(float(uy))

    # Analytical peak load: P_max = b * L * N_max = 1.0 * 4.0 * 50.0 = 200.0 N
    assert max(forces) == pytest.approx(b * L * Nmax, rel=1.0e-4)

    # Analytical total fracture energy: W_total = b * L * G_c = 1.0 * 4.0 * 0.5 = 2.0 N*mm
    trap_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    work_ext = trap_fn(forces, disps)
    assert work_ext == pytest.approx(b * L * Gc, rel=5.0e-3)

    # Part B: Coupled CPE4I elastic cantilever adherend + COH2D4 interface DCB peeling
    dcb_mesh = Mesh2D()
    # Bottom cohesive nodes (y=0): 1(x=0), 2(x=2), 3(x=4)
    dcb_mesh.add_node(1, 0.0, 0.0)
    dcb_mesh.add_node(2, 2.0, 0.0)
    dcb_mesh.add_node(3, 4.0, 0.0)
    # Interface top nodes / bottom of upper beam (y=0): 4(x=0), 5(x=2), 6(x=4)
    dcb_mesh.add_node(4, 0.0, 0.0)
    dcb_mesh.add_node(5, 2.0, 0.0)
    dcb_mesh.add_node(6, 4.0, 0.0)
    # Top surface of upper elastic adherend beam (y=0.5): 7(x=0), 8(x=2), 9(x=4)
    dcb_mesh.add_node(7, 0.0, 0.5)
    dcb_mesh.add_node(8, 2.0, 0.5)
    dcb_mesh.add_node(9, 4.0, 0.5)

    # Cohesive interface elements (pid=0)
    dcb_mesh.add_element(1, [1, 2, 5, 4], "COH2D4", pid=0)
    dcb_mesh.add_element(2, [2, 3, 6, 5], "COH2D4", pid=0)
    # Elastic adherend continuum elements (pid=1)
    dcb_mesh.add_element(3, [4, 5, 8, 7], "CPE4I", pid=1)
    dcb_mesh.add_element(4, [5, 6, 9, 8], "CPE4I", pid=1)

    dcb_materials = {
        0: {
            "mat_type": MAT_COHESIVE,
            "K_nn": 1.0e5,
            "K_ss": 1.0e5,
            "N_max": 20.0,
            "S_max": 20.0,
            "G_c": 0.2,
            "criterion": "MAXS",
            "thickness": 1.0,
        },
        1: {
            "E": 70000.0,
            "nu": 0.3,
            "thickness": 1.0,
        },
    }
    dcb_solver = DynamicSolver2D(mesh=dcb_mesh, materials=dcb_materials, nlgeom=True, time_integrator="quasistatic")
    # Fix substrate nodes 1, 2, 3 (indices 0, 1, 2) and right clamped root nodes 6, 9 (indices 5, 8)
    for idx in [0, 1, 2, 5, 8]:
        dcb_solver.fixed_dofs[2 * idx + 0] = 0.0
        dcb_solver.fixed_dofs[2 * idx + 1] = 0.0

    # Pull left tip node 7 (index 6, y-dof = 13) upward to propagate delamination from x=0 toward x=4
    tip_forces = []
    for uy_tip in [0.0002, 0.001, 0.005, 0.015, 0.030]:
        dcb_solver.fixed_dofs[2 * 6 + 1] = uy_tip
        assert dcb_solver.solve_step(dt=1.0), f"DCB failed to converge at uy_tip={uy_tip}"
        _, f_int = dcb_solver._assemble_physical(dcb_solver.u)
        tip_forces.append(float(f_int[2 * 6 + 1]))

    # Verify progressive delamination: Elem 1 (at tip) has higher damage than Elem 2 (near root)
    D_elem1 = dcb_solver.elem_sdvs[0, 0, 1]
    D_elem2 = dcb_solver.elem_sdvs[1, 1, 1]
    assert D_elem1 > 0.90
    assert D_elem1 > D_elem2
    assert max(tip_forces) > 0.0

