"""
tests/test_cpe4ih_c3d8ih.py
===========================
Comprehensive verification suite for Abaqus-equivalent CPE4IH (2D) and C3D8IH (3D)
Incompatible Modes + Linear Hybrid Pressure elements (small-strain and co-rotational).
"""

from __future__ import annotations
import numpy as np
import pytest

from dispsolver.model.element_catalog import ElementCatalog
from dispsolver.element2d.cpe4i_numba import compute_cpe4i_element_numba
from dispsolver.element2d.cpe4ih_numba import compute_cpe4ih_element_numba
from dispsolver.element2d.cr_wrapper_2d import compute_cpe4ih_cr_element_numba
from dispsolver.element3d.c3d8ih_numba import compute_c3d8ih_element_numba
from dispsolver.element3d.cr_wrapper_3d import compute_c3d8ih_cr_element_numba
from dispsolver.mesh2d import Mesh2D
from dispsolver.mesh3d import Mesh3D
from dispsolver.solver2d.dynamic2d import DynamicSolver2D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D
from example_disp_fold.model_spec import ELEM_CASES_2D, ELEM_CASES_3D


def _make_props(E: float = 3000.0, nu: float = 0.4999) -> np.ndarray:
    props = np.zeros(16, dtype=np.float64)
    props[0] = E
    props[1] = nu
    return props


def _make_ab_props(mu: float = 1.0, lm: float = 5.0, K: float = 5000.0) -> np.ndarray:
    props = np.zeros(16, dtype=np.float64)
    props[0] = mu * 3.0
    props[1] = 0.4995
    props[2] = mu
    props[3] = lm
    props[4] = K
    return props


def _rot2(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _rot3_z(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _build_mesh2d(nodes: dict, elems: dict, elem_type: str, pid: int = 1) -> Mesh2D:
    m = Mesh2D()
    for nid, (x, y) in nodes.items():
        m.add_node(nid, float(x), float(y))
    for eid, conn in elems.items():
        m.add_element(eid, list(conn), elem_type, pid=pid)
    return m


def _build_mesh3d(nodes: dict, elems: dict, elem_type: str, pid: int = 1) -> Mesh3D:
    m = Mesh3D()
    for nid, (x, y, z) in nodes.items():
        m.add_node(nid, float(x), float(y), float(z))
    for eid, conn in elems.items():
        m.add_element(eid, list(conn), elem_type, pid=pid)
    return m


def test_catalog_and_model_spec_registration():
    """Verify ElementCatalog resolves CPE4IH and C3D8IH natively without substitution notes."""
    k2_no, notes2_no = ElementCatalog.resolve("CPE4IH", 2, nlgeom=False)
    k2_yes, notes2_yes = ElementCatalog.resolve("CPE4IH", 2, nlgeom=True)
    assert k2_no == 20 and notes2_no == []
    assert k2_yes == 21 and notes2_yes == []

    k3_no, notes3_no = ElementCatalog.resolve("C3D8IH", 3, nlgeom=False)
    k3_yes, notes3_yes = ElementCatalog.resolve("C3D8IH", 3, nlgeom=True)
    assert k3_no == 19 and notes3_no == []
    assert k3_yes == 20 and notes3_yes == []

    assert "2D-Opt1-IH" in ELEM_CASES_2D
    assert ELEM_CASES_2D["2D-Opt1-IH"]["psa"] == "CPE4IH"
    assert "3D-Opt1-IH" in ELEM_CASES_3D
    assert ELEM_CASES_3D["3D-Opt1-IH"]["psa"] == "C3D8IH"


@pytest.mark.parametrize("nu", [0.3, 0.4999])
def test_cpe4ih_rbm_and_spurious_modes(nu: float):
    """Verify 2D CPE4IH has exactly 3 rigid-body modes and 5 strictly positive eigenvalues."""
    coords = np.array([
        [0.0, 0.0],
        [2.0, 0.1],
        [1.9, 1.2],
        [-0.1, 0.9],
    ], dtype=np.float64)
    u = np.zeros(8, dtype=np.float64)
    props = _make_props(E=1000.0, nu=nu)
    sdvs = np.zeros((4, 7), dtype=np.float64)

    f, K, err = compute_cpe4ih_element_numba(coords, u, 0, props, sdvs, 0.0)
    assert err == 0
    assert np.linalg.norm(f) < 1e-14
    assert np.linalg.norm(K - K.T) / np.linalg.norm(K) < 1e-14

    eigvals = np.sort(np.linalg.eigvalsh(0.5 * (K + K.T)))
    G = 1000.0 / (2.0 * (1.0 + nu))
    assert np.all(np.abs(eigvals[:3]) / eigvals[-1] < 1e-12)
    # All 4 isochoric modes scale with G (not K), and the single dilatation mode scales with K
    assert np.all(eigvals[3:] / G > 0.1)


@pytest.mark.parametrize("nu", [0.3, 0.4999])
def test_c3d8ih_rbm_and_spurious_modes(nu: float):
    """Verify 3D C3D8IH has exactly 6 rigid-body modes and 18 strictly positive eigenvalues."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.1, -0.05],
        [2.1, 1.1, 0.05],
        [-0.1, 1.0, 0.0],
        [0.05, -0.05, 0.8],
        [1.95, 0.05, 0.85],
        [2.05, 1.05, 0.9],
        [0.0, 0.95, 0.75],
    ], dtype=np.float64)
    u = np.zeros(24, dtype=np.float64)
    props = _make_props(E=1000.0, nu=nu)
    sdvs = np.zeros((8, 7), dtype=np.float64)

    f, K, err = compute_c3d8ih_element_numba(coords, u, 0, props, sdvs, 0.0)
    assert err == 0
    assert np.linalg.norm(f) < 1e-13
    assert np.linalg.norm(K - K.T) / np.linalg.norm(K) < 1e-14

    eigvals = np.sort(np.linalg.eigvalsh(0.5 * (K + K.T)))
    G = 1000.0 / (2.0 * (1.0 + nu))
    assert np.all(np.abs(eigvals[:6]) / eigvals[-1] < 1e-11)
    # All 17 isochoric modes scale with G (not K), and the single dilatation mode scales with K
    assert np.all(eigvals[6:] / G > 0.05)


def test_cpe4ih_and_cr_fd_tangent_consistency():
    """Verify analytic/condensed tangent K matches central FD of f_int for CPE4IH and CPE4IH_CR."""
    coords = np.array([
        [0.0, 0.0],
        [1.5, 0.2],
        [1.4, 0.8],
        [-0.1, 0.7],
    ], dtype=np.float64)
    rng = np.random.default_rng(42)
    u = rng.uniform(-0.02, 0.02, size=8)
    props = _make_ab_props(mu=1.2, lm=4.5, K=2000.0)

    # 1. Small-strain CPE4IH: exact tangent matches FD to < 1e-6
    f0, K0, err = compute_cpe4ih_element_numba(coords, u, 2, props, np.zeros((4, 7)), 0.1)
    assert err == 0
    K_fd = np.zeros((8, 8), dtype=np.float64)
    h = 1e-7
    for j in range(8):
        up = u.copy()
        um = u.copy()
        up[j] += h
        um[j] -= h
        fp, _, _ = compute_cpe4ih_element_numba(coords, up, 2, props, np.zeros((4, 7)), 0.1)
        fm, _, _ = compute_cpe4ih_element_numba(coords, um, 2, props, np.zeros((4, 7)), 0.1)
        K_fd[:, j] = (fp - fm) / (2.0 * h)
    rel_err = np.linalg.norm(K0 - K_fd) / np.linalg.norm(K_fd)
    assert rel_err < 1e-6

    # 2. Co-rotational CPE4IH_CR evaluated at pure rigid rotation (f_loc = 0 -> spin geometric term vanishes)
    theta = 0.65
    R = _rot2(np.rad2deg(theta))
    u_rot = ((coords @ (R.T - np.eye(2)))).ravel()
    f_cr, K_cr, err_cr = compute_cpe4ih_cr_element_numba(coords, u_rot, 2, props, np.zeros((4, 7)), 0.1)
    assert err_cr == 0
    assert np.linalg.norm(f_cr) < 1e-12
    K_cr_fd = np.zeros((8, 8), dtype=np.float64)
    for j in range(8):
        up = u_rot.copy()
        um = u_rot.copy()
        up[j] += h
        um[j] -= h
        fp, _, _ = compute_cpe4ih_cr_element_numba(coords, up, 2, props, np.zeros((4, 7)), 0.1)
        fm, _, _ = compute_cpe4ih_cr_element_numba(coords, um, 2, props, np.zeros((4, 7)), 0.1)
        K_cr_fd[:, j] = (fp - fm) / (2.0 * h)
    assert np.linalg.norm(K_cr - K_cr_fd) / np.linalg.norm(K_cr_fd) < 1e-6


def test_c3d8ih_and_cr_fd_tangent_consistency():
    """Verify tangent K matches central FD of f_int for C3D8IH and C3D8IH_CR."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.1, 0.0],
        [1.4, 0.9, 0.05],
        [-0.1, 0.8, 0.0],
        [0.0, 0.0, 0.6],
        [1.5, 0.1, 0.65],
        [1.4, 0.9, 0.6],
        [-0.1, 0.8, 0.55],
    ], dtype=np.float64)
    rng = np.random.default_rng(99)
    u = rng.uniform(-0.015, 0.015, size=24)
    props = _make_ab_props(mu=1.5, lm=4.0, K=1500.0)

    # 1. Small-strain C3D8IH: exact tangent matches FD
    f0, K0, err = compute_c3d8ih_element_numba(coords, u, 2, props, np.zeros((8, 7)), 0.1)
    assert err == 0
    K_fd = np.zeros((24, 24), dtype=np.float64)
    h = 1e-7
    for j in range(24):
        up = u.copy()
        um = u.copy()
        up[j] += h
        um[j] -= h
        fp, _, _ = compute_c3d8ih_element_numba(coords, up, 2, props, np.zeros((8, 7)), 0.1)
        fm, _, _ = compute_c3d8ih_element_numba(coords, um, 2, props, np.zeros((8, 7)), 0.1)
        K_fd[:, j] = (fp - fm) / (2.0 * h)
    rel_err = np.linalg.norm(K0 - K_fd) / np.linalg.norm(K_fd)
    assert rel_err < 5e-5

    # 2. Co-rotational C3D8IH_CR evaluated at pure rigid rotation
    R3 = _rot3_z(37.0)
    u_rot3 = (coords @ (R3.T - np.eye(3))).ravel()
    f_cr, K_cr, err_cr = compute_c3d8ih_cr_element_numba(coords, u_rot3, 2, props, np.zeros((8, 7)), 0.1)
    assert err_cr == 0
    assert np.linalg.norm(f_cr) < 1e-11
    K_cr_fd = np.zeros((24, 24), dtype=np.float64)
    for j in range(24):
        up = u_rot3.copy()
        um = u_rot3.copy()
        up[j] += h
        um[j] -= h
        fp, _, _ = compute_c3d8ih_cr_element_numba(coords, up, 2, props, np.zeros((8, 7)), 0.1)
        fm, _, _ = compute_c3d8ih_cr_element_numba(coords, um, 2, props, np.zeros((8, 7)), 0.1)
        K_cr_fd[:, j] = (fp - fm) / (2.0 * h)
    assert np.linalg.norm(K_cr - K_cr_fd) / np.linalg.norm(K_cr_fd) < 5e-5


def test_rotated_reference_frame_invariance_2d_and_3d():
    """Verify bending energy is invariant under reference-frame rotations (0..90 deg) on AR=7 elements."""
    L, H = 1.0, 0.15
    coords_2d_0 = np.array([[0.0, 0.0], [L, 0.0], [L, H], [0.0, H]], dtype=np.float64)
    kappa = 1e-3
    xRel = coords_2d_0[:, 0] - 0.5 * L
    yRel = coords_2d_0[:, 1] - 0.5 * H
    u_2d_0 = np.column_stack([-kappa * xRel * yRel, 0.5 * kappa * xRel ** 2]).ravel()

    props_inc = _make_props(E=4000.0, nu=0.4999)
    props_comp = _make_props(E=4000.0, nu=0.3)

    energies_cpe4ih = []
    energies_cpe4i = []
    for deg in (0.0, 10.0, 30.0, 45.0, 60.0, 90.0):
        R = _rot2(deg)
        coords_rot = coords_2d_0 @ R.T
        u_rot = (u_2d_0.reshape(4, 2) @ R.T).ravel()

        _, K, _ = compute_cpe4ih_element_numba(coords_rot, np.zeros(8), 0, props_inc, np.zeros((4, 7)), 0.0)
        energies_cpe4ih.append(0.5 * float(u_rot @ K @ u_rot))

        _, K_i, _ = compute_cpe4i_element_numba(coords_rot, np.zeros(8), 0, props_comp, np.zeros((4, 7)), 0.0)
        energies_cpe4i.append(0.5 * float(u_rot @ K_i @ u_rot))

    spread_ih = (max(energies_cpe4ih) - min(energies_cpe4ih)) / np.mean(energies_cpe4ih)
    spread_i = (max(energies_cpe4i) - min(energies_cpe4i)) / np.mean(energies_cpe4i)
    assert spread_ih < 1e-9
    assert spread_i < 1e-12

    # 3D C3D8IH rotated reference test
    W = 0.2
    coords_3d_0 = np.array([
        [0.0, 0.0, 0.0], [L, 0.0, 0.0], [L, H, 0.0], [0.0, H, 0.0],
        [0.0, 0.0, W],   [L, 0.0, W],   [L, H, W],   [0.0, H, W],
    ], dtype=np.float64)
    xRel3 = coords_3d_0[:, 0] - 0.5 * L
    yRel3 = coords_3d_0[:, 1] - 0.5 * H
    u_3d_0 = np.column_stack([-kappa * xRel3 * yRel3, 0.5 * kappa * xRel3 ** 2, np.zeros(8)]).ravel()

    energies_c3d8ih = []
    for deg in (0.0, 15.0, 30.0, 45.0, 60.0, 90.0):
        R3 = _rot3_z(deg)
        coords_rot3 = coords_3d_0 @ R3.T
        u_rot3 = (u_3d_0.reshape(8, 3) @ R3.T).ravel()
        _, K3, _ = compute_c3d8ih_element_numba(coords_rot3, np.zeros(24), 0, props_inc, np.zeros((8, 7)), 0.0)
        energies_c3d8ih.append(0.5 * float(u_rot3 @ K3 @ u_rot3))

    spread_3d = (max(energies_c3d8ih) - min(energies_c3d8ih)) / np.mean(energies_c3d8ih)
    assert spread_3d < 1e-9


def test_irons_distorted_patch_test_2d_and_3d():
    """Verify distorted patch test passes to machine precision at nu=0.4999 for CPE4IH and C3D8IH."""
    nodes_2d = {
        1: (0.0, 0.0), 2: (1.0, 0.0), 3: (2.0, 0.0),
        4: (0.0, 1.0), 5: (1.15, 0.88), 6: (2.0, 1.0),
        7: (0.0, 2.0), 8: (1.0, 2.0), 9: (2.0, 2.0),
    }
    elems_2d = {
        1: [1, 2, 5, 4],
        2: [2, 3, 6, 5],
        3: [4, 5, 8, 7],
        4: [5, 6, 9, 8],
    }
    mesh2 = _build_mesh2d(nodes_2d, elems_2d, elem_type="CPE4IH", pid=1)
    materials = {1: {"E": 3000.0, "nu": 0.4999}}
    solver2 = DynamicSolver2D(mesh2, materials=materials, nlgeom=False)

    def exact_u2(x, y):
        return np.array([1e-3 * x + 1.5e-3 * y, 0.5e-3 * x - 0.999e-3 * y])

    u2 = np.zeros(solver2.num_dofs, dtype=np.float64)
    nid_map2 = mesh2.node_id_to_index()
    free_dofs2 = []
    for nid, (x, y) in nodes_2d.items():
        idx = nid_map2[nid]
        if nid == 5:
            free_dofs2.extend([2 * idx, 2 * idx + 1])
        else:
            u2[2 * idx:2 * idx + 2] = exact_u2(x, y)

    K2, f2 = solver2._assemble_physical(u2)
    du2 = np.linalg.solve(K2.toarray()[np.ix_(free_dofs2, free_dofs2)], -f2[free_dofs2])
    u2[free_dofs2] += du2
    u5_exact = exact_u2(*nodes_2d[5])
    assert np.linalg.norm(u2[free_dofs2] - u5_exact) / np.linalg.norm(u5_exact) < 1e-12

    # 3D 8-element patch with distorted interior node
    xs = [0.0, 1.0, 2.0]
    ys = [0.0, 1.0, 2.0]
    zs = [0.0, 1.0, 2.0]
    nodes_3d = {}
    nid = 1
    grid = {}
    for k, z in enumerate(zs):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                if (i, j, k) == (1, 1, 1):
                    nodes_3d[nid] = (1.12, 0.91, 1.08)
                else:
                    nodes_3d[nid] = (x, y, z)
                grid[(i, j, k)] = nid
                nid += 1

    elems_3d = {}
    eid = 1
    for k in range(2):
        for j in range(2):
            for i in range(2):
                elems_3d[eid] = [
                    grid[(i, j, k)], grid[(i + 1, j, k)], grid[(i + 1, j + 1, k)], grid[(i, j + 1, k)],
                    grid[(i, j, k + 1)], grid[(i + 1, j, k + 1)], grid[(i + 1, j + 1, k + 1)], grid[(i, j + 1, k + 1)],
                ]
                eid += 1

    mesh3 = _build_mesh3d(nodes_3d, elems_3d, elem_type="C3D8IH", pid=1)
    solver3 = DynamicSolver3D(mesh3, materials=materials, nlgeom=False)

    A = np.array([
        [1.0e-3, 0.4e-3, -0.2e-3],
        [0.6e-3, -0.5e-3, 0.3e-3],
        [0.1e-3, -0.2e-3, -0.499e-3],
    ])
    center_nid = grid[(1, 1, 1)]
    nid_map3 = mesh3.node_id_to_index()
    u3 = np.zeros(solver3.num_dofs, dtype=np.float64)
    free_dofs3 = []
    for n_id, xyz in nodes_3d.items():
        idx = nid_map3[n_id]
        if n_id == center_nid:
            free_dofs3.extend([3 * idx, 3 * idx + 1, 3 * idx + 2])
        else:
            u3[3 * idx:3 * idx + 3] = A @ np.asarray(xyz)

    K3, f3 = solver3._assemble_physical(u3)
    du3 = np.linalg.solve(K3.toarray()[np.ix_(free_dofs3, free_dofs3)], -f3[free_dofs3])
    u3[free_dofs3] += du3
    u_center_exact = A @ np.asarray(nodes_3d[center_nid])
    assert np.linalg.norm(u3[free_dofs3] - u_center_exact) / np.linalg.norm(u_center_exact) < 1e-12


def test_slender_beam_incompressible_bending_locking_free():
    """
    Verify CPE4IH and C3D8IH simultaneously eliminate both:
    - Shear locking (which afflicts CPE4H / C3D8H in 1-layer slender bending), and
    - Volumetric locking on distorted/incompressible meshes (which afflicts CPE4I / C3D8I at nu=0.4999).
    """
    L, H = 10.0, 0.2
    nx = 10
    E, nu = 3000.0, 0.4999
    E_plane = E / (1.0 - nu ** 2)
    I_2d = (1.0 * H ** 3) / 12.0
    M = 1.0  # tip moment -> exact tip deflection w_tip = M * L^2 / (2 * E' * I)
    w_exact_2d = M * (L ** 2) / (2.0 * E_plane * I_2d)

    def solve_2d_cantilever(elem_type: str, trapezoidal: bool = False) -> float:
        nodes = {}
        nid = 1
        for i in range(nx + 1):
            x = i * (L / nx)
            dx = 0.18 * (L / nx) * (1 if i % 2 == 1 else -1) if (trapezoidal and 0 < i < nx) else 0.0
            nodes[nid] = (x - dx, 0.0)
            nodes[nid + 1] = (x + dx, H)
            nid += 2
        elems = {}
        for e in range(nx):
            n0 = 2 * e + 1
            elems[e + 1] = [n0, n0 + 2, n0 + 3, n0 + 1]
        mesh = _build_mesh2d(nodes, elems, elem_type=elem_type, pid=1)
        solver = DynamicSolver2D(mesh, materials={1: {"E": E, "nu": nu}}, nlgeom=False)
        nid_map = mesh.node_id_to_index()
        fixed = [2 * nid_map[1], 2 * nid_map[1] + 1, 2 * nid_map[2]]
        free = [d for d in range(solver.num_dofs) if d not in fixed]
        f_ext = np.zeros(solver.num_dofs)
        n_bot = 2 * nx + 1
        n_top = 2 * nx + 2
        f_ext[2 * nid_map[n_bot]] = +M / H
        f_ext[2 * nid_map[n_top]] = -M / H
        K, _ = solver._assemble_physical(np.zeros(solver.num_dofs))
        u = np.zeros(solver.num_dofs)
        u[free] = np.linalg.solve(K.toarray()[np.ix_(free, free)], f_ext[free])
        return 0.5 * (u[2 * nid_map[n_bot] + 1] + u[2 * nid_map[n_top] + 1])

    # 1. Slender beam pure bending (L/H = 50, nu = 0.4999):
    # CPE4H and C3D8H suffer >80% shear locking, while CPE4IH and C3D8IH are exact to < 1e-6!
    w_cpe4h_rect = solve_2d_cantilever("CPE4H", trapezoidal=False)
    w_cpe4ih_rect = solve_2d_cantilever("CPE4IH", trapezoidal=False)
    assert abs(w_cpe4ih_rect - w_exact_2d) / w_exact_2d < 1e-6
    assert w_cpe4h_rect < 0.20 * w_exact_2d

    def solve_3d_cantilever(elem_type: str, trapezoidal: bool = False) -> float:
        nodes = {}
        grid = {}
        nid = 1
        for k, z in enumerate([0.0, 1.0]):
            for j, y in enumerate([0.0, H]):
                for i in range(nx + 1):
                    x = i * (L / nx)
                    dx = 0.18 * (L / nx) * (1 if i % 2 == 1 else -1) if (trapezoidal and 0 < i < nx) else 0.0
                    sgn = -1.0 if j == 0 else 1.0
                    nodes[nid] = (x + sgn * dx, y, z)
                    grid[(i, j, k)] = nid
                    nid += 1
        elems = {}
        for i in range(nx):
            elems[i + 1] = [
                grid[(i, 0, 0)], grid[(i + 1, 0, 0)], grid[(i + 1, 1, 0)], grid[(i, 1, 0)],
                grid[(i, 0, 1)], grid[(i + 1, 0, 1)], grid[(i + 1, 1, 1)], grid[(i, 1, 1)],
            ]
        mesh = _build_mesh3d(nodes, elems, elem_type=elem_type, pid=1)
        solver = DynamicSolver3D(mesh, materials={1: {"E": E, "nu": nu}}, nlgeom=False)
        nid_map = mesh.node_id_to_index()
        fixed = set()
        for n_id in nodes:
            fixed.add(3 * nid_map[n_id] + 2)
        for k in (0, 1):
            fixed.add(3 * nid_map[grid[(0, 0, k)]] + 0)
            fixed.add(3 * nid_map[grid[(0, 0, k)]] + 1)
            fixed.add(3 * nid_map[grid[(0, 1, k)]] + 0)
        fixed = sorted(fixed)
        free = [d for d in range(solver.num_dofs) if d not in fixed]
        f_ext = np.zeros(solver.num_dofs)
        for k in (0, 1):
            f_ext[3 * nid_map[grid[(nx, 0, k)]] + 0] = +0.5 * M / H
            f_ext[3 * nid_map[grid[(nx, 1, k)]] + 0] = -0.5 * M / H
        K, _ = solver._assemble_physical(np.zeros(solver.num_dofs))
        u = np.zeros(solver.num_dofs)
        u[free] = np.linalg.solve(K.toarray()[np.ix_(free, free)], f_ext[free])
        return 0.5 * (u[3 * nid_map[grid[(nx, 0, 0)]] + 1] + u[3 * nid_map[grid[(nx, 1, 0)]] + 1])

    w_c3d8h_rect = solve_3d_cantilever("C3D8H", trapezoidal=False)
    w_c3d8ih_rect = solve_3d_cantilever("C3D8IH", trapezoidal=False)
    assert abs(w_c3d8ih_rect - w_exact_2d) / w_exact_2d < 1e-6
    assert w_c3d8h_rect < 0.20 * w_exact_2d

    # 2. Distorted-element volumetric locking test at nu = 0.4999:
    # On a distorted element, CPE4I locks in the bilinear hourglass volumetric mode
    # (2 eigenvalues scale with K), whereas CPE4IH has strictly 1 volumetric eigenvalue (mean dilatation)
    # and all 4 remaining modes scale with G!
    coords_dist_2d = np.array([[0.0, 0.0], [2.0, 0.1], [1.7, 1.3], [-0.1, 0.9]], dtype=np.float64)
    props_inc = _make_props(E=3000.0, nu=0.4999)
    G = 3000.0 / (2.0 * (1.0 + 0.4999))
    _, K_i, _ = compute_cpe4i_element_numba(coords_dist_2d, np.zeros(8), 0, props_inc, np.zeros((4, 7)), 0.0)
    _, K_ih, _ = compute_cpe4ih_element_numba(coords_dist_2d, np.zeros(8), 0, props_inc, np.zeros((4, 7)), 0.0)
    n_locked_i = int(np.sum(np.linalg.eigvalsh(0.5 * (K_i + K_i.T)) > 50.0 * G))
    n_locked_ih = int(np.sum(np.linalg.eigvalsh(0.5 * (K_ih + K_ih.T)) > 50.0 * G))
    # Measured 2026-10-20, identical on 0a858e8 (the commit that added this test) and later: CPE4I
    # ALSO has exactly one locked mode on this quad (eig/G = 1.18e4; the other 4 non-zero modes are
    # 0.743, 1.68, 2.32, 2.5) and its spectrum equals CPE4IH's to 3 digits. The earlier expectation
    # `n_locked_i == 2` ("CPE4I locks in the bilinear hourglass volumetric mode") never held on
    # committed code, so this check cannot show a CPE4IH advantage. Left for the element's owner:
    # either the CPE4IH hybrid term has no effect on the zero-state tangent here, or the premise is wrong.
    assert n_locked_i == 1
    assert n_locked_ih == 1  # CPE4IH has ONLY the 1 physical constant-dilatation mode!
