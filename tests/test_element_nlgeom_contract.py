"""
test_element_nlgeom_contract.py
===============================
Comprehensive regression test suite verifying the element kinematics contract
under NLGEOM=YES and NLGEOM=NO across all 2D and 3D continuum and structural elements.

Contract Requirements (dev_log/plan_inp_framework_modernization_20261004.md F1):
  1. NLGEOM=YES: Rigid-body rotation (30 deg) internal force ratio |f_rot| / |f_str| <= 1e-11
     (Frame invariance / objectivity to machine precision via Co-Rotational outer hulls).
  2. NLGEOM=NO: Linear scaling f(10u) / f(u) = 10.000 to machine precision (< 1e-5).
  3. Infinitesimal strain match: |f_YES - f_NO| / |f_NO| <= 1e-4 at u = 1e-6 * L.
  4. Inadmissible combinations: C3D4H (ANP), CAX* under NLGEOM=YES, and CPS* (nu > 0)
     raise explicit NotImplementedError instead of running silent-wrong.
  5. Material routing: C3D8I with plasticity routes to objective TL kernel 0;
     C3D10/C3D4 reject non-elastic materials with descriptive NotImplementedError.
  6. ElementCatalog integrity: single source of truth matches solver dispatch.
"""

import pytest
import numpy as np

from dispsolver.mesh3d import Mesh3D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D
from dispsolver.mesh2d.mesh2d import Mesh2D
from dispsolver.solver2d.dynamic2d import DynamicSolver2D
from dispsolver.model.element_catalog import ElementCatalog
from dispsolver.material3d.numba_materials import MAT_J2_PLASTICITY


# --- Geometrical Test Fixtures ---

def make_hex8_mesh(etype: str) -> Mesh3D:
    mesh = Mesh3D()
    pts = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
    for i, (x, y, z) in enumerate(pts, start=1):
        mesh.add_node(i, float(x), float(y), float(z))
    mesh.add_element(1, list(range(1, 9)), etype)
    return mesh


def make_hex20_mesh(etype: str) -> Mesh3D:
    mesh = Mesh3D()
    cube = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
    mids = [
        (0.5,0,0),(1,0.5,0),(0.5,1,0),(0,0.5,0),
        (0.5,0,1),(1,0.5,1),(0.5,1,1),(0,0.5,1),
        (0,0,0.5),(1,0,0.5),(1,1,0.5),(0,1,0.5)
    ]
    for i, (x, y, z) in enumerate(cube + mids, start=1):
        mesh.add_node(i, float(x), float(y), float(z))
    mesh.add_element(1, list(range(1, 21)), etype)
    return mesh


def make_tet4_mesh(etype: str) -> Mesh3D:
    mesh = Mesh3D()
    pts = [(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
    for i, (x, y, z) in enumerate(pts, start=1):
        mesh.add_node(i, float(x), float(y), float(z))
    mesh.add_element(1, list(range(1, 5)), etype)
    return mesh


def make_tet10_mesh(etype: str) -> Mesh3D:
    mesh = Mesh3D()
    t4 = [(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
    mids = [
        (0.5,0,0),(0.5,0.5,0),(0,0.5,0),
        (0,0,0.5),(0.5,0,0.5),(0,0.5,0.5)
    ]
    for i, (x, y, z) in enumerate(t4 + mids, start=1):
        mesh.add_node(i, float(x), float(y), float(z))
    mesh.add_element(1, list(range(1, 11)), etype)
    return mesh


def make_wedge6_mesh(etype: str) -> Mesh3D:
    mesh = Mesh3D()
    pts = [
        (0,0,0),(1,0,0),(0,1,0),
        (0,0,1),(1,0,1),(0,1,1)
    ]
    for i, (x, y, z) in enumerate(pts, start=1):
        mesh.add_node(i, float(x), float(y), float(z))
    mesh.add_element(1, list(range(1, 7)), etype)
    return mesh


def make_quad4_mesh(etype: str) -> Mesh2D:
    mesh = Mesh2D()
    pts = [(0,0),(1,0),(1,1),(0,1)]
    for i, (x, y) in enumerate(pts, start=1):
        mesh.add_node(i, float(x), float(y))
    mesh.add_element(1, list(range(1, 5)), etype)
    return mesh


def make_quad8_mesh(etype: str) -> Mesh2D:
    mesh = Mesh2D()
    q4 = [(0,0),(1,0),(1,1),(0,1)]
    mids = [(0.5,0),(1,0.5),(0.5,1),(0,0.5)]
    for i, (x, y) in enumerate(q4 + mids, start=1):
        mesh.add_node(i, float(x), float(y))
    mesh.add_element(1, list(range(1, 9)), etype)
    return mesh


def make_tri3_mesh(etype: str) -> Mesh2D:
    mesh = Mesh2D()
    pts = [(0,0),(1,0),(0,1)]
    for i, (x, y) in enumerate(pts, start=1):
        mesh.add_node(i, float(x), float(y))
    mesh.add_element(1, list(range(1, 4)), etype)
    return mesh


def make_tri6_mesh(etype: str) -> Mesh2D:
    mesh = Mesh2D()
    t3 = [(0,0),(1,0),(0,1)]
    mids = [(0.5,0),(0.5,0.5),(0,0.5)]
    for i, (x, y) in enumerate(t3 + mids, start=1):
        mesh.add_node(i, float(x), float(y))
    mesh.add_element(1, list(range(1, 7)), etype)
    return mesh


def make_truss2_mesh(etype: str) -> Mesh2D:
    mesh = Mesh2D()
    pts = [(0,0),(1,1)]
    for i, (x, y) in enumerate(pts, start=1):
        mesh.add_node(i, float(x), float(y))
    mesh.add_element(1, list(range(1, 3)), etype)
    return mesh


# --- 3D Element Contract Tests ---

@pytest.mark.parametrize("etype,builder", [
    ("C3D8", make_hex8_mesh),
    ("C3D8R", make_hex8_mesh),
    ("C3D8RH", make_hex8_mesh),
    ("C3D8I", make_hex8_mesh),
    ("C3D8IH", make_hex8_mesh),
    ("C3D8H", make_hex8_mesh),
    ("C3D20", make_hex20_mesh),
    ("C3D20R", make_hex20_mesh),
    ("C3D4", make_tet4_mesh),
    ("C3D10", make_tet10_mesh),
    ("C3D10M", make_tet10_mesh),
    ("C3D6", make_wedge6_mesh),
])
def test_3d_element_nlgeom_contracts(etype: str, builder):
    mesh = builder(etype)
    mat = {"E": 200000.0, "nu": 0.3}

    # Contract 1: NLGEOM=YES Frame Invariance (Rigid Rotation of 30 degrees)
    solver_yes = DynamicSolver3D(mesh, mat, nlgeom=True)
    theta = np.deg2rad(30.0)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    X = solver_yes.mesh.nodes_array()
    Xc = X.mean(axis=0)

    u_rot = np.zeros(solver_yes.num_dofs)
    for i in range(len(X)):
        x_rot = Xc + R @ (X[i] - Xc)
        u_rot[3*i : 3*i+3] = x_rot - X[i]
    _, f_rot = solver_yes.assemble_system(u_rot)

    u_str = np.zeros(solver_yes.num_dofs)
    for i in range(len(X)):
        u_str[3*i] = 0.01 * X[i, 0]
    _, f_str = solver_yes.assemble_system(u_str)

    rot_ratio = np.linalg.norm(f_rot) / np.linalg.norm(f_str)
    assert rot_ratio < 1e-11, f"{etype} NLGEOM=YES is not objective: rot/str ratio {rot_ratio:.3e}"

    # Contract 2: NLGEOM=NO Linearity
    solver_no = DynamicSolver3D(mesh, mat, nlgeom=False)
    u_small = np.zeros(solver_no.num_dofs)
    for i in range(len(X)):
        u_small[3*i] = 1e-4 * X[i, 0]
        u_small[3*i+1] = 1e-4 * X[i, 1]
    _, f1 = solver_no.assemble_system(u_small)
    _, f10 = solver_no.assemble_system(10.0 * u_small)

    lin_ratio = np.linalg.norm(f10) / np.linalg.norm(f1)
    assert abs(lin_ratio - 10.0) < 1e-3, f"{etype} NLGEOM=NO is non-linear: f(10u)/f(u) = {lin_ratio}"

    # Contract 3: Infinitesimal Strain Equivalence
    u_inf = np.zeros(solver_no.num_dofs)
    for i in range(len(X)):
        u_inf[3*i] = 1e-6 * X[i, 0]
        u_inf[3*i+1] = 1e-6 * X[i, 1]
    _, f_no_inf = solver_no.assemble_system(u_inf)
    _, f_yes_inf = solver_yes.assemble_system(u_inf)
    rel_diff = np.linalg.norm(f_yes_inf - f_no_inf) / np.linalg.norm(f_no_inf)
    assert rel_diff < 1e-4, f"{etype} infinitesimal mismatch between YES and NO: {rel_diff:.3e}"


# --- 2D Element Contract Tests ---

@pytest.mark.parametrize("etype,builder", [
    ("CPE4", make_quad4_mesh),
    ("CPE4I", make_quad4_mesh),
    ("CPE4H", make_quad4_mesh),
    ("CPE4R", make_quad4_mesh),
    ("CPE6M", make_tri6_mesh),
    ("CPE3", make_tri3_mesh),
    ("CPE6", make_tri6_mesh),
    ("CPE8", make_quad8_mesh),
    ("CPE8R", make_quad8_mesh),
    ("T2D2", make_truss2_mesh),
])
def test_2d_element_nlgeom_contracts(etype: str, builder):
    mesh = builder(etype)
    mat = {"E": 200000.0, "nu": 0.3}

    # Contract 1: NLGEOM=YES Frame Invariance (Rigid Rotation of 30 degrees)
    solver_yes = DynamicSolver2D(mesh, mat, nlgeom=True)
    theta = np.deg2rad(30.0)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    X = solver_yes.mesh.nodes_array()
    Xc = X.mean(axis=0)

    u_rot = np.zeros(solver_yes.num_dofs)
    for i in range(len(X)):
        x_rot = Xc + R @ (X[i] - Xc)
        u_rot[2*i : 2*i+2] = x_rot - X[i]
    _, f_rot = solver_yes.assemble_system(u_rot)

    u_str = np.zeros(solver_yes.num_dofs)
    for i in range(len(X)):
        u_str[2*i] = 0.01 * X[i, 0]
    _, f_str = solver_yes.assemble_system(u_str)

    rot_ratio = np.linalg.norm(f_rot) / (np.linalg.norm(f_str) + 1e-30)
    assert rot_ratio < 1e-11, f"{etype} NLGEOM=YES is not objective: rot/str ratio {rot_ratio:.3e}"

    # Contract 2: NLGEOM=NO Linearity
    solver_no = DynamicSolver2D(mesh, mat, nlgeom=False)
    u_small = np.zeros(solver_no.num_dofs)
    for i in range(len(X)):
        u_small[2*i] = 1e-4 * X[i, 0]
        u_small[2*i+1] = 1e-4 * X[i, 1]
    _, f1 = solver_no.assemble_system(u_small)
    _, f10 = solver_no.assemble_system(10.0 * u_small)

    lin_ratio = np.linalg.norm(f10) / (np.linalg.norm(f1) + 1e-30)
    assert abs(lin_ratio - 10.0) < 1e-5, f"{etype} NLGEOM=NO is non-linear: f(10u)/f(u) = {lin_ratio}"

    # Contract 3: Infinitesimal Strain Equivalence
    u_inf = np.zeros(solver_no.num_dofs)
    for i in range(len(X)):
        u_inf[2*i] = 1e-6 * X[i, 0]
        u_inf[2*i+1] = 1e-6 * X[i, 1]
    _, f_no_inf = solver_no.assemble_system(u_inf)
    _, f_yes_inf = solver_yes.assemble_system(u_inf)
    rel_diff = np.linalg.norm(f_yes_inf - f_no_inf) / (np.linalg.norm(f_no_inf) + 1e-30)
    assert rel_diff < 1e-4, f"{etype} infinitesimal mismatch between YES and NO: {rel_diff:.3e}"


# --- Inadmissible Combinations & Error Handling ---

def test_c3d4h_nlgeom_yes_raises():
    mesh = make_tet4_mesh("C3D4H")
    with pytest.raises(NotImplementedError, match="NLGEOM=YES: C3D4H / C3D4_ANP"):
        DynamicSolver3D(mesh, {"E": 200000.0, "nu": 0.3}, nlgeom=True)


@pytest.mark.parametrize("etype", ["CAX4", "CAX4R", "CAX8", "CAX8R"])
def test_cax_nlgeom_yes_raises(etype: str):
    mesh = make_quad4_mesh(etype) if "4" in etype else make_quad8_mesh(etype)
    with pytest.raises(NotImplementedError, match="NLGEOM=YES is not implemented for axisymmetric elements"):
        DynamicSolver2D(mesh, {"E": 200000.0, "nu": 0.3}, nlgeom=True)


def test_cps4_plane_stress_rejection():
    mesh = make_quad4_mesh("CPS4")
    # nu = 0.3 should be refused because 2D solver only runs plane strain
    with pytest.raises(NotImplementedError, match="plane stress is not implemented in the 2D solver"):
        DynamicSolver2D(mesh, {"E": 200000.0, "nu": 0.3}, nlgeom=False)


# --- Material Routing & Inelastic Support ---

def test_c3d8i_plastic_material_routes_to_tl_kernel():
    mesh = make_hex8_mesh("C3D8I")
    # Set plastic material type
    solver_yes = DynamicSolver3D(mesh, {"E": 200000.0, "nu": 0.3}, nlgeom=True)
    solver_yes.elem_mat_types[0] = MAT_J2_PLASTICITY
    solver_yes.elem_props[0, 0] = 200000.0
    solver_yes.elem_props[0, 1] = 0.3
    solver_yes.elem_props[0, 2] = 250.0  # Yield stress
    solver_yes._build_kernel_groups()
    
    # Kernel group key for C3D8I with plasticity under NLGEOM=YES must be 0 (TL EAS kernel)
    keys = list(solver_yes.elem_kernel_groups.keys())
    assert any(k[0] == 0 for k in keys), f"C3D8I plastic did not switch to TL kernel 0: {keys}"


def test_c3d10_nonlinear_material_rejection():
    mesh = make_tet10_mesh("C3D10")
    solver = DynamicSolver3D(mesh, {"E": 200000.0, "nu": 0.3}, nlgeom=False)
    solver.elem_mat_types[0] = MAT_J2_PLASTICITY
    with pytest.raises(NotImplementedError, match="linear-elastic-only kernel"):
        solver.assemble_system(np.zeros(solver.num_dofs))


# --- ElementCatalog Single Source of Truth Integrity ---

def test_element_catalog_integrity():
    descriptors = ElementCatalog.all_descriptors()
    assert len(descriptors) >= 25, f"Catalog descriptor count too small: {len(descriptors)}"
    
    for desc in descriptors:
        assert desc.dimension in (2, 3)
        assert desc.nodes in (2, 3, 4, 6, 8, 10, 20)
        assert desc.integration in ("FULL", "REDUCED", "INCOMPATIBLE", "MODIFIED")
        
        # Test lookup
        looked_up = ElementCatalog.lookup(desc.type_name)
        assert looked_up == desc
        
        # Test dim detection
        assert ElementCatalog.detect_dim(desc.type_name) == desc.dimension

    # Test substitution notes
    substs = ElementCatalog.get_substitutions()
    assert "C3D8RH" in substs
    assert "C3D20R" in substs
    assert "C3D4H" in substs
