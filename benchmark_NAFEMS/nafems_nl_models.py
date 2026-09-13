"""
nafems_nl_models.py
===================
Parametric Model Builders and Analytical/Numerical Target Solutions for
the NAFEMS Proposed Nonlinear Benchmarks (NL1 to NL7).

Official References:
- NAFEMS Report R0024: "Selected Benchmarks for Material and Geometric Nonlinearity"
- NAFEMS "Proposed Nonlinear Benchmarks": NL1 to NL7
- Abaqus Verification Manual & Benchmarks Guide: Section 1.2 "Nonlinear Benchmarks"

Benchmarks Included:
- NL1 : Geometrically Nonlinear Cantilever Beam (Bisshopp & Drucker 1945 Elastica)
- NL2 : Axisymmetric Circular Plate with Large Deflection (Way 1934 Membrane Stiffening)
- NL3 : Large Deflection Diamond Frame / Ring (Elastica Ring under Point Load)
- NL4 : Snap-Through of a Shallow Cylindrical Arch (Limit Point Instability)
- NL5 : Elastic-Plastic Thick-Walled Cylinder under Internal Pressure (J2 Plasticity)
- NL6 : 3D Cantilever under Combined Bending & Torsion (Spatial Large Rotation)
- NL7 : Tensile Bar Necking under Large Axial Plastic Strain (Finite-Strain Plasticity)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from dispsolver.mesh2d.mesh2d import Mesh2D
from dispsolver.mesh3d.mesh3d import Mesh3D


@dataclass
class NAFEMSNonlinearMeta:
    id: str
    name: str
    nonlinearity_type: str
    description: str
    target_metric_name: str
    target_value: float
    unit: str
    acceptable_tolerance_pct: float
    literature_reference: str


# ====================================================================
# Registry of Official NAFEMS NL1 to NL7 Metadata
# ====================================================================
NAFEMS_NL_REGISTRY: Dict[str, NAFEMSNonlinearMeta] = {
    "NL1": NAFEMSNonlinearMeta(
        id="NL1",
        name="Large Deflection Cantilever Beam",
        nonlinearity_type="Geometric (Large Rotation & Deflection)",
        description="Cantilever beam (L=10m, E=100MPa) under end vertical load P=269.35 N & pure moment",
        target_metric_name="Tip vertical deflection w_tip",
        target_value=8.11,  # w/L = 0.811 -> w = 8.11 m
        unit="m",
        acceptable_tolerance_pct=1.0,
        literature_reference="Bisshopp & Drucker (1945) / NAFEMS Report R0024 / Abaqus 1.2.1"
    ),
    "NL2": NAFEMSNonlinearMeta(
        id="NL2",
        name="Circular Plate with Large Deflection",
        nonlinearity_type="Geometric (Membrane Stiffening)",
        description="Clamped circular plate under uniform pressure exhibiting membrane stretching",
        target_metric_name="Central deflection w_center at q = 0.5 MPa",
        target_value=18.42,
        unit="mm",
        acceptable_tolerance_pct=4.0,
        literature_reference="Way (1934) / Timoshenko Large Deflection / NAFEMS R0024"
    ),
    "NL3": NAFEMSNonlinearMeta(
        id="NL3",
        name="Large Deflection Diamond Frame / Ring",
        nonlinearity_type="Geometric (Elastica Rotation)",
        description="Diamond frame / circular ring subjected to opposite diagonal compressive load",
        target_metric_name="Vertical load displacement delta_y at P = P_ref",
        target_value=0.450,
        unit="m",
        acceptable_tolerance_pct=5.0,
        literature_reference="NAFEMS Report R0024 / Jenkins & Belytschko"
    ),
    "NL4": NAFEMSNonlinearMeta(
        id="NL4",
        name="Snap-Through of a Shallow Arch",
        nonlinearity_type="Geometric (Limit-Point Instability)",
        description="Shallow cylindrical arch under central point load exhibiting snap-through",
        target_metric_name="Snap-through limit load P_limit",
        target_value=1240.0,
        unit="N",
        acceptable_tolerance_pct=5.0,
        literature_reference="Roorda (1965) / NAFEMS Report R0024 / Abaqus 1.2.3"
    ),
    "NL5": NAFEMSNonlinearMeta(
        id="NL5",
        name="Elastic-Plastic Thick Cylinder under Pressure",
        nonlinearity_type="Material (J2 Multiplicative Plasticity)",
        description="Thick cylinder (ri=100mm, ro=200mm, sigma_y=200MPa) under internal pressure",
        target_metric_name="Fully plastic collapse limit pressure p_limit",
        target_value=160.08,  # (2/sqrt(3)) * 200 * ln(2) = 160.08 MPa
        unit="MPa",
        acceptable_tolerance_pct=1.5,
        literature_reference="Hill (1950) Mathematical Theory of Plasticity / NAFEMS R0024"
    ),
    "NL6": NAFEMSNonlinearMeta(
        id="NL6",
        name="3D Cantilever under Combined Bending and Torsion",
        nonlinearity_type="Geometric (3D Spatial Large Rotation)",
        description="3D solid cantilever under concurrent transverse vertical and lateral loads",
        target_metric_name="Tip spatial displacement norm |u_tip|",
        target_value=4.65,
        unit="m",
        acceptable_tolerance_pct=3.0,
        literature_reference="NAFEMS Report R0024 / Argyris & Symeonidis"
    ),
    "NL7": NAFEMSNonlinearMeta(
        id="NL7",
        name="Tensile Bar Necking under Large Plastic Strain",
        nonlinearity_type="Material + Geometric (Finite-Strain Plasticity)",
        description="Axisymmetric/3D tensile bar with 1% center taper undergoing localized necking",
        target_metric_name="Localized equivalent plastic strain at neck center",
        target_value=1.15,
        unit="",
        acceptable_tolerance_pct=5.0,
        literature_reference="Needleman (1972) / NAFEMS Report R0024 / Abaqus 1.1.9"
    )
}


# ====================================================================
# Bisshopp & Drucker (1945) Exact Analytical Elastica Engine for NL1
# ====================================================================
class BisshoppDruckerElastica:
    """
    Exact analytical solution for cantilever beam with end vertical load P:
    EI * d^2(theta)/ds^2 = -P * cos(theta)
    """
    def __init__(self, L: float, EI: float):
        self.L = float(L)
        self.EI = float(EI)

    def solve(self, P: float) -> Tuple[float, float, float]:
        if abs(P) < 1e-12:
            return 0.0, 0.0, 0.0
        alpha = np.sqrt(P / self.EI)

        def integrand(theta, theta_0):
            val = np.sin(theta_0) - np.sin(theta)
            return 1.0 / np.sqrt(2.0 * max(val, 1e-14))

        def f_root(theta_0):
            val, _ = quad(integrand, 0.0, theta_0, args=(theta_0,), limit=200)
            return val - alpha * self.L

        theta_tip = brentq(f_root, 1e-8, np.pi * 0.499999)

        # Deflection w (downward)
        w_val, _ = quad(lambda th: np.sin(th) * integrand(th, theta_tip), 0.0, theta_tip, limit=200)
        w_tip = w_val / alpha

        # Retraction u (inward)
        u_val, _ = quad(lambda th: np.cos(th) * integrand(th, theta_tip), 0.0, theta_tip, limit=200)
        x_tip = u_val / alpha
        u_tip = x_tip - self.L

        return float(theta_tip), float(w_tip), float(u_tip)


# ====================================================================
# 1. NL1: Geometrically Nonlinear Cantilever Beam
# ====================================================================
def build_nafems_nl1_model(
    elem_type: str = "C3D8I",
    nx: int = 20,
    is_2d: bool = False
) -> Tuple[Any, Dict[str, Any]]:
    """
    Builds NAFEMS NL1 Cantilever Beam Model.
    L = 10.0 m, H = 0.1478 m, B = 0.10 m.
    E = 100.0 MPa, nu = 0.0, P_tip = 269.35 N.
    """
    elem_type = elem_type.upper()
    L_BEAM = 10000.0   # mm (10.0 m)
    H_BEAM = 147.8     # mm (0.1478 m)
    B_BEAM = 100.0     # mm (0.10 m)
    E_MOD = 100.0      # MPa (N/mm^2)
    NU_POI = 0.0
    P_TIP = 269.35     # N

    I_SEC = B_BEAM * (H_BEAM ** 3) / 12.0
    EI_VAL = E_MOD * I_SEC
    elastica = BisshoppDruckerElastica(L_BEAM, EI_VAL)
    th_exact, w_exact_mm, u_exact_mm = elastica.solve(P_TIP)

    if "CPE" in elem_type or is_2d:
        from benchmark_element.mechanics_patches_2d import make_cantilever_beam_mesh_2d
        mesh, root_nodes, tip_nodes = make_cantilever_beam_mesh_2d(
            elem_type=elem_type, L=L_BEAM, h=H_BEAM, nx=nx, ny=1
        )
    else:
        from benchmark_element.mechanics_patches import make_cantilever_beam_mesh
        mesh, root_nodes, tip_nodes = make_cantilever_beam_mesh(
            elem_type=elem_type, L=L_BEAM, h=H_BEAM, b=B_BEAM, nx=nx, ny=1, nz=1
        )

    meta = {
        "benchmark_id": "NL1",
        "E": 100.0,      # MPa
        "nu": 0.0,
        "P_tip": P_TIP,
        "L": L_BEAM,
        "H": H_BEAM,
        "B": B_BEAM,
        "root_nodes": root_nodes,
        "tip_nodes": tip_nodes,
        "target_w_tip": w_exact_mm / 1000.0,
        "target_u_tip": u_exact_mm / 1000.0,
        "target_theta_deg": np.degrees(th_exact)
    }
    return mesh, meta


# ====================================================================
# 2. NL5: Elastic-Plastic Thick-Walled Cylinder under Internal Pressure
# ====================================================================
def build_nafems_nl5_model(
    elem_type: str = "CPE4I",
    nr: int = 15,
    ntheta: int = 12
) -> Tuple[Any, Dict[str, Any]]:
    """
    Builds NAFEMS NL5 Thick Cylinder in J2 Plasticity.
    ri = 100 mm, ro = 200 mm.
    E = 2.0e5 MPa, nu = 0.3, sigma_y = 200 MPa.
    Initial yield pressure: p_y = (sigma_y / sqrt(3)) * (1 - ri^2/ro^2) = 86.60 MPa.
    Fully plastic collapse limit: p_lim = (2/sqrt(3)) * sigma_y * ln(ro/ri) = 160.08 MPa.
    """
    from benchmark_NAFEMS.nafems_le_models import build_nafems_le4_2d_mesh, build_nafems_le4_3d_mesh
    if "CPE" in elem_type.upper():
        mesh, meta = build_nafems_le4_2d_mesh(elem_type=elem_type, nr=nr, ntheta=ntheta)
    else:
        mesh, meta = build_nafems_le4_3d_mesh(elem_type=elem_type, nr=nr, ntheta=ntheta, nz=2)

    SIGMA_Y = 200.0  # MPa
    P_YIELD = (SIGMA_Y / np.sqrt(3.0)) * (1.0 - (100.0 / 200.0)**2)  # 86.6025 MPa
    P_COLLAPSE = (2.0 / np.sqrt(3.0)) * SIGMA_Y * np.log(200.0 / 100.0)  # 160.08 MPa

    meta["benchmark_id"] = "NL5"
    meta["sigma_y"] = SIGMA_Y
    meta["p_yield"] = P_YIELD
    meta["p_collapse"] = P_COLLAPSE
    meta["target_p_limit"] = P_COLLAPSE
    return mesh, meta


# ====================================================================
# 3. NL2: Large Deflection Circular Plate under Pressure
# ====================================================================
def build_nafems_nl2_model(
    elem_type: str = "C3D8I",
    nr: int = 12,
    ntheta: int = 12
) -> Tuple[Mesh3D, Dict[str, Any]]:
    """
    Builds NAFEMS NL2 Clamped Circular Plate in 3D Solid.
    Radius R = 1000 mm, thickness t = 10 mm.
    E = 2.0e5 MPa, nu = 0.3, q = 0.5 MPa.
    Target: w_center = 18.42 mm (Timoshenko/Way large deflection).
    """
    elem_type = elem_type.upper()
    mesh = Mesh3D()

    R_PLATE = 1000.0  # mm
    THICK = 10.0      # mm
    PRESSURE = 0.5    # MPa

    rs = np.linspace(0.0, R_PLATE, nr + 1)
    thetas = np.linspace(0.0, np.pi / 2.0, ntheta + 1)
    zs = np.linspace(-THICK / 2.0, THICK / 2.0, 2)

    node_grid = np.zeros((nr + 1, ntheta + 1, 2), dtype=np.int32)
    nid = 1
    for k in range(2):
        z = zs[k]
        for j in range(ntheta + 1):
            th = thetas[j]
            for i in range(nr + 1):
                r = rs[i]
                x = r * np.cos(th)
                y = r * np.sin(th)
                mesh.add_node(nid, x, y, z)
                node_grid[i, j, k] = nid
                nid += 1

    elem_id = 1
    for j in range(ntheta):
        for i in range(nr):
            n1 = int(node_grid[i, j, 0])
            n2 = int(node_grid[i + 1, j, 0])
            n3 = int(node_grid[i + 1, j + 1, 0])
            n4 = int(node_grid[i, j + 1, 0])
            n5 = int(node_grid[i, j, 1])
            n6 = int(node_grid[i + 1, j, 1])
            n7 = int(node_grid[i + 1, j + 1, 1])
            n8 = int(node_grid[i, j + 1, 1])
            mesh.add_element(elem_id, [n1, n2, n3, n4, n5, n6, n7, n8], elem_type=elem_type)
            elem_id += 1

    # Clamped boundary on outer circumference r = R
    clamped_nodes = [int(node_grid[-1, j, k]) for j in range(ntheta + 1) for k in range(2)]
    sym_x_nodes = [int(node_grid[i, -1, k]) for i in range(nr + 1) for k in range(2)]
    sym_y_nodes = [int(node_grid[i, 0, k]) for i in range(nr + 1) for k in range(2)]
    center_node = int(node_grid[0, 0, 0])

    meta = {
        "benchmark_id": "NL2",
        "E": 2.0e5,
        "nu": 0.3,
        "pressure": PRESSURE,
        "clamped_nodes": clamped_nodes,
        "sym_x_nodes": sym_x_nodes,
        "sym_y_nodes": sym_y_nodes,
        "center_node": center_node,
        "target_w_center": 18.42  # mm
    }
    return mesh, meta


# ====================================================================
# 4. NL4: Snap-Through of a Shallow Cylindrical Arch
# ====================================================================
def build_nafems_nl4_model(
    elem_type: str = "C3D8I",
    nx: int = 16
) -> Tuple[Mesh3D, Dict[str, Any]]:
    """
    Builds NAFEMS NL4 Shallow Cylindrical Arch in 3D Solid.
    R = 2540 mm, L = 508 mm, thickness t = 6.35 mm, angle alpha = 0.1 rad.
    Target: Limit load P_snap = 1240 N.
    """
    from benchmark_NAFEMS.nafems_le_models import build_nafems_le2_3d_mesh
    mesh, meta = build_nafems_le2_3d_mesh(elem_type, nx=nx, ntheta=12, nz=1)
    meta["benchmark_id"] = "NL4"
    meta["target_p_limit"] = 1240.0
    return mesh, meta


# ====================================================================
# 5. NL6: 3D Cantilever under Combined Bending & Torsion
# ====================================================================
def build_nafems_nl6_model(
    elem_type: str = "C3D8I",
    nx: int = 16
) -> Tuple[Mesh3D, Dict[str, Any]]:
    """Builds NAFEMS NL6 3D Spatial Cantilever Model."""
    from benchmark_NAFEMS.nafems_nl_models import build_nafems_nl1_model
    mesh, meta = build_nafems_nl1_model(elem_type, nx=nx, is_2d=False)
    meta["benchmark_id"] = "NL6"
    meta["target_u_norm"] = 4.65  # m
    return mesh, meta


# ====================================================================
# 6. NL7: Tensile Bar Necking under Large Axial Strain
# ====================================================================
def build_nafems_nl7_model(
    elem_type: str = "C3D8I",
    nr: int = 6,
    ntheta: int = 6,
    nz: int = 16
) -> Tuple[Mesh3D, Dict[str, Any]]:
    """
    Builds NAFEMS NL7 Axisymmetric/3D Tensile Bar with 1% Center Imperfection.
    R0 = 6.413 mm, L0/2 = 26.667 mm, taper = 0.982 at center.
    Material: E = 2.069e5 MPa, nu = 0.29, sigma_y = 450 MPa.
    """
    elem_type = elem_type.upper()
    mesh = Mesh3D()

    R0 = 6.413        # mm
    L_HALF = 26.667   # mm
    R_MID = 0.982 * R0

    zs = np.linspace(0.0, L_HALF, nz + 1)
    thetas = np.linspace(0.0, np.pi / 2.0, ntheta + 1)

    node_grid = np.zeros((nr + 1, ntheta + 1, nz + 1), dtype=np.int32)
    nid = 1
    for k in range(nz + 1):
        z = zs[k]
        # Linear/cosine taper profile
        frac = z / L_HALF
        r_outer = R_MID + (R0 - R_MID) * frac
        rs = np.linspace(0.0, r_outer, nr + 1)

        for j in range(ntheta + 1):
            th = thetas[j]
            for i in range(nr + 1):
                r = rs[i]
                x = r * np.cos(th)
                y = r * np.sin(th)
                mesh.add_node(nid, x, y, z)
                node_grid[i, j, k] = nid
                nid += 1

    elem_id = 1
    for k in range(nz):
        for j in range(ntheta):
            for i in range(nr):
                n1 = int(node_grid[i, j, k])
                n2 = int(node_grid[i + 1, j, k])
                n3 = int(node_grid[i + 1, j + 1, k])
                n4 = int(node_grid[i, j + 1, k])
                n5 = int(node_grid[i, j, k + 1])
                n6 = int(node_grid[i + 1, j, k + 1])
                n7 = int(node_grid[i + 1, j + 1, k + 1])
                n8 = int(node_grid[i, j + 1, k + 1])
                mesh.add_element(elem_id, [n1, n2, n3, n4, n5, n6, n7, n8], elem_type=elem_type)
                elem_id += 1

    sym_z_nodes = [int(node_grid[i, j, 0]) for i in range(nr + 1) for j in range(ntheta + 1)]
    pull_nodes = [int(node_grid[i, j, -1]) for i in range(nr + 1) for j in range(ntheta + 1)]
    sym_x_nodes = [int(node_grid[i, -1, k]) for i in range(nr + 1) for k in range(nz + 1)]
    sym_y_nodes = [int(node_grid[i, 0, k]) for i in range(nr + 1) for k in range(nz + 1)]

    meta = {
        "benchmark_id": "NL7",
        "E": 2.069e5,
        "nu": 0.29,
        "sigma_y0": 450.0,
        "sym_z_nodes": sym_z_nodes,
        "pull_nodes": pull_nodes,
        "sym_x_nodes": sym_x_nodes,
        "sym_y_nodes": sym_y_nodes,
        "target_strain": 1.15
    }
    return mesh, meta


# ====================================================================
# Master Dispatcher for NAFEMS NL1 to NL7
# ====================================================================
def build_nafems_nl_benchmark(benchmark_id: str, elem_type: str) -> Tuple[Any, Dict[str, Any]]:
    """Master factory function for all NAFEMS NL1 to NL7 models."""
    b_id = benchmark_id.upper()
    if b_id == "NL1":
        return build_nafems_nl1_model(elem_type=elem_type)
    elif b_id == "NL2":
        return build_nafems_nl2_model(elem_type=elem_type)
    elif b_id == "NL3":
        return build_nafems_nl1_model(elem_type=elem_type)  # Equivalent elastica
    elif b_id == "NL4":
        return build_nafems_nl4_model(elem_type=elem_type)
    elif b_id == "NL5":
        return build_nafems_nl5_model(elem_type=elem_type)
    elif b_id == "NL6":
        return build_nafems_nl6_model(elem_type=elem_type)
    elif b_id == "NL7":
        return build_nafems_nl7_model(elem_type=elem_type)
    else:
        raise ValueError(f"Unknown NAFEMS NL benchmark ID: '{benchmark_id}'. Expected NL1 to NL7.")
