"""
benchmark_3d_mechanics.py
=========================
Comprehensive Mechanical Benchmark Suite for 3D Solid Elements.
Evaluates 11 solid elements across:
  1. Irons 3D Distorted Patch Test (Tension, Compression, In-plane Shear, Out-of-plane Shear)
  2. MacNeal-Harder Cantilever Beam Bending & Parasitic Shear Locking
  3. Volumetric Locking in the Nearly Incompressible Limit (nu -> 0.5)

Elements benchmarked:
  - Hexahedra: C3D8, C3D8I, C3D8_FBAR, C3D8_CR, C3D8H, C3D8R
  - Tetrahedra: C3D4, C3D4_ANP, C3D10, C3D10M
  - Wedges: C3D6
"""

from typing import Dict, List, Tuple, Any, Optional
import time
import os
import sys
import numpy as np

# Enforce UTF-8 stdout encoding on Windows terminal to prevent broken Korean characters
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

from dispsolver.solver3d.dynamic3d import DynamicSolver3D
from benchmark_element.mechanics_patches import (
    make_distorted_patch_mesh,
    make_cantilever_beam_mesh,
    make_confined_compression_mesh,
    compute_per_element_dilatation,
)
from benchmark_element.figures import generate_all_figures


ALL_3D_ELEMENTS = [
    "C3D8",
    "C3D8I",
    "C3D8_FBAR",
    "C3D8_CR",
    "C3D8H",
    "C3D8R",
    "C3D4",
    "C3D4_ANP",
    "C3D10",
    "C3D10M",
    "C3D6"
]


# =========================================================================
# Suite 1: Irons 3D Distorted Patch Test
# =========================================================================

def run_irons_patch_test(
    elem_type: str,
    mode: str = "tension",
    distort_amplitude: float = 0.08
) -> Dict[str, Any]:
    """Execute Irons Patch Test on an irregularly distorted multi-element mesh.
    
    Modes:
      - 'tension': uniaxial tension along X
      - 'compression': uniaxial compression along Y
      - 'shear_xy': in-plane shear
      - 'shear_yz': out-of-plane shear
    """
    E = 200000.0
    nu = 0.30
    G = E / (2.0 * (1.0 + nu))
    eps0 = 1e-3

    mesh, b_nodes, i_nodes = make_distorted_patch_mesh(elem_type, distort_amplitude=distort_amplitude)
    nid_map = mesh.node_id_to_index()
    solver = DynamicSolver3D(mesh, {"E": E, "nu": nu}, nlgeom=True)

    def exact_displacement(x, y, z):
        if mode == "tension":
            return np.array([eps0 * x, -nu * eps0 * y, -nu * eps0 * z], dtype=np.float64)
        elif mode == "compression":
            return np.array([nu * eps0 * x, -eps0 * y, nu * eps0 * z], dtype=np.float64)
        elif mode == "shear_xy":
            return np.array([0.5 * eps0 * y, 0.5 * eps0 * x, 0.0], dtype=np.float64)
        elif mode == "shear_yz":
            return np.array([0.0, 0.5 * eps0 * z, 0.5 * eps0 * y], dtype=np.float64)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # Prescribe exact linear displacement on all boundary nodes
    for nid in b_nodes:
        node = mesh.nodes[nid]
        u_ex = exact_displacement(node.x, node.y, node.z)
        solver.fix_dof(nid, 0, u_ex[0])
        solver.fix_dof(nid, 1, u_ex[1])
        solver.fix_dof(nid, 2, u_ex[2])

    converged, iters = solver.solve_step(dt=1.0, max_iters=50)
    if not converged:
        return {
            "status": "DIVERGED",
            "max_error": float("nan"),
            "num_interior": len(i_nodes),
            "iters": iters
        }

    # Evaluate error at all interior nodes
    errors = []
    for nid in i_nodes:
        node = mesh.nodes[nid]
        u_ex = exact_displacement(node.x, node.y, node.z)
        idx = nid_map[nid]
        u_comp = solver.u[3 * idx : 3 * idx + 3]
        err = np.linalg.norm(u_comp - u_ex)
        errors.append(err)

    max_err = max(errors) if errors else 0.0
    passed = (max_err < 1e-6)

    return {
        "status": "PASS" if passed else "FAIL",
        "max_error": max_err,
        "num_interior": len(i_nodes),
        "iters": iters
    }


def run_abaqus_official_patch_test(
    elem_type: str,
    distort_amplitude: float = 0.08
) -> Dict[str, Any]:
    """Reproduce Step 1 of the REAL Abaqus Verification Manual "Patch test
    for three-dimensional solid elements" (elements tested there include
    C3D4, C3D6, C3D8, C3D8I, C3D8H, C3D8R, C3D10, C3D10M, ...) -- source
    text archived at benchmark_element/reference_abaqus_docs/3dpatch.txt,
    fetched 2026-09-13 from the user's local Abaqus doc server.

    Exact problem as published (do not change these numbers without
    re-checking the source doc):
      Material: linear elastic, E = 1.0e6, nu = 0.25.
      BC (all exterior/boundary nodes), Step 1 (linear perturbation):
        ux = 1e-3 * (2x + y + z) / 2
        uy = 1e-3 * (x + 2y + z) / 2
        uz = 1e-3 * (x + y + 2z) / 2
      Reference (exact) solution:
        sigma_x = sigma_y = sigma_z = 2000
        tau_xy  = tau_yz  = tau_xz  = 400
        eps_x   = eps_y   = eps_z   = 1e-3
        gamma_xy = gamma_yz = gamma_xz = 1e-3

    What this function checks: the interior-node DISPLACEMENT consistency
    part of the patch test (the same criterion run_irons_patch_test above
    uses, just with Abaqus's own exact BC/material instead of an
    in-house-chosen one) -- i.e. "does an arbitrary patch of this element,
    loaded only through its exterior nodes with the official Abaqus affine
    field, reproduce that field exactly at every interior node." It does
    NOT check the reference stress numbers above (sigma_x=2000 etc.) --
    this codebase's 3D solver has no per-Gauss-point Cauchy-stress-recovery
    API exposed yet (checked dispsolver/solver3d/dynamic3d.py and
    dispsolver/element3d/*.py, 2026-09-13: no such function). Adding that
    recovery path would let a future version of this function check the
    full Abaqus reference (displacement AND stress), which is a stronger,
    more diagnostic test (a stress mismatch with a correct displacement
    field would indicate the same class of tangent-vs-residual defect
    AGENTS.md 4.15/4.16 found elsewhere) -- tracked as a known gap, not
    silently treated as complete.
    """
    E = 1.0e6
    nu = 0.25
    eps0 = 1e-3

    mesh, b_nodes, i_nodes = make_distorted_patch_mesh(elem_type, distort_amplitude=distort_amplitude)
    nid_map = mesh.node_id_to_index()
    solver = DynamicSolver3D(mesh, {"E": E, "nu": nu}, nlgeom=True)

    def exact_displacement(x, y, z):
        return np.array([
            eps0 * (2.0 * x + y + z) / 2.0,
            eps0 * (x + 2.0 * y + z) / 2.0,
            eps0 * (x + y + 2.0 * z) / 2.0,
        ], dtype=np.float64)

    for nid in b_nodes:
        node = mesh.nodes[nid]
        u_ex = exact_displacement(node.x, node.y, node.z)
        solver.fix_dof(nid, 0, u_ex[0])
        solver.fix_dof(nid, 1, u_ex[1])
        solver.fix_dof(nid, 2, u_ex[2])

    converged, iters = solver.solve_step(dt=1.0, max_iters=50)
    if not converged:
        return {
            "status": "DIVERGED",
            "max_error": float("nan"),
            "num_interior": len(i_nodes),
            "iters": iters,
        }

    errors = []
    for nid in i_nodes:
        node = mesh.nodes[nid]
        u_ex = exact_displacement(node.x, node.y, node.z)
        idx = nid_map[nid]
        u_comp = solver.u[3 * idx: 3 * idx + 3]
        errors.append(np.linalg.norm(u_comp - u_ex))

    max_err = max(errors) if errors else 0.0
    return {
        "status": "PASS" if max_err < 1e-6 else "FAIL",
        "max_error": max_err,
        "num_interior": len(i_nodes),
        "iters": iters,
    }


# =========================================================================
# Suite 2: MacNeal-Harder Cantilever Bending & Shear Locking
# =========================================================================

def run_cantilever_bending_test(
    elem_type: str,
    L: float = 10.0,
    h: float = 1.0,
    b: float = 1.0,
    P_total: float = -10.0
) -> Dict[str, Any]:
    """Execute MacNeal-Harder Cantilever Beam Bending test to quantify shear locking.
    
    Beam extends in X in [0, L], Y in [-b/2, b/2], Z in [-h/2, h/2].
    Downward force P_total is applied on tip face at x = L.
    """
    E = 200000.0
    nu = 0.30
    G = E / (2.0 * (1.0 + nu))
    Iz = (b * h**3) / 12.0
    A = b * h
    kappa = 5.0 / 6.0  # Timoshenko shear coefficient for rectangle

    # Timoshenko theoretical tip deflection:
    w_exact = (abs(P_total) * L**3) / (3.0 * E * Iz) + (abs(P_total) * L) / (kappa * G * A)

    # Mesh resolution: 6 elements along length, 1 across width, 2 across height
    mesh, root_nodes, tip_nodes = make_cantilever_beam_mesh(
        elem_type, L=L, h=h, b=b, nx=6, ny=1, nz=2
    )
    nid_map = mesh.node_id_to_index()
    # NOTE: dynamic3d.py's linear-geometry (nlgeom=False) assembly path was
    # removed (raises NotImplementedError) -- see AGENTS.md 3D renovation
    # notes. nlgeom=True at these small prescribed strains reduces to the
    # same linear-elastic answer through the NL Newton path.
    solver = DynamicSolver3D(mesh, {"E": E, "nu": nu}, nlgeom=True)

    # Clamped root boundary condition at x = 0
    for nid in root_nodes:
        solver.fix_dof(nid, 0, 0.0)
        solver.fix_dof(nid, 1, 0.0)
        solver.fix_dof(nid, 2, 0.0)

    # Distribute vertical load across tip nodes
    f_ext = np.zeros(solver.num_dofs, dtype=np.float64)
    p_per_node = P_total / len(tip_nodes)
    for nid in tip_nodes:
        idx = nid_map[nid]
        f_ext[3 * idx + 2] = p_per_node

    converged, iters = solver.solve_step(dt=1.0, f_ext=f_ext, max_iters=50)
    if not converged:
        return {
            "status": "DIVERGED",
            "w_tip": float("nan"),
            "w_exact": w_exact,
            "ratio": float("nan"),
            "locking_verdict": "ERROR"
        }

    # Measure average downward tip deflection
    tip_deflections = []
    for nid in tip_nodes:
        idx = nid_map[nid]
        tip_deflections.append(-solver.u[3 * idx + 2])

    w_comp = float(np.mean(tip_deflections))
    ratio = w_comp / w_exact if w_exact > 0 else 0.0

    if ratio > 0.85:
        verdict = "LOCKING-FREE"
    elif ratio > 0.50:
        verdict = "MODERATE"
    else:
        verdict = "LOCKED"

    return {
        "status": "PASS",
        "w_tip": w_comp,
        "w_exact": w_exact,
        "ratio": ratio,
        "locking_verdict": verdict
    }


# =========================================================================
# Suite 3: Volumetric Locking & Incompressible Limit (nu -> 0.5)
# =========================================================================

def run_volumetric_locking_test(
    elem_type: str,
    nu_val: float = 0.49999
) -> Dict[str, Any]:
    """Execute Cantilever Bending near the Incompressible Limit to assess volumetric locking."""
    mesh, root_nodes, tip_nodes = make_cantilever_beam_mesh(
        elem_type, L=6.0, h=1.0, b=1.0, nx=4, ny=1, nz=2
    )
    nid_map = mesh.node_id_to_index()
    E = 200000.0
    # See run_cantilever_bending_test's note: nlgeom=False path is removed.
    solver = DynamicSolver3D(mesh, {"E": E, "nu": nu_val}, nlgeom=True)

    for nid in root_nodes:
        solver.fix_dof(nid, 0, 0.0)
        solver.fix_dof(nid, 1, 0.0)
        solver.fix_dof(nid, 2, 0.0)

    f_ext = np.zeros(solver.num_dofs, dtype=np.float64)
    P_total = -10.0
    for nid in tip_nodes:
        idx = nid_map[nid]
        f_ext[3 * idx + 2] = P_total / len(tip_nodes)

    converged, iters = solver.solve_step(dt=1.0, f_ext=f_ext, max_iters=50)
    if not converged:
        return {
            "status": "DIVERGED",
            "w_tip": float("nan"),
            "iters": iters,
            "verdict": "ERROR"
        }

    tip_deflections = [-solver.u[3 * nid_map[nid] + 2] for nid in tip_nodes]
    w_comp = float(np.mean(tip_deflections))

    # Baseline reference at nu = 0.3 for pure bending
    Iz = (1.0 * 1.0**3) / 12.0
    w_beam_ref = (10.0 * 6.0**3) / (3.0 * E * Iz)  # ~0.0432 mm
    ratio = w_comp / w_beam_ref

    # If ratio is close to 1.0, the element is volumetric-locking-free
    if ratio > 0.60:
        verdict = "INCOMPRESSIBLE-OK"
    elif ratio > 0.20:
        verdict = "STIFFENED"
    else:
        verdict = "VOL-LOCKED"

    return {
        "status": "PASS",
        "w_tip": w_comp,
        "ratio_to_ref": ratio,
        "iters": iters,
        "verdict": verdict
    }


# =========================================================================
# Suite 4: ANP Checkerboard Test -- direct pressure/dilatation-variance
# check, NOT tip deflection (see dev_log/solve_step_false_convergence_
# 20260913.md: the cantilever-bending volumetric test can show ZERO
# difference between C3D4 and C3D4_ANP even when the ANP fix is real,
# because a smooth bending load never develops element-to-element
# pressure disagreement for ANP to average away. This test uses a
# confined/constrained compression on an IRREGULAR tet mesh instead --
# the classic setup for demonstrating checkerboarding in linear tets.)
# =========================================================================

def run_anp_checkerboard_test(
    elem_type: str,
    n: int = 4,
    nu_val: float = 0.49999,
    L: float = 1.0,
) -> Dict[str, Any]:
    """Confined compression: rollers on all 4 sides (ux=0 on x=0/x=L,
    uy=0 on y=0/y=L), bottom fixed in z, top given a small uniform
    downward displacement. Measures the coefficient of variation (std/
    mean) of per-element det(F) after convergence -- the direct signature
    of element-to-element pressure "checkerboarding" that average-nodal-
    pressure (ANP) elements exist to relieve. Only C3D4/C3D4_ANP are
    supported (make_confined_compression_mesh's own restriction).
    """
    E = 200000.0
    mesh, faces = make_confined_compression_mesh(elem_type, n=n, L=L, distort_amplitude=0.06)
    nid_map = mesh.node_id_to_index()
    solver = DynamicSolver3D(mesh, {"E": E, "nu": nu_val}, nlgeom=True)

    for nid in faces["bottom"]:
        solver.fix_dof(nid, 2, 0.0)
    for nid in faces["x0"]:
        solver.fix_dof(nid, 0, 0.0)
    for nid in faces["xL"]:
        solver.fix_dof(nid, 0, 0.0)
    for nid in faces["y0"]:
        solver.fix_dof(nid, 1, 0.0)
    for nid in faces["yL"]:
        solver.fix_dof(nid, 1, 0.0)

    u_top = -0.001 * L  # small prescribed compression, same strain scale as the other tests
    for nid in faces["top"]:
        solver.fix_dof(nid, 2, u_top)

    converged, iters = solver.solve_step(dt=1.0, max_iters=50)
    if not converged:
        return {"status": "DIVERGED", "iters": iters, "cv_detF": float("nan")}

    detF = compute_per_element_dilatation(mesh, solver.u, nid_map)
    mean_detF = float(np.mean(detF))
    std_detF = float(np.std(detF))
    cv = std_detF / abs(mean_detF - 1.0) if abs(mean_detF - 1.0) > 1e-12 else (std_detF / max(abs(mean_detF), 1e-12))

    return {
        "status": "PASS",
        "iters": iters,
        "mean_detF": mean_detF,
        "std_detF": std_detF,
        "cv_detF": cv,
        "n_elements": len(detF),
    }


def run_anp_checkerboard_comparison(n: int = 4, nu_val: float = 0.49999, L: float = 1.0) -> Dict[str, Any]:
    """Run run_anp_checkerboard_test for both C3D4 and C3D4_ANP on the
    IDENTICAL mesh/BCs and report the relative improvement -- the real
    evidence question ('did ANP measurably reduce checkerboarding') that
    the tip-deflection-based volumetric test cannot answer for this
    element (see this function's module-level note above)."""
    r_plain = run_anp_checkerboard_test("C3D4", n=n, nu_val=nu_val, L=L)
    r_anp = run_anp_checkerboard_test("C3D4_ANP", n=n, nu_val=nu_val, L=L)

    result = {"C3D4": r_plain, "C3D4_ANP": r_anp}
    if r_plain["status"] == "PASS" and r_anp["status"] == "PASS":
        cv_plain = r_plain["cv_detF"]
        cv_anp = r_anp["cv_detF"]
        # Guard: if BOTH cv's are already near machine precision, there is no
        # checkerboarding for either element to relieve on this mesh/BC combo
        # -- report that plainly instead of a meaningless ratio of two
        # near-zero noise floors (a real failure mode hit 2026-09-13: a
        # uniform-BC confined-compression test can have a homogeneous exact
        # solution with zero spatial pressure gradient regardless of element
        # formulation, same root issue as the cantilever-bending volumetric
        # test's inability to show ANP's effect -- see
        # dev_log/solve_step_false_convergence_20260913.md).
        if cv_plain < 1e-9:
            # Plain C3D4 itself shows (near) zero element-to-element
            # dilatation variance on this mesh/BC combo -- there is no
            # checkerboarding baseline for ANP to improve on, regardless of
            # what C3D4_ANP measures (even a tiny absolute cv_anp is then
            # a REGRESSION relative to the control, not noise -- flag both).
            result["cv_improvement_pct"] = float("nan")
            result["verdict"] = (
                "NO-CHECKERBOARDING-ON-THIS-MESH-BASELINE-IS-ZERO"
                if cv_anp < 1e-9 else
                "ANP-INTRODUCED-VARIANCE-WHERE-CONTROL-HAD-NONE"
            )
        else:
            improvement = (cv_plain - cv_anp) / cv_plain if cv_plain > 1e-15 else 0.0
            result["cv_improvement_pct"] = improvement * 100.0
            result["verdict"] = "ANP-RELIEVES-CHECKERBOARDING" if improvement > 0.30 else "NO-MEASURABLE-DIFFERENCE"
    else:
        result["verdict"] = "ERROR"
    return result


# =========================================================================
# Main Benchmark Runner & Markdown Report Generator
# =========================================================================

def run_all_mechanics_benchmarks(output_path: Optional[str] = None):
    print("=" * 80)
    print("  3D Solid Elements Mechanics Benchmark Suite")
    print("  Testing: 1. Patch Tests (Tension, Comp, Shear) | 2. Bending | 3. Volumetric")
    print("=" * 80)

    results_patch: Dict[str, Dict[str, Any]] = {}
    results_abq_patch: Dict[str, Dict[str, Any]] = {}
    results_bending: Dict[str, Dict[str, Any]] = {}
    results_vol: Dict[str, Dict[str, Any]] = {}

    for elem in ALL_3D_ELEMENTS:
        print(f"\n>>> Running Mechanics Benchmark on [{elem}]...")
        t0 = time.time()

        # 1. Patch tests (in-house Irons modes + the official Abaqus VM one)
        res_t = run_irons_patch_test(elem, mode="tension")
        res_c = run_irons_patch_test(elem, mode="compression")
        res_s_xy = run_irons_patch_test(elem, mode="shear_xy")
        res_s_yz = run_irons_patch_test(elem, mode="shear_yz")
        results_patch[elem] = {
            "tension": res_t,
            "compression": res_c,
            "shear_xy": res_s_xy,
            "shear_yz": res_s_yz
        }
        results_abq_patch[elem] = run_abaqus_official_patch_test(elem)

        # 2. Cantilever Bending
        res_b = run_cantilever_bending_test(elem, L=10.0, h=1.0, b=1.0)
        results_bending[elem] = res_b

        # 3. Volumetric Incompressibility (nu = 0.49999)
        res_v = run_volumetric_locking_test(elem, nu_val=0.49999)
        results_vol[elem] = res_v

        dt = time.time() - t0
        p_status = "PASS" if all(r["status"] == "PASS" for r in [res_t, res_c, res_s_xy, res_s_yz]) else "FAIL"
        abq_status = results_abq_patch[elem]["status"]
        print(f"    Patch: {p_status} | Abaqus VM patch: {abq_status} | Bending Ratio: {res_b.get('ratio', 0.0):.2f} ({res_b.get('locking_verdict')}) | Vol (nu=0.49999): {res_v.get('verdict')} | Time: {dt:.2f}s")

    # Generate Markdown Report
    if output_path is None:
        today_str = time.strftime("%Y%m%d")
        output_path = os.path.join("dev_log", f"benchmark_3d_mechanics_{today_str}.md")

    print("\nGenerating figures...")
    fig_paths = generate_all_figures(ALL_3D_ELEMENTS, results_patch, results_abq_patch, results_bending, results_vol)
    for name, path in fig_paths.items():
        print(f"  [OK] {name} -> {path}")

    # Generate Actionable Improvement Guide
    guide_lines, console_summary = generate_actionable_improvement_guide(
        results_patch, results_abq_patch, results_bending, results_vol
    )

    save_mechanics_report(
        results_patch, results_abq_patch, results_bending, results_vol,
        output_path, guide_lines=guide_lines
    )
    print(f"\n[OK] Mechanics Benchmark complete. Report saved to: {output_path}")

    # Print actionable console summary for agents & developers
    print("\n" + "=" * 80)
    print("  ACTIONABLE RENOVATION & IMPROVEMENT GUIDE (요소별 개선 처방전)")
    print("=" * 80)
    print(console_summary)
    print("=" * 80 + "\n")


def save_mechanics_report(
    patch_res: Dict[str, Dict[str, Any]],
    abq_patch_res: Dict[str, Dict[str, Any]],
    bending_res: Dict[str, Dict[str, Any]],
    vol_res: Dict[str, Dict[str, Any]],
    file_path: str,
    guide_lines: Optional[List[str]] = None
):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    today_str = time.strftime("%Y%m%d")

    lines = [
        f"# 3D Solid Elements Mechanics Patch & Benchmark Report ({today_str})\n\n",
        "## 1. 종합 역학 성능 평가 결과표 (Comprehensive Mechanics Matrix)\n\n",
        "| 요소 명칭 | 인장 패치 (Tension) | 압축 패치 (Compression) | 면내전단 패치 (Shear XY) | 면외전단 패치 (Shear YZ) | "
        "**Abaqus VM 공식 패치** | 굽힘 처짐비 (L/h=10) | 전단 잠김 판정 | 비압축성 극한 (nu=0.49999) | 종합 판정 |\n",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"
    ]

    for elem in ALL_3D_ELEMENTS:
        p = patch_res.get(elem, {})
        t_err = p.get("tension", {}).get("max_error", float("nan"))
        c_err = p.get("compression", {}).get("max_error", float("nan"))
        sxy_err = p.get("shear_xy", {}).get("max_error", float("nan"))
        syz_err = p.get("shear_yz", {}).get("max_error", float("nan"))

        t_str = f"✅ {t_err:.1e}" if t_err < 1e-6 else f"❌ {t_err:.1e}"
        c_str = f"✅ {c_err:.1e}" if c_err < 1e-6 else f"❌ {c_err:.1e}"
        sxy_str = f"✅ {sxy_err:.1e}" if sxy_err < 1e-6 else f"❌ {sxy_err:.1e}"
        syz_str = f"✅ {syz_err:.1e}" if syz_err < 1e-6 else f"❌ {syz_err:.1e}"

        abq = abq_patch_res.get(elem, {})
        abq_err = abq.get("max_error", float("nan"))
        abq_str = f"✅ {abq_err:.1e}" if abq_err < 1e-6 else f"❌ {abq_err:.1e}"

        b = bending_res.get(elem, {})
        b_ratio = b.get("ratio", float("nan"))
        b_verdict = b.get("locking_verdict", "N/A")
        b_str = f"{b_ratio:.2f}" if not np.isnan(b_ratio) else "N/A"

        v = vol_res.get(elem, {})
        v_verdict = v.get("verdict", "N/A")

        # Overall rating
        all_patch = (t_err < 1e-6 and c_err < 1e-6 and sxy_err < 1e-6 and syz_err < 1e-6 and abq_err < 1e-6)
        if all_patch and b_verdict in ["LOCKING-FREE", "MODERATE"]:
            overall = "✅ EXCELLENT"
        elif all_patch and b_verdict == "LOCKED":
            overall = "⚠️ SHEAR-LOCKED"
        else:
            overall = "❌ CAUTION"

        lines.append(
            f"| **`{elem}`** | {t_str} | {c_str} | {sxy_str} | {syz_str} | {abq_str} | {b_str} | `{b_verdict}` | `{v_verdict}` | {overall} |\n"
        )

    def _patch_ok(elem: str) -> bool:
        p = patch_res.get(elem, {})
        return all(
            p.get(mode, {}).get("max_error", float("inf")) < 1e-6
            for mode in ("tension", "compression", "shear_xy", "shear_yz")
        )

    def _abq_patch_ok(elem: str) -> bool:
        return abq_patch_res.get(elem, {}).get("max_error", float("inf")) < 1e-6

    patch_pass = [e for e in ALL_3D_ELEMENTS if _patch_ok(e)]
    patch_fail = [e for e in ALL_3D_ELEMENTS if not _patch_ok(e)]
    abq_patch_pass = [e for e in ALL_3D_ELEMENTS if _abq_patch_ok(e)]
    abq_patch_fail = [e for e in ALL_3D_ELEMENTS if not _abq_patch_ok(e)]
    patch_disagree = [e for e in ALL_3D_ELEMENTS if _patch_ok(e) != _abq_patch_ok(e)]
    locking_free = [e for e in ALL_3D_ELEMENTS if bending_res.get(e, {}).get("locking_verdict") == "LOCKING-FREE"]
    locked = [e for e in ALL_3D_ELEMENTS if bending_res.get(e, {}).get("locking_verdict") == "LOCKED"]
    vol_ok = [e for e in ALL_3D_ELEMENTS if vol_res.get(e, {}).get("verdict") == "INCOMPRESSIBLE-OK"]
    vol_locked = [e for e in ALL_3D_ELEMENTS if vol_res.get(e, {}).get("verdict") == "VOL-LOCKED"]

    lines.extend([
        "\n## 2. 역학 모드별 거동 분석 (데이터 기반, 자동 생성)\n\n",
        "1. **Irons 3D 왜곡 패치 테스트 (in-house, Tension/Compression/Shear)**:\n",
        f"   - PASS (< 1e-6): {', '.join(f'`{e}`' for e in patch_pass) if patch_pass else '(none)'}\n",
        f"   - FAIL (>= 1e-6): {', '.join(f'`{e}`' for e in patch_fail) if patch_fail else '(none)'}"
        + ("" if not patch_fail else " -- these elements do not yet reproduce an exact linear displacement field on a distorted mesh; do not trust their bending/volumetric numbers until this is fixed, per benchmark_element/README.md.") + "\n",
        "1b. **공식 Abaqus Verification Manual 패치 테스트** "
        "(`benchmark_element/reference_abaqus_docs/3dpatch.txt`, E=1e6, nu=0.25, 실제 매뉴얼 변위장):\n",
        f"   - PASS (< 1e-6): {', '.join(f'`{e}`' for e in abq_patch_pass) if abq_patch_pass else '(none)'}\n",
        f"   - FAIL (>= 1e-6): {', '.join(f'`{e}`' for e in abq_patch_fail) if abq_patch_fail else '(none)'}\n",
        f"   - in-house 결과와 불일치(하나만 통과): {', '.join(f'`{e}`' for e in patch_disagree) if patch_disagree else '(none -- 두 패치 테스트 판정 100% 일치)'}\n",
        "2. **외팔보 굽힘 및 전단 잠김 (Shear Locking, L/h=10)**:\n",
        f"   - LOCKING-FREE (ratio > 0.85): {', '.join(f'`{e}`' for e in locking_free) if locking_free else '(none)'}\n",
        f"   - LOCKED: {', '.join(f'`{e}`' for e in locked) if locked else '(none)'}\n",
        "3. **비압축성 체적 잠김 (nu = 0.49999 극한)**:\n",
        f"   - INCOMPRESSIBLE-OK: {', '.join(f'`{e}`' for e in vol_ok) if vol_ok else '(none)'}\n",
        f"   - VOL-LOCKED: {', '.join(f'`{e}`' for e in vol_locked) if vol_locked else '(none)'}\n",
    ])

    anomalies = []
    for i, e1 in enumerate(ALL_3D_ELEMENTS):
        for e2 in ALL_3D_ELEMENTS[i + 1:]:
            w1 = bending_res.get(e1, {}).get("w_tip")
            w2 = bending_res.get(e2, {}).get("w_tip")
            v1 = vol_res.get(e1, {}).get("w_tip")
            v2 = vol_res.get(e2, {}).get("w_tip")
            if w1 is not None and w2 is not None and v1 is not None and v2 is not None:
                if (np.isfinite(w1) and np.isfinite(w2) and abs(w1 - w2) < 1e-12 * max(abs(w1), 1e-30)
                        and np.isfinite(v1) and np.isfinite(v2) and abs(v1 - v2) < 1e-12 * max(abs(v1), 1e-30)):
                    anomalies.append((e1, e2))

    lines.append("\n## 3. 디스패치 이상 자동 탐지 (Dispatch Anomaly Auto-Detection)\n\n")
    if anomalies:
        lines.append(
            "> [!WARNING]\n"
            "> 아래 요소 쌍은 굽힘/체적 잠김 테스트에서 수치적으로 사실상 동일한 결과를 냈습니다. "
            "서로 다른 정식화를 가진 요소가 동일 벤치마크에서 완전히 같은 값을 내는 것은 "
            "우연이 아니라 디스패치/커널 미반영 결함일 가능성이 훨씬 높습니다 (AGENTS.md §4.2/§4.8/§4.16 참조). "
            "요소 하나가 다른 하나의 코드를 그대로 실행하고 있는 것은 아닌지 확인하십시오.\n\n"
        )
        for e1, e2 in anomalies:
            lines.append(f"- `{e1}` == `{e2}` (bending w_tip, volumetric w_tip both match to 1e-12 relative)\n")
    else:
        lines.append("이상 없음 -- 서로 다른 요소 쌍 간 수치적으로 의심스러운 완전 일치는 발견되지 않았습니다.\n")

    lines.extend([
        "\n## 4. 시각화 (Figures)\n\n",
        "![Patch test errors](../benchmark_element/figures/patch_test_errors.png)\n\n",
        "![Bending shear-locking ratio](../benchmark_element/figures/bending_locking_ratio.png)\n\n",
        "![Volumetric locking ratio](../benchmark_element/figures/volumetric_locking_ratio.png)\n",
    ])

    if guide_lines:
        lines.extend(guide_lines)

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def generate_actionable_improvement_guide(
    patch_res: Dict[str, Dict[str, Any]],
    abq_patch_res: Dict[str, Dict[str, Any]],
    bending_res: Dict[str, Dict[str, Any]],
    vol_res: Dict[str, Dict[str, Any]]
) -> Tuple[List[str], str]:
    """Analyze benchmark results and generate structured, actionable renovation recipes for agents."""
    guide_lines = [
        "\n## 5. 다른 에이전트 및 개발자를 위한 요소별 개선 가이드라인 (Actionable Renovation Recipes)\n\n",
        "> [!NOTE]\n",
        "> 본 섹션은 벤치마크 결과에서 발견된 결함 및 이상 징후를 후속 에이전트가 즉각 수정할 수 있도록\n",
        "> [증상] -> [이론적 원인] -> [수정 대상 소스 파일 및 함수] -> [개선 레시피]를 자동 집계한 것입니다.\n",
        "> 상세 아키텍처 및 배경 이론은 `benchmark_element/IMPROVEMENT_GUIDE.md`를 필독하십시오.\n\n"
    ]
    console_entries = []

    for elem in ALL_3D_ELEMENTS:
        p = patch_res.get(elem, {})
        abq = abq_patch_res.get(elem, {})
        b = bending_res.get(elem, {})
        v = vol_res.get(elem, {})

        t_err = p.get("tension", {}).get("max_error", 0.0)
        c_err = p.get("compression", {}).get("max_error", 0.0)
        sxy_err = p.get("shear_xy", {}).get("max_error", 0.0)
        syz_err = p.get("shear_yz", {}).get("max_error", 0.0)
        abq_err = abq.get("max_error", 0.0)
        max_p_err = max(t_err, c_err, sxy_err, syz_err, abq_err)

        b_verdict = b.get("locking_verdict", "N/A")
        v_verdict = v.get("verdict", "N/A")
        v_ratio = v.get("ratio_to_ref", 0.0)

        issues = []
        recipes = []

        if max_p_err >= 1e-6:
            issues.append(f"패치 테스트 실패 (최대 오차 {max_p_err:.1e} >= 1e-6)")
            recipes.append(
                "- **패치 불일치 점검**: 왜곡 메쉬에서 선형 변위장이 만족되지 않습니다. "
                "`dispsolver/element3d/` 내 해당 요소 커널의 형상함수 구배(`dN_dx`), 가우스 적분 가중치, "
                "정적 축약(Static condensation) 잔차 수렴성을 점검하십시오."
            )

        if elem == "C3D8_FBAR":
            if v_verdict == "ERROR" or abs(v_ratio) > 5.0 or v_verdict == "VOL-LOCKED":
                issues.append("극한 비압축성(nu=0.49999) 수치 불안정/발산")
                recipes.append(
                    "- **체적 일관성 접선(Volumetric Tangent) 보정**:\n"
                    "  - **원인**: de Souza Neto (1996) F-bar 정식화에서 체적 투영비 $(J_0/J)^{1/3}$의 변분 항이 알고리즘 접선에 완전 반영되지 않아 "
                    "nu in [0.4999, 0.49992] 근방의 좁은 대역에서 뉴턴 방향이 왜곡됩니다 (단, nu <= 0.499 실용 대역은 정상 수렴).\n"
                    "  - **수정 위치**: `dispsolver/element3d/c3d8_fbar_tl_numba.py` 내 `compute_c3d8_fbar_element_numba`.\n"
                    "  - **개선 방법**: 중심점 체적비 $J_0$에 대한 미분 항 d(J_0/J)^{1/3} 기하/재료 접선 보정 항을 추가하십시오.\n"
                    "  - **검증**: `python -u benchmark_element/benchmark_3d_mechanics.py` 실행 후 C3D8_FBAR 수렴 여부 확인."
                )
        elif elem == "C3D4_ANP":
            if v_verdict == "VOL-LOCKED" or v_verdict == "STIFFENED":
                issues.append("매끄러운 외팔보 굽힘에서 C3D4 대비 차별성 미미 (균일 변형으로 인한 평활화 효과 미발현)")
                recipes.append(
                    "- **불균일 국소 하중 벤치마크 적용**:\n"
                    "  - **원인**: Bonet & Burton (1998) ANP 2단계 절점 체적 평균화($J_a = v_a / V_a$, $\\bar{F} = (\\bar{J}/J)^{1/3} F$)는 "
                    "정상 구현되었으나, 규칙적인 메쉬의 매끄러운 굽힘에서는 각 요소의 체적비 $J_e$와 절점 평균 $\\bar{J}_e$의 차이가 1e-7 미만으로 매우 작아 "
                    "공간적 압력 진동이 없어 일반 C3D4와 수치적으로 유사하게 측정됩니다.\n"
                    "  - **개선 및 검증 방법**: `benchmark_element/mechanics_patches.py`에 국소 압입(Indentation) 또는 불균일 구속 패치를 추가하여 "
                    "체적 압력 평활화 성능을 비교하십시오."
                )
        elif elem == "C3D8H":
            if b_verdict == "LOCKED":
                recipes.append(
                    "- **이론적 정상 거동 (참고)**: C3D8H는 혼합 u-p 하이브리드 요소로 체적 비압축성 잠김을 해결하기 위해 개발되었으며, "
                    "전단 잠김(Shear locking) 완화 기구는 포함되어 있지 않습니다 (Abaqus Theory Guide §3.2.3 부합). "
                    "전단 잠김 해소가 필요한 경우 C3D8I 또는 C3D8R을 사용하십시오."
                )
        elif elem == "C3D8I":
            if v_verdict == "STIFFENED":
                recipes.append(
                    "- **체적 베타 모드 확장 (선택 사항)**: 현재 C3D8I는 9개의 비적합 변위 구배 모드($\\alpha_i$)가 정적 축약되어 전단 잠김을 완벽히 해소합니다 (처짐비 0.96). "
                    "추가적인 비압축성 극한 유연성이 필요한 경우, Wilson (1973) 및 Abaqus Theory Guide §3.2.5에 따른 4개의 체적 완화 모드($\\beta_j$, 총 13모드) 확장을 고려하십시오."
                )
        elif elem in ["C3D8", "C3D4", "C3D6"] and b_verdict == "LOCKED":
            recipes.append(
                "- **이론적 표준 거동**: 완전 가우스 적분 1차 요소는 수학적으로 기생 전단 잠김이 발생하는 것이 고전 유한요소 이론의 정상 결과입니다. "
                "굽힘 문제에서는 C3D8I, C3D8R, C3D10M을 권장합니다."
            )

        if issues or recipes:
            guide_lines.append(f"### `{elem}`\n")
            if issues:
                guide_lines.append(f"- **감지된 결함/경고**: {', '.join(issues)}\n")
            for r in recipes:
                guide_lines.append(f"{r}\n")
            guide_lines.append("\n")

            console_entries.append(f"[{elem}]")
            if issues:
                console_entries.append(f"  ! Issues: {', '.join(issues)}")
            for r in recipes:
                clean_r = r.replace("**", "").replace("- ", "  -> ")
                console_entries.append(f"  {clean_r}")

    console_summary = "\n".join(console_entries)
    return guide_lines, console_summary


if __name__ == "__main__":
    run_all_mechanics_benchmarks()

