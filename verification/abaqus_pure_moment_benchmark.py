"""
abaqus_pure_moment_benchmark.py
================================
Cross-code verification: our Q4_COROTATIONAL / Q4_COROTATIONAL_SRI elements
vs a real local Abaqus (Learning Edition, <=1000 nodes) run, on an
elastic-plastic large-rotation cantilever under an end force couple.

Why this exists
----------------
The hand-derived 1D elastic-plastic beam theory (verification/theory.py's
elastic_plastic_pure_moment_tip_state) does NOT correctly predict this
solver's plane-strain multiaxial J2 response once the section yields
(both Q4_COROTATIONAL and Q4_COROTATIONAL_SRI showed ~70-90% "error"
against that 1D theory, roughly EQUALLY for both elements and NOT
decreasing with mesh refinement -- a signature of a wrong reference, not
a wrong element, since a real element bug would differentiate between
element types the way the earlier SRI variational-consistency bug did).

Comparing directly against a real Abaqus solve of the IDENTICAL boundary
value problem sidesteps deriving the correct plane-strain elastic-plastic
moment-curvature relation entirely -- Abaqus's own converged nonlinear
FEA solution to the same loads/BCs/mesh/material is the reference.

Loading scheme
--------------
FIXED-direction (not follower) self-equilibrated nodal point-force couple
at the right (tip) edge, identical in both models: the same linear-
elastic-consistent force distribution `_consistent_axial_nodal_forces`
already uses in verification/convergence.py, computed ONCE from the
target moment and held fixed in direction (not re-oriented from the
deformed tip section) for the whole load history. This is deliberately
simpler than the "rigid-coupling reference point + concentrated moment"
scheme discussed but not built here -- this codebase's RBE2/RigidBodyPart
constraint machinery is Dirichlet (prescribed-motion) only, not
Neumann/force-driven, so a force-driven rigid coupling would be new
constraint-system work, not a benchmark script. Since the SAME fixed
loads are applied on both sides, this is still a valid, controlled
apples-to-apples comparison.

Choice of Abaqus element type -- READ THIS BEFORE CHANGING IT
-------------------------------------------------------------
The reference element is **CPE4I** (incompatible modes), NOT CPE4.

CPE4 is a fully-integrated bilinear quad and therefore carries parasitic
shear -- the Abaqus Theory Guide (sec. 3.2.5, "Continuum elements with
incompatible modes") states outright that the incompatible modes exist
"to eliminate the parasitic shear stresses that cause the response of
regular first-order displacement elements to be too stiff in bending".
Using CPE4 as ground truth for a *bending* benchmark therefore compares
our elements against a known-locked answer.

This is not hypothetical -- it is what this script originally did, and it
inverted the verdict.  Measured on the 21x5 mesh (tip U1, U2):

    Abaqus CPE4   (locks)          -2.186, 7.780
    Abaqus CPE4I  (locking-free)   -2.788, 8.676
    Abaqus CPE4R  (locking-free)   -2.861, 8.743
    Abaqus CPE4 refined to 61x13   -2.853, 8.758   <- CPE4's own limit
    ours Q4_COROTATIONAL           -2.088, 7.623
    ours Q4_COROTATIONAL_SRI       -2.816, 8.724

CPE4 on this mesh is ~11% too stiff against its OWN mesh-converged limit,
and that limit agrees with CPE4I/CPE4R on the coarse mesh.  Plain
Q4_COROTATIONAL is fully integrated too, so it locks the same way and
"agreed" with coarse CPE4 for the wrong reason, while the locking-free
Q4_COROTATIONAL_SRI was scored as the bad element for giving the right
answer.

Confirmed as pure element locking, unrelated to plasticity: rerunning the
identical problem with sigma_y0 = 1e12 (never yields) reproduces the same
gap -- CPE4 7.821 vs CPE4I 8.718, ours COROT 7.628 vs SRI 8.7245 -- i.e.
SRI matches CPE4I to 0.08% with the return map taken out of the picture
entirely.  See the "SRI + plasticity" investigation notes for the full
argument.

Mesh: 21 x 5 nodes (105 nodes, 80 elements) -- well under the Abaqus
Learning Edition's 1000-node limit.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import shutil

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from verification.mesh_utils import build_beam_mesh
from verification.theory import plane_strain_modulus, beam_I
from verification.convergence import _consistent_axial_nodal_forces

ABAQUS_CMD = r"C:\SIMULIA\Commands\abaqus.bat"

L = 20.0
HEIGHT = 1.0
NX, NY = 21, 5           # 105 nodes, 80 elements -- under 1000-node LE limit
E = 4000.0
NU = 0.3
SIGMA_Y0 = 80.0
HARDENING_H = 400.0
THETA_TARGET_DEG = 90.0
N_STEPS = 40

# Locking-free reference (see module docstring -- do NOT change to CPE4).
ABQ_REF_ELEMENT = "CPE4I"
# Fully-integrated control, reported alongside so the locking margin the
# reference is correcting for stays visible in every run's output.
ABQ_CONTROL_ELEMENT = "CPE4"


def _plastic_table(sigma_y0: float, H: float, n_points: int = 8,
                   eps_p_max: float = 1.0) -> str:
    """Abaqus *PLASTIC table (stress, plastic strain pairs) for a linear
    isotropic hardening law sigma = sigma_y0 + H*eps_p -- exact piecewise-
    linear representation since Abaqus interpolates linearly between
    table points and the underlying law already IS linear.
    """
    eps_p = np.linspace(0.0, eps_p_max, n_points)
    sigma = sigma_y0 + H * eps_p
    lines = [f"{s:.6f}, {ep:.6f}" for s, ep in zip(sigma, eps_p)]
    return "\n".join(lines)


def pick_target_element(mesh, info: dict) -> int:
    """Mid-span (away from both the fixed-end and load-application
    Saint-Venant zones -- moment is theoretically uniform = M_target
    along the whole beam for this pure end-moment loading, so mid-span
    is just the location cleanest of end effects), upper half (tension
    side for a positive-sense moment) element -- eid whose centroid is
    closest to (L/2, +H/4).
    """
    target_x, target_y = L / 2.0, HEIGHT / 4.0
    best_eid, best_d = None, None
    for eid, el in mesh.elements.items():
        coords = np.array([[mesh.nodes[n].x, mesh.nodes[n].y] for n in el.node_ids])
        cx, cy = coords[:, 0].mean(), coords[:, 1].mean()
        d = (cx - target_x) ** 2 + (cy - target_y) ** 2
        if best_d is None or d < best_d:
            best_d, best_eid = d, eid
    return best_eid


def write_abaqus_inp(path: str, mesh, info: dict, node_forces: dict,
                     target_eid: int, element_type: str = ABQ_REF_ELEMENT) -> None:
    """Write a plane-strain .inp deck for the pure-moment benchmark.

    element_type: Abaqus 2D solid element keyword. Defaults to the
        locking-free CPE4I -- see the module docstring for why CPE4 is
        not an acceptable reference for a bending benchmark.

    node_forces: {node_id (0-based, this codebase's convention): fx}
        applied at the tip (right) edge, DOF 1 (x / axial direction).
    Abaqus requires node/element labels >= 1, so every ID is offset by +1
    when written (mesh/info stay 0-based internally, matching every other
    consumer in this codebase).
    """
    nodes = sorted(mesh.nodes.keys())
    elems = sorted(mesh.elements.keys())

    lines = ["*HEADING", "Pure-moment elastic-plastic cantilever -- cross-code check vs dispsolver"]

    lines.append("*NODE")
    for nid in nodes:
        n = mesh.nodes[nid]
        lines.append(f"{nid + 1}, {n.x:.8f}, {n.y:.8f}")

    lines.append(f"*ELEMENT, TYPE={element_type}, ELSET=ALLEL")
    for eid in elems:
        el = mesh.elements[eid]
        conn1 = [c + 1 for c in el.node_ids]
        lines.append(f"{eid + 1}, " + ", ".join(str(c) for c in conn1))

    lines.append("*SOLID SECTION, ELSET=ALLEL, MATERIAL=MAT1")
    lines.append("1.0")

    lines.append("*MATERIAL, NAME=MAT1")
    lines.append("*ELASTIC")
    lines.append(f"{E:.6f}, {NU:.6f}")
    lines.append("*PLASTIC")
    lines.append(_plastic_table(SIGMA_Y0, HARDENING_H))

    left_nodes = [n + 1 for n in info['left']]
    lines.append("*NSET, NSET=LEFT")
    lines.append(", ".join(str(n) for n in left_nodes))
    lines.append("*BOUNDARY")
    lines.append("LEFT, 1, 2, 0.0")

    right_nodes_1based = [n + 1 for n in info['right']]
    lines.append("*NSET, NSET=RIGHT")
    lines.append(", ".join(str(n) for n in right_nodes_1based))

    tip_node_1based = right_nodes_1based[len(right_nodes_1based) // 2]
    lines.append("*NSET, NSET=TIPMID")
    lines.append(f"{tip_node_1based}")

    lines.append("*ELSET, ELSET=TARGETEL")
    lines.append(f"{target_eid + 1}")

    lines.append("*STEP, NLGEOM=YES, INC=200")
    lines.append("*STATIC")
    lines.append("0.025, 1.0, 1e-6, 0.1")
    lines.append("*CLOAD")
    for nid0, fx in node_forces.items():
        lines.append(f"{nid0 + 1}, 1, {fx:.8f}")
    lines.append("*OUTPUT, FIELD")
    lines.append("*NODE OUTPUT")
    lines.append("U, RF")
    lines.append("*ELEMENT OUTPUT")
    lines.append("S")
    lines.append("*OUTPUT, HISTORY")
    lines.append("*NODE OUTPUT, NSET=TIPMID")
    lines.append("U1, U2")
    lines.append("*NODE PRINT, NSET=TIPMID, FREQUENCY=1")
    lines.append("U1, U2")
    lines.append("*NODE PRINT, NSET=LEFT, FREQUENCY=1")
    lines.append("RF1, RF2")
    lines.append("*EL PRINT, ELSET=TARGETEL, FREQUENCY=1, POSITION=CENTROIDAL")
    lines.append("S")
    lines.append("*END STEP")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_abaqus_job(inp_path: str, job_name: str, workdir: str,
                   tip_node_1based: int, left_nodes_1based: list,
                   target_eid_1based: int,
                   timeout: int = 600) -> dict:
    """Run the .inp through the local Abaqus install, wait for
    completion, and parse the tip node's final U1/U2, the LEFT-nset
    reaction forces, and the target element's centroidal stress from the
    .dat file (all three share the same 3/4-column text-table shape, so
    matches are disambiguated by node/element ID membership, then the
    LAST occurrence of each -- the final converged increment -- is kept).
    """
    dest_inp = os.path.join(workdir, f"{job_name}.inp")
    if os.path.abspath(inp_path) != os.path.abspath(dest_inp):
        shutil.copy(inp_path, dest_inp)
    cmd = [ABAQUS_CMD, f"job={job_name}", "interactive", "ask_delete=OFF"]
    result = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                            timeout=timeout)

    sta_path = os.path.join(workdir, f"{job_name}.sta")
    completed = False
    if os.path.exists(sta_path):
        with open(sta_path) as f:
            sta_text = f.read()
        completed = "COMPLETED" in sta_text and "NOT BEEN COMPLETED" not in sta_text

    dat_path = os.path.join(workdir, f"{job_name}.dat")
    u1 = u2 = None
    rf_total = None
    stress = None
    if os.path.exists(dat_path):
        with open(dat_path) as f:
            dat_text = f.read()

        # 3-column tables (node id, val1, val2) -- used for both TIPMID
        # U1/U2 and LEFT RF1/RF2 (same shape, disambiguated by node ID).
        three_col = re.findall(
            r"^\s*(\d+)\s+([-\d.E+]+)\s+([-\d.E+]+)\s*$",
            dat_text, re.MULTILINE,
        )
        tip_rows = [m for m in three_col if int(m[0]) == tip_node_1based]
        if tip_rows:
            u1, u2 = float(tip_rows[-1][1]), float(tip_rows[-1][2])

        left_set = set(left_nodes_1based)
        left_rows_by_occurrence = [m for m in three_col if int(m[0]) in left_set]
        # Each printed increment contributes len(left_nodes_1based) rows,
        # in the same left-to-node order every time -- take the last block.
        n_left = len(left_nodes_1based)
        if len(left_rows_by_occurrence) >= n_left:
            last_block = left_rows_by_occurrence[-n_left:]
            rf1_sum = sum(float(m[1]) for m in last_block)
            rf2_sum = sum(float(m[2]) for m in last_block)
            # Reaction moment about the root (x=0): M = sum(RF1_i * y_i),
            # y taken from the ordered left_nodes_1based/left_rows pairing.
            rf_total = {'RF1_sum': rf1_sum, 'RF2_sum': rf2_sum,
                       'rows': [(int(m[0]), float(m[1]), float(m[2])) for m in last_block]}

        # 5-column element stress table (element id, S11, S22, S33, S12).
        stress_rows = re.findall(
            rf"^\s*{target_eid_1based}\s+([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)\s*$",
            dat_text, re.MULTILINE,
        )
        if stress_rows:
            s11, s22, s33, s12 = (float(v) for v in stress_rows[-1])
            stress = {'S11': s11, 'S22': s22, 'S33': s33, 'S12': s12}

    return {
        'completed': completed,
        'u1': u1, 'u2': u2,
        'rf': rf_total,
        'stress': stress,
        'returncode': result.returncode,
        'stdout_tail': result.stdout[-2000:] if result.stdout else '',
        'stderr_tail': result.stderr[-2000:] if result.stderr else '',
        'workdir': workdir,
    }


def _cauchy_stress_at_centroid(mesh, u: np.ndarray, eid: int, nid_to_idx: dict,
                               material, gp_state: np.ndarray) -> dict:
    """Cauchy stress (S11, S22, S12) at an element's centroid, computed
    directly from the GLOBAL displacement gradient -- no corotational
    frame extraction needed: PK2 stress for an isotropic
    hyperelastic/plastic material is a function of C = F^T F only, and
    C is invariant under any rigid rotation superposed on F
    (C = (R F)^T (R F) = F^T R^T R F = F^T F), so computing F directly
    from global nodal displacements and calling the material's own
    pk2_voigt gives the exact, objective physical stress regardless of
    which element formulation (plain vs corotational vs SRI) produced
    the converged displacement field -- matches the same approach
    dispsolver/postprocess/viewer.py already uses for stress
    visualization (see its module docstring / _deformation_gradient).

    Returns Cauchy (push-forward) stress: sigma = F @ S @ F^T / det(F),
    directly comparable to Abaqus's default (Cauchy/true stress) S output.
    """
    from dispsolver.element import q4

    el = mesh.elements[eid]
    node_ids = el.node_ids
    coords = np.array([[mesh.nodes[n].x, mesh.nodes[n].y] for n in node_ids])
    u_elem = np.empty(8)
    for a, nid in enumerate(node_ids):
        idx = nid_to_idx[nid]
        u_elem[2 * a] = u[idx * 2]
        u_elem[2 * a + 1] = u[idx * 2 + 1]

    _, _, invJ = q4.jacobian(0.0, 0.0, coords)
    dN_dxi, dN_deta = q4.shape_derivatives(0.0, 0.0)
    gX = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
    gY = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
    ux = u_elem[0::2]
    uy = u_elem[1::2]
    F = np.eye(2) + np.array([[ux @ gX, ux @ gY], [uy @ gX, uy @ gY]])

    S_v, _, _ = material.pk2_voigt(F, {}, gp_state)
    S = np.array([[S_v[0], S_v[2]], [S_v[2], S_v[1]]])
    detF = np.linalg.det(F)
    sigma = (F @ S @ F.T) / detF
    return {'S11': float(sigma[0, 0]), 'S22': float(sigma[1, 1]), 'S12': float(sigma[0, 1])}


def run_our_solver(mesh, info, node_forces: dict, element_type: str,
                   target_eid: int = None,
                   n_steps: int = N_STEPS) -> dict:
    """Run the identical mesh/material/fixed-direction loads through our
    own DynamicSolver, ramped over n_steps, returns tip mid-node U1/U2,
    LEFT-nset reaction forces, and (if target_eid given) Cauchy stress at
    that element's centroid.
    """
    from dispsolver.solver import DynamicSolver
    from dispsolver.material import J2Plasticity

    nid_to_idx = info['nid_to_idx']
    mat = J2Plasticity(E=E, nu=NU, sigma_y0=SIGMA_Y0, H=HARDENING_H)

    solver = DynamicSolver(
        mesh, {0: mat}, rho=1e-6, material_params={},
        max_iter=50, tol=1e-4, atol=1e-7, rtol=5e-3,
        mode='quasistatic',
        element_type={0: element_type},
    )
    solver.sta_status = False

    bc_dofs, bc_vals = [], []
    for nid in info['left']:
        idx = nid_to_idx[nid]
        bc_dofs.extend([idx * 2, idx * 2 + 1])
        bc_vals.extend([0.0, 0.0])
    solver.set_prescribed_dofs(bc_dofs, bc_vals)

    load_dofs = []
    for nid0 in node_forces:
        idx = nid_to_idx[nid0]
        load_dofs.append(idx * 2)  # DOF 1 = x

    total_iter = 0
    n_fail = 0
    for step_i in range(1, n_steps + 1):
        frac = step_i / n_steps
        load_vals = [node_forces[nid0] * frac for nid0 in node_forces]
        solver.apply_load(load_dofs, load_vals)
        n_iter = solver.solve_step(dt=1.0)
        if n_iter < 0:
            n_iter1 = solver.solve_step(dt=0.5)
            if n_iter1 < 0:
                n_fail += 1
                break
            n_iter2 = solver.solve_step(dt=0.5)
            if n_iter2 < 0:
                n_fail += 1
                break
            n_iter = n_iter1 + n_iter2
        total_iter += int(n_iter)

    tip_nid = info['right'][len(info['right']) // 2]
    idx = nid_to_idx[tip_nid]
    u1 = float(solver.u[idx * 2])
    u2 = float(solver.u[idx * 2 + 1])

    rf_full = solver.reaction_forces()
    rf_rows = []
    for nid in info['left']:
        idx_l = nid_to_idx[nid]
        rf_rows.append((nid, float(rf_full[idx_l * 2]), float(rf_full[idx_l * 2 + 1])))
    rf = {
        'RF1_sum': sum(r[1] for r in rf_rows),
        'RF2_sum': sum(r[2] for r in rf_rows),
        'rows': rf_rows,
    }

    stress = None
    if target_eid is not None:
        n_vars = mat.n_internal_vars
        eid_to_idx = {eid: e for e, eid in enumerate(solver.elem_ids)}
        e_idx = eid_to_idx[target_eid]
        gp_state = solver.state[e_idx, 0, :n_vars] if solver.state is not None else np.array([1., 0., 0., 1., 0.])
        stress = _cauchy_stress_at_centroid(mesh, solver.u, target_eid, nid_to_idx, mat, gp_state)

    return {'u1': u1, 'u2': u2, 'n_iter_total': total_iter, 'n_fail': n_fail,
            'rf': rf, 'stress': stress}


def main():
    mesh, info = build_beam_mesh(NX, NY, length=L, height=HEIGHT)
    nid_to_idx = info['nid_to_idx']

    E_star = plane_strain_modulus(E, NU)
    I_beam = beam_I(HEIGHT)
    theta_target = np.deg2rad(THETA_TARGET_DEG)
    M_target = theta_target * E_star * I_beam / L

    right_nids = info['right']
    y_nodes = np.array([mesh.nodes[nid].y for nid in right_nids])
    f_axial = _consistent_axial_nodal_forces(M_target, HEIGHT, NY, y_nodes)
    node_forces = {nid: float(f) for nid, f in zip(right_nids, f_axial)}

    target_eid = pick_target_element(mesh, info)
    target_coords = [(mesh.nodes[n].x, mesh.nodes[n].y) for n in mesh.elements[target_eid].node_ids]

    print(f"Mesh: {NX}x{NY} = {len(mesh.nodes)} nodes, {len(mesh.elements)} elements")
    print(f"M_target = {M_target:.4f} (theta_target={THETA_TARGET_DEG} deg, elastic-equivalent)")
    print(f"Tip node forces (DOF 1): {node_forces}")
    print(f"Target element for stress comparison: eid={target_eid}, coords={target_coords}")

    tip_nid = right_nids[len(right_nids) // 2]

    def _run_abq(etype: str) -> dict:
        wd = tempfile.mkdtemp(prefix=f"abaqus_pure_moment_{etype}_")
        p = os.path.join(wd, "job.inp")
        write_abaqus_inp(p, mesh, info, node_forces, target_eid,
                         element_type=etype)
        return run_abaqus_job(
            p, "job", wd,
            tip_node_1based=tip_nid + 1,
            left_nodes_1based=[n + 1 for n in info['left']],
            target_eid_1based=target_eid + 1,
        )

    print("\n=== Running Abaqus (Learning Edition, local) ===")
    abq = _run_abq(ABQ_REF_ELEMENT)
    print(f"[reference] {ABQ_REF_ELEMENT}: completed={abq['completed']} "
          f"tip U1={abq['u1']}, U2={abq['u2']}")
    if not abq['completed']:
        print("STDOUT tail:", abq['stdout_tail'])
        print("STDERR tail:", abq['stderr_tail'])
    print(f"[reference] LEFT reactions: {abq['rf']}")
    print(f"[reference] target-element stress: {abq['stress']}")

    abq_ctrl = _run_abq(ABQ_CONTROL_ELEMENT)
    print(f"[control]   {ABQ_CONTROL_ELEMENT} (fully integrated, locks): "
          f"completed={abq_ctrl['completed']} tip U1={abq_ctrl['u1']}, U2={abq_ctrl['u2']}")
    if abq['u2'] and abq_ctrl['u2']:
        lock = abs(abq_ctrl['u2'] - abq['u2']) / abs(abq['u2'])
        print(f"            -> reference-vs-control tip-U2 gap = {lock:.1%} "
              f"(this is the locking margin CPE4 would have hidden; an element "
              f"that 'agrees' with the control is locking too)")

    results = {'abaqus': abq, 'abaqus_control': abq_ctrl}
    for et in ['Q4_COROTATIONAL', 'Q4_COROTATIONAL_SRI']:
        print(f"\n=== Running our solver: {et} ===")
        r = run_our_solver(mesh, info, node_forces, et, target_eid=target_eid)
        print(f"  tip U1={r['u1']:.6f}, U2={r['u2']:.6f}, iters={r['n_iter_total']}, fail={r['n_fail']}")
        print(f"  reactions: RF1_sum={r['rf']['RF1_sum']:.4f}, RF2_sum={r['rf']['RF2_sum']:.4f}")
        print(f"  stress: {r['stress']}")
        results[et] = r

    if abq['u1'] is not None and abq['u2'] is not None:
        print(f"\n=== Comparison vs Abaqus {ABQ_REF_ELEMENT} (locking-free reference) ===")
        for et in ['Q4_COROTATIONAL', 'Q4_COROTATIONAL_SRI']:
            r = results[et]
            du1 = r['u1'] - abq['u1']
            du2 = r['u2'] - abq['u2']
            err = np.sqrt(du1**2 + du2**2) / L
            print(f"  {et}: tip position error vs Abaqus / L = {err:.4e}")
            if abq['rf']:
                drf1 = r['rf']['RF1_sum'] - abq['rf']['RF1_sum']
                drf2 = r['rf']['RF2_sum'] - abq['rf']['RF2_sum']
                print(f"    reaction sum diff: dRF1={drf1:.4f}, dRF2={drf2:.4f}"
                     f"  (Abaqus RF1_sum={abq['rf']['RF1_sum']:.4f}, RF2_sum={abq['rf']['RF2_sum']:.4f})")
            if abq['stress'] and r['stress']:
                for comp in ('S11', 'S22', 'S12'):
                    a_val = abq['stress'][comp]
                    r_val = r['stress'][comp]
                    rel = abs(r_val - a_val) / max(abs(a_val), 1e-6)
                    print(f"    {comp}: ours={r_val:.4f}  abaqus={a_val:.4f}  rel_err={rel:.4e}")

    print(f"\nAbaqus workdirs kept at:"
          f"\n  {ABQ_REF_ELEMENT} (reference): {abq['workdir']}"
          f"\n  {ABQ_CONTROL_ELEMENT} (control):  {abq_ctrl['workdir']}")
    return results


if __name__ == "__main__":
    main()
