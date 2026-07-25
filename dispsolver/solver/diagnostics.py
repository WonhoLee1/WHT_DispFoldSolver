"""
diagnostics.py
==============
Runtime sanity checks for multi-part (rigid plate + tie + deformable
display) folding simulations.

Motivation: a solver step can report a fully converged Newton iteration
(small residual, no cutback) while the *physics* is still wrong - e.g. a
display region tied to a rotating rigid plate that never actually moves
because the tie's force/stiffness contribution silently never reached the
global system (see AGENTS.md section 4.8 for the concrete bug this was
written to catch). Convergence success alone does not prove the coupling
is doing what you think it's doing; these checks compare *physical*
quantities (displacement magnitude, tie gap, achieved hinge angle)
against what should be true if the simulation is behaving correctly.

Intended usage: call `sanity_report(...)` once per step (or every few
steps) from an example's time-stepping loop; it prints WARN lines and
returns False the moment something looks physically wrong, so a bug like
"display frozen at u=0" is caught within the first few steps instead of
only being noticed by eyeballing a final PNG after a multi-minute run.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np


def check_region_tracking(
    u: np.ndarray,
    driving_dofs: Sequence[int],
    driven_dofs: Sequence[int],
    label_driving: str = "driving",
    label_driven: str = "driven",
    ratio_warn: float = 0.05,
    min_move_mm: float = 1e-3,
) -> Dict:
    """Compare displacement magnitude between a driving region (e.g. a
    rigid plate moved by a prescribed hinge rotation) and a region that is
    only supposed to move *because* it's coupled to the driving region
    (e.g. a display tied to the plate's top surface).

    Flags 'suspicious' if the driving region has moved meaningfully but
    the driven region has moved by less than `ratio_warn` of that amount
    - the coupling (tie / RBE2 / whatever links them) is most likely not
    actually contributing forces/stiffness to the driven region.
    """
    max_drv = float(np.max(np.abs(u[driving_dofs]))) if len(driving_dofs) else 0.0
    max_drn = float(np.max(np.abs(u[driven_dofs]))) if len(driven_dofs) else 0.0
    suspicious = (max_drv > min_move_mm) and (max_drn < ratio_warn * max_drv)
    return {
        f"max|u|_{label_driving}": max_drv,
        f"max|u|_{label_driven}": max_drn,
        "ratio": (max_drn / max_drv) if max_drv > 0 else 1.0,
        "suspicious": suspicious,
    }


def check_tie_gap(
    u: np.ndarray,
    coords: np.ndarray,
    tie_constraint,
    nid_to_idx: Dict[int, int],
) -> Dict:
    """Current geometric gap (mm) for every pair in a SurfaceTieConstraint,
    evaluated the same way SurfaceTieConstraint.apply_penalty does - a
    perfectly-behaving tie should keep this small relative to the overall
    deformation scale (it's a *penalty* spring, not an exact constraint,
    so some nonzero gap is expected - a gap comparable to the imposed
    displacement itself means the tie isn't meaningfully constraining
    anything).
    """
    gaps = []
    for s_nid, m1_nid, m2_nid, xi in tie_constraint.pairs:
        s_idx = nid_to_idx[s_nid]
        m1_idx = nid_to_idx[m1_nid]
        m2_idx = nid_to_idx[m2_nid]
        xs = coords[s_idx] + u[2 * s_idx: 2 * s_idx + 2]
        xm1 = coords[m1_idx] + u[2 * m1_idx: 2 * m1_idx + 2]
        xm2 = coords[m2_idx] + u[2 * m2_idx: 2 * m2_idx + 2]
        N1 = 0.5 * (1.0 - xi)
        N2 = 0.5 * (1.0 + xi)
        gap = xs - (N1 * xm1 + N2 * xm2)
        gaps.append(float(np.linalg.norm(gap)))
    gaps_arr = np.array(gaps) if gaps else np.array([0.0])
    return {
        "n_pairs": len(gaps),
        "mean_gap_mm": float(gaps_arr.mean()),
        "max_gap_mm": float(gaps_arr.max()),
    }


def check_hinge_rotation(
    u: np.ndarray,
    coords: np.ndarray,
    nid_to_idx: Dict[int, int],
    master_id: int,
    ref_slave_id: int,
    target_theta_rad: float,
    theta_tol_deg: float = 2.0,
) -> Dict:
    """Verify a master/reference-slave node pair's current relative angle
    actually matches the prescribed hinge rotation `target_theta_rad`,
    rather than trusting that "the BC was set" means "the BC took effect".
    """
    m_idx = nid_to_idx[master_id]
    s_idx = nid_to_idx[ref_slave_id]
    d0 = coords[s_idx] - coords[m_idx]
    xm = coords[m_idx] + u[2 * m_idx: 2 * m_idx + 2]
    xs = coords[s_idx] + u[2 * s_idx: 2 * s_idx + 2]
    d_now = xs - xm
    ang0 = np.arctan2(d0[1], d0[0])
    ang_now = np.arctan2(d_now[1], d_now[0])
    achieved = ang_now - ang0
    achieved = (achieved + np.pi) % (2.0 * np.pi) - np.pi  # wrap to [-pi, pi]
    err_deg = float(np.degrees(achieved - target_theta_rad))
    return {
        "achieved_theta_deg": float(np.degrees(achieved)),
        "target_theta_deg": float(np.degrees(target_theta_rad)),
        "error_deg": err_deg,
        "ok": abs(err_deg) < theta_tol_deg,
    }


def check_smooth_curvature(
    u: np.ndarray,
    coords: np.ndarray,
    nid_to_idx: Dict[int, int],
    hinge_node_ids: Sequence[int],
    kink_ratio_warn: float = 4.0,
) -> Dict:
    """Detect a sharp crease/kink in the free hinge zone (AGENTS.md section
    1.0 criterion 2: the real fold shape must be a smooth rounded U/teardrop,
    not a sharp V, because real foldable displays have a minimum bend
    radius).

    Sorts `hinge_node_ids` by their initial X coordinate, takes the
    resulting `uy(x)` profile, and computes a discrete second-difference
    (curvature proxy) at each interior node. A physically smooth bend has
    curvature that varies gradually along the profile; a crease shows up
    as one node's curvature magnitude spiking far above its neighbors'.
    Flags 'kink_detected' if any interior curvature magnitude exceeds
    `kink_ratio_warn` times the median of all interior curvature
    magnitudes (robust outlier check, not a fixed physical threshold,
    since the "normal" curvature magnitude changes a lot over the course
    of a folding run).
    """
    idx = [nid_to_idx[n] for n in hinge_node_ids]
    x0 = coords[idx, 0]
    order = np.argsort(x0)
    idx_sorted = [idx[i] for i in order]
    x_sorted = x0[order]
    uy_sorted = np.array([u[2 * i + 1] for i in idx_sorted])

    if len(idx_sorted) < 3:
        return {"n_nodes": len(idx_sorted), "kink_detected": False, "max_curv_ratio": 0.0}

    # Discrete second derivative w.r.t. x (non-uniform spacing safe: use
    # simple index-based central difference since the hinge mesh is
    # structured/near-uniform along x).
    curv = np.abs(uy_sorted[2:] - 2.0 * uy_sorted[1:-1] + uy_sorted[:-2])
    median_curv = np.median(curv) + 1e-12
    max_curv = float(np.max(curv))
    max_curv_ratio = max_curv / median_curv
    kink_idx = int(np.argmax(curv)) + 1  # +1 to map back to uy_sorted index
    return {
        "n_nodes": len(idx_sorted),
        "max_curv_ratio": max_curv_ratio,
        "kink_detected": bool(max_curv_ratio > kink_ratio_warn),
        "kink_x": float(x_sorted[kink_idx]) if max_curv_ratio > kink_ratio_warn else None,
    }


def check_plate_closure(
    u: np.ndarray,
    coords: np.ndarray,
    nid_to_idx: Dict[int, int],
    left_master_id: int,
    left_ref_id: int,
    right_master_id: int,
    right_ref_id: int,
    target_theta_l_rad: float,
    target_theta_r_rad: float,
    theta_tol_deg: float = 2.0,
) -> Dict:
    """AGENTS.md section 1.0 criterion 3: at full fold the two plates should
    end up close to parallel and facing each other (like a closed book),
    not just each independently having reached its own target angle in
    isolation from the display (that part is checked separately by
    `check_region_tracking`/`sanity_report`).

    Uses `check_hinge_rotation` for each plate (master + one reference
    slave node) to verify each side actually achieved its prescribed
    rotation, then reports the current center-to-center gap between the
    two master RPs as a proxy for "how closed" the fold is (shrinks
    monotonically toward the U-loop's diameter as folding progresses).
    """
    left = check_hinge_rotation(u, coords, nid_to_idx, left_master_id, left_ref_id,
                                  target_theta_l_rad, theta_tol_deg)
    right = check_hinge_rotation(u, coords, nid_to_idx, right_master_id, right_ref_id,
                                   target_theta_r_rad, theta_tol_deg)
    lm_idx = nid_to_idx[left_master_id]
    rm_idx = nid_to_idx[right_master_id]
    xl = coords[lm_idx] + u[2 * lm_idx: 2 * lm_idx + 2]
    xr = coords[rm_idx] + u[2 * rm_idx: 2 * rm_idx + 2]
    gap_mm = float(np.linalg.norm(xr - xl))
    return {
        "left": left,
        "right": right,
        "both_ok": bool(left["ok"] and right["ok"]),
        "plate_gap_mm": gap_mm,
    }


def fold_success_verdict(
    u: np.ndarray,
    coords: np.ndarray,
    nid_to_idx: Dict[int, int],
    hinge_node_ids: Sequence[int],
    tie_constraints: Optional[List] = None,
    driving_dofs: Optional[Sequence[int]] = None,
    driven_dofs: Optional[Sequence[int]] = None,
    plate_closure_kwargs: Optional[Dict] = None,
) -> Dict:
    """Combined AGENTS.md section 1.0 verdict: is this a genuine U-shape
    fold, not just "Newton converged"? Checks all three listed criteria
    (element quality is expected to be checked separately via
    `solver._check_mesh_quality`, since that needs the live solver
    object, not just `u`/`coords`).
    """
    result: Dict = {}
    if driving_dofs is not None and driven_dofs is not None:
        result["tracking"] = check_region_tracking(u, driving_dofs, driven_dofs)
    result["curvature"] = check_smooth_curvature(u, coords, nid_to_idx, hinge_node_ids)
    if tie_constraints:
        result["tie_gaps"] = [check_tie_gap(u, coords, tie, nid_to_idx) for tie in tie_constraints]
    if plate_closure_kwargs is not None:
        result["closure"] = check_plate_closure(u, coords, nid_to_idx, **plate_closure_kwargs)

    ok = True
    if "tracking" in result and result["tracking"]["suspicious"]:
        ok = False
    if result["curvature"]["kink_detected"]:
        ok = False
    if "closure" in result and not result["closure"]["both_ok"]:
        ok = False
    result["u_shape_ok"] = ok
    return result


def sanity_report(
    step: int,
    u: np.ndarray,
    coords: np.ndarray,
    nid_to_idx: Dict[int, int],
    tie_constraints: Optional[List] = None,
    driving_dofs: Optional[Sequence[int]] = None,
    driven_dofs: Optional[Sequence[int]] = None,
    driving_label: str = "plate",
    driven_label: str = "display",
    tie_gap_warn_mm: float = 2.0,
    verbose: bool = True,
) -> bool:
    """One combined sanity check per call, meant to run every step (or
    every few steps) inside a time-stepping loop. Prints `[WARN]` lines
    for anything physically suspicious (not just non-convergence) and
    returns False the moment something looks wrong, so the caller can
    `break`/investigate immediately rather than discovering it later from
    a final plot.
    """
    ok = True
    lines = []

    if driving_dofs is not None and driven_dofs is not None:
        r = check_region_tracking(u, driving_dofs, driven_dofs, driving_label, driven_label)
        if r["suspicious"]:
            ok = False
            lines.append(
                f"  [WARN] step {step}: {driven_label} NOT tracking {driving_label} "
                f"(max|u|_{driving_label}={r[f'max|u|_{driving_label}']:.3f}mm, "
                f"max|u|_{driven_label}={r[f'max|u|_{driven_label}']:.3f}mm, "
                f"ratio={r['ratio']:.4f} - expected coupling force/stiffness is "
                f"likely not reaching the global system)"
            )

    if tie_constraints:
        for tie in tie_constraints:
            g = check_tie_gap(u, coords, tie, nid_to_idx)
            if g["max_gap_mm"] > tie_gap_warn_mm:
                ok = False
                name = getattr(tie, "name", "?")
                lines.append(
                    f"  [WARN] step {step}: tie '{name}' max_gap={g['max_gap_mm']:.3f}mm "
                    f"(mean={g['mean_gap_mm']:.3f}mm over {g['n_pairs']} pairs) "
                    f"- tie constraint may be failing to hold"
                )

    if verbose:
        for ln in lines:
            print(ln, flush=True)

    return ok
