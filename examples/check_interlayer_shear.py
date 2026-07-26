"""
check_interlayer_shear.py
=========================
Verifies the *purpose* of the low-modulus PSA layers in the 14-layer
PET-PSA display stack (examples/gen_ex12_inp.py).

Why the PSA layers exist
------------------------
The alternating soft PSA (Arruda-Boyce, mu ~ 0.168 MPa) layers between
stiff PET (E = 4000 MPa) layers are there to let the stack **shear
between layers** during folding. Without interlayer shear, folding a
bonded 0.5mm laminate to 90deg/side would force huge axial strain into
the outer fibres. With it, each layer slides slightly relative to its
neighbours, and that accumulated slip shows up as a book-page /
staircase offset between layers.

Where to look for the staircase -- NOT necessarily the free tip
---------------------------------------------------------------
Naively you would look at the free tips (x = +-40). But in this model
the display's bottom row is **tied to the rigid plate over the whole
plate span** (|x| >= 10), which pins the bottom layer there. The
staircase can then only survive out to the tip if the shear-lag length

    l = sqrt(E_PET * t_PET * t_PSA / G_PSA)
      = sqrt(4000 * 0.0357 * 0.0357 / 0.168) ~= 5.5 mm

is comparable to the 30mm plate span -- it is not, so any slip generated
in the free hinge span (|x| < 10) decays like exp(-d/5.5mm) and is
~0.4% of its original value by x = +-40. **The staircase is therefore
expected near the hinge edge (x ~ +-10), not at the tip.** This script
scans slip along the whole length so you can see where it actually
lives, instead of checking one column and concluding "no shear".

What is measured
----------------
At each x-column, in the *deformed* configuration:

    t_hat  : unit tangent of the display's bottom surface at that x
    n_hat  : unit normal (t_hat rotated 90deg)
    slip_j : (P_j - P_0) . t_hat      <- staircase offset of layer j
    thk_j  : (P_j - P_0) . n_hat      <- remaining through-thickness

`slip_j` vanishes for rigid-body motion and for pure bending with no
interlayer shear, so any non-zero value is genuine interlayer shear.

Usage
-----
    python -u examples/check_interlayer_shear.py           # solve, then analyze
    python -u examples/check_interlayer_shear.py --cached  # reuse saved u
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from ex12_abaqus_inp_plate_fold import run_abaqus_inp_folding, _laminate_layer_materials
from dispsolver.io import read_abaqus_input

HERE = os.path.dirname(__file__)
INP = os.path.join(HERE, "ex12_rigid_plate_display_fold.inp")
U_CACHE = os.path.join(HERE, "ex12_final_u.npy")
TOL = 1e-6
# Set by main() from the actual mesh via _laminate_layer_materials() --
# NOT hard-coded, since gen_ex12_inp.py's layup is no longer a uniform
# 14-row PET/PSA alternation (see AGENTS.md 1.4): it's now 14 physical
# layers of uneven row counts (3 PET rows + 1 PSA row, repeated 7x = 28
# mesh rows). Hard-coding this here once already went stale once.
N_LAYERS = None
LAYER_MATERIAL = None


def _display_columns(mesh):
    """{x: [node ids bottom->top]} for the display block only (nid < 10000)."""
    cols = {}
    for nid, n in mesh.nodes.items():
        if nid >= 10000:
            continue
        cols.setdefault(round(n.x, 6), []).append((n.y, nid))
    return {x: [nid for _, nid in sorted(v)] for x, v in sorted(cols.items())}


def _column_slip(mesh, u, nid_to_idx, col, col_ref, ref_is_behind):
    """slip/thickness profile of one through-thickness column.

    `col_ref` is a neighbouring column used to build the deformed surface
    tangent. `ref_is_behind` flags that it lies at smaller x, so the
    tangent must be negated to keep t_hat pointing along +x everywhere
    (otherwise the slip sign flips at the last column).
    """
    def deformed(nid):
        idx = nid_to_idx[nid]
        n = mesh.nodes[nid]
        return np.array([n.x + u[2 * idx], n.y + u[2 * idx + 1]])

    P = [deformed(nid) for nid in col]
    t_vec = deformed(col_ref[0]) - P[0]
    if ref_is_behind:
        t_vec = -t_vec
    nrm = np.linalg.norm(t_vec)
    if nrm < 1e-12:
        return None
    t_hat = t_vec / nrm
    n_hat = np.array([-t_hat[1], t_hat[0]])
    slip = np.array([float((p - P[0]) @ t_hat) for p in P])
    thk = np.array([float((p - P[0]) @ n_hat) for p in P])
    return {"P": np.array(P), "slip": slip, "thk": thk}


def scan(mesh, u):
    """Slip profile at every display x-column."""
    nid_to_idx = mesh.node_id_to_index()
    cols = _display_columns(mesh)
    xs = list(cols.keys())
    out = {}
    for i, x in enumerate(xs):
        # Tangent from the next column along +x; at the very last column
        # fall back to the previous one and negate, so t_hat points along
        # +x for every column and the slip sign stays comparable.
        at_end = (i + 1 >= len(xs))
        j = i - 1 if at_end else i + 1
        r = _column_slip(mesh, u, nid_to_idx, cols[x], cols[xs[j]], at_end)
        if r is not None:
            out[x] = r
    return out


def print_column(x, r, title):
    print(f"\n{'=' * 80}")
    print(f" {title}   (x = {x:+.2f} mm)")
    print(f"{'=' * 80}")
    print(f" {'layer':>5} {'material':>8} {'slip [um]':>12} {'step [um]':>12} "
          f"{'shear angle':>13}")
    print(f" {'-' * 76}")
    slip, thk = r["slip"], r["thk"]
    for j in range(len(slip)):
        mat = LAYER_MATERIAL[j - 1] if 0 < j <= N_LAYERS else "-"
        step = (slip[j] - slip[j - 1]) * 1e3 if j > 0 else 0.0
        dthk = (thk[j] - thk[j - 1]) if j > 0 else 0.0
        gam = np.degrees(np.arctan2(step * 1e-3, dthk)) if j > 0 and abs(dthk) > 1e-12 else 0.0
        print(f" {j:>5} {mat:>8} {slip[j] * 1e3:>12.2f} {step:>12.2f} {gam:>12.2f}d")

    psa = sum(slip[j] - slip[j - 1] for j in range(1, len(slip))
              if LAYER_MATERIAL[j - 1] == "PSA")
    pet = sum(slip[j] - slip[j - 1] for j in range(1, len(slip))
              if LAYER_MATERIAL[j - 1] == "PET")
    den = abs(psa) + abs(pet)
    print(f"\n  total staircase (top vs bottom) : {slip[-1] * 1e3:+9.2f} um")
    print(f"  carried by PSA rows             : {psa * 1e3:+9.2f} um "
          f"({100 * abs(psa) / den if den else 0:5.1f}%)")
    print(f"  carried by PET rows             : {pet * 1e3:+9.2f} um "
          f"({100 * abs(pet) / den if den else 0:5.1f}%)")


def plot(profile, save_path):
    xs = np.array(sorted(profile.keys()))
    tot = np.array([profile[x]["slip"][-1] * 1e3 for x in xs])

    fig, axes = plt.subplots(2, 1, figsize=(11, 9))

    ax = axes[0]
    ax.plot(xs, tot, "-", lw=2, color="tab:red")
    ax.axvspan(-10, 10, color="tab:blue", alpha=0.10, label="free hinge span")
    for xb in (-10, 10):
        ax.axvline(xb, color="tab:blue", ls="--", lw=1)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("total staircase offset, top vs bottom [um]")
    ax.set_title("Interlayer slip along the display\n"
                 "(shaded = free hinge span; outside it the bottom row is tied "
                 "to the rigid plate)")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    x_probe = min(profile.keys(), key=lambda x: abs(abs(x) - 10.0))
    r = profile[x_probe]
    P = r["P"]
    colors = ["tab:blue" if m == "PET" else "tab:red" for m in LAYER_MATERIAL]
    for j in range(len(P) - 1):
        ax.plot(P[j:j + 2, 0], P[j:j + 2, 1], "-", lw=4,
                color=colors[j], solid_capstyle="butt")
    ax.plot(P[:, 0], P[:, 1], "k.", ms=6, zorder=3)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"Through-thickness column at x = {x_probe:+.2f} mm (hinge edge), "
                 f"deformed & zoomed\nblue = PET, red = PSA -- "
                 f"staircase = {r['slip'][-1] * 1e3:+.1f} um")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"\n[PNG] slip profile + zoomed column saved: {save_path}")


def main():
    global N_LAYERS, LAYER_MATERIAL

    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", action="store_true",
                    help="reuse examples/ex12_final_u.npy instead of re-solving")
    args = ap.parse_args()

    if args.cached and os.path.exists(U_CACHE):
        print(f"Using cached displacement field: {U_CACHE}")
        u = np.load(U_CACHE)
        result = read_abaqus_input(INP)
        mesh = result.mesh
    else:
        info = run_abaqus_inp_folding()
        u = info["solver"].u
        result = info["result"]
        mesh = result.mesh
        np.save(U_CACHE, u)
        print(f"Saved displacement field for fast re-analysis: {U_CACHE}")

    LAYER_MATERIAL = _laminate_layer_materials(
        mesh, getattr(result, "material_names", {}) or {}
    ) or []
    N_LAYERS = len(LAYER_MATERIAL)

    print("\n\n" + "#" * 80)
    print("# INTERLAYER SHEAR CHECK -- do the soft PSA layers actually slip?")
    print("#" * 80)

    profile = scan(mesh, u)
    xs = sorted(profile.keys())

    for target, title in ((-40.0, "LEFT free tip"),
                          (-10.0, "LEFT hinge edge (plate boundary)"),
                          (0.0, "hinge centre"),
                          (10.0, "RIGHT hinge edge (plate boundary)"),
                          (40.0, "RIGHT free tip")):
        x = min(xs, key=lambda v: abs(v - target))
        print_column(x, profile[x], title)

    plot(profile, os.path.join(HERE, "ex12_interlayer_slip_profile.png"))

    tot = {x: profile[x]["slip"][-1] * 1e3 for x in xs}
    x_max = max(tot, key=lambda v: abs(tot[v]))
    print(f"\n{'=' * 80}")
    print(" VERDICT")
    print(f"{'=' * 80}")
    print(f"  peak |staircase|      : {abs(tot[x_max]):.2f} um  at x = {x_max:+.2f} mm")
    print(f"  at free tips          : {tot[xs[0]]:+.2f} um (x={xs[0]:+.1f}), "
          f"{tot[xs[-1]]:+.2f} um (x={xs[-1]:+.1f})")
    if abs(tot[x_max]) < 1.0:
        print("  -> NO meaningful interlayer shear anywhere: the PSA layers are NOT "
              "doing their job.")
    elif abs(x_max) <= 12.0:
        print("  -> Interlayer shear IS present and peaks near the hinge edge, as "
              "shear-lag theory predicts.\n"
              "     The free tip shows little/none because the bottom row is tied "
              "to the rigid plate\n"
              "     over the full 30mm plate span (lag length ~5.5mm) -- this is "
              "expected, not a defect.")
    else:
        print("  -> Interlayer shear peaks away from the hinge edge; inspect the "
              "profile plot.")


if __name__ == "__main__":
    main()
