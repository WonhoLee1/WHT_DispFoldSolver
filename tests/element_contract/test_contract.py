"""
tests/element_contract/test_contract.py
========================================
The **element contract suite** (C1-C11) -- one parametrized harness applied
to every element type `DynamicSolver` actually dispatches.

Motivation, in one line each:

  C1  rigid-rotation canary                 gross kinematic errors
  C2  strain-then-rotate objectivity        F3 / SRI Cartesian-sampling class
  C3  UL == TL consistency                  F4 (missing PK2 push-forward)
  C4  mid-step `F_n` idempotence            F5 (`_ul_F_n` written per iteration)
  C5  constant-strain patch test, irregular basic correctness
  C6  rotated-reference AR sweep            F6 (`J0^-1` vs `J0^-T`)
  C7  global-axis isotropy (whole problem)  F6 / SRI class, strictest form
  C8  K IS the FD Jacobian of its own residual, on a ROTATED reference   B1
  C9  condensation symmetry                 B2, B3
  C10 JAX == Numba on a tilted+distorted probe   backend drift
  C11 FD step is mesh-invariant             B4 (fixed absolute `h = 1e-6`)

C1/C3/C4/C5 existed in scattered, per-kernel form; C2/C6/C7/C8/C9/C10/C11
did not exist as permanent tests. Five of the six defects shipped since
2026-09-08 would have been caught by C8 alone.

**This file is a characterization baseline, not an aspiration.** Elements
that fail a contract are marked `xfail` with the MEASURED number in the
reason string, so a regression shows up as an `xpass` or as a changed
number, never as a silent drift. Run `python -m tests.element_contract.test_contract`
to print the full table.

See `harness.py` for why every probe goes through `DynamicSolver._assemble`
rather than calling an element kernel directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from dispsolver.mesh import Mesh
from dispsolver.solver.dynamic import DynamicSolver

from .harness import (
    REGISTRY, BY_NAME, KNOWN_BROKEN_BATCH, NOT_YET_DISPATCHED, ElementSpec, Probe,
    probe_for,
    UNIT_AR1, UNIT_AR7, E_REF, NU_REF, VISCO_PARAMS,
    affine_u, bending_u, make_material, rel, rot2, rot8,
)

ALL_NAMES = [s.name for s in REGISTRY]
INTERNAL_DOF_NAMES = [s.name for s in REGISTRY if s.has_internal_dofs]
NUMBA_NAMES = [s.name for s in REGISTRY if s.numba]


@pytest.fixture(autouse=True)
def _skip_unwired(request):
    """Registered-but-unwired types skip every contract until Stage 1."""
    name = request.node.callspec.params.get("name") if hasattr(request.node, "callspec") else None
    if name in NOT_YET_DISPATCHED:
        pytest.skip(f"{name} has no dispatch branch yet (Stage 1)")

# Reference rotations used by C2 / C6 / C7. pi/2 is included deliberately:
# it is the one angle at which a `J0^-1`-vs-`J0^-T` defect goes back to
# EXACTLY 1.000 on a rectangle (both are diagonal again), so an element
# that only ever gets probed at 90 degrees looks perfect.
ROTATIONS = (0.1, 0.5, 1.0, np.pi / 2)
REF_ROT_DEG = (0.0, 10.0, 30.0, 45.0, 60.0, 90.0)

_RNG = np.random.default_rng(0)
U_SMALL = _RNG.uniform(-1.0, 1.0, size=8) * 4e-4


# ======================================================================
# Known-bad characterization -- recorded, not hidden
# ======================================================================
# Every entry is a MEASURED number from the 2026-09-11 baseline run
# (`python -m tests.element_contract.test_contract`), with why. An entry
# whose test starts passing shows up as an `xpass` -- that is the signal to
# delete the entry, not to widen it. Categories:
#
#   BY DESIGN   the element is doing what its formulation says; the
#               contract does not bind for it. Documented, not a defect.
#   DEFECT      a real inconsistency, characterized here and left for the
#               stage that owns it. NOT fixed in Stage 0 -- Stage 0 changes
#               no formulation.
_XFAIL = {
    # ---- C2: objectivity under a superposed rigid rotation ----------
    ("C2", "Q4"): (1.53e+00, "BY DESIGN: small-strain B-bar. Its B-operator is "
                   "built on the reference config and never updated, so it is "
                   "simply not a finite-rotation element. It drives only the "
                   "Dirichlet-prescribed rigid plates (AGENTS.md 4.10)."),

    # ---- C6: bending stiffness must not depend on orientation ---------
    # Measured at AR 7. See the __main__ table for the full AR 7.5/15/30
    # sweep, which is the R3 measurement Stage 4's premise rests on.
    ("C6", "Q4"): (6.98e-01, "DEFECT, NEW: plain Q4's TANGENT is not "
                   "frame-covariant, although its internal force is (C7 passes "
                   "at 3.6e-14). Traced 2026-09-11 to J2Plasticity's NumPy "
                   "elastic tangent, not to the element -- see "
                   "test_material_tangent.py. Bending energy swings 69.3 -> "
                   "20.9 -> 69.3 over 0-45-90 degrees."),
    ("C6", "Q4_SRI"): (1.58e+01, "DEFECT, known: the C7 anisotropy, seen as "
                       "bending stiffness. 1.225 axis-aligned (essentially "
                       "locking-free) -> 20.6 at 45 degrees."),
    ("C6", "Q4_COROTATIONAL_SRI"): (1.58e+01, "DEFECT, known: bit-identical to "
                                    "Q4_SRI. PRODUCTION element (PET + GLASS) -- "
                                    "this is the R3 measurement; see the report."),
    ("C6", "CPE4IH"): (4.19e+01, "DEFECT, NEW: 0.467 at 0 degrees (too SOFT) "
                       "to 20.0 at 90 degrees. Same kernel family as C7/C8's "
                       "CPE4IH findings. Stage 3 owns it."),

    # ---- C7: global-axis isotropy ------------------------------------
    ("C7", "Q4_SRI"): (6.71e-02, "DEFECT, known: shear sampled as the raw "
                       "Cartesian off-diagonal of dU/dX -- not a frame-invariant "
                       "object (AGENTS.md 4.14). In-house device, no Abaqus "
                       "counterpart."),
    ("C7", "Q4_COROTATIONAL_SRI"): (6.71e-02, "DEFECT, known: identical to "
                                    "Q4_SRI -- the co-rotational wrapper removes "
                                    "the RELATIVE rotation only, leaving the "
                                    "element's absolute orientation intact. "
                                    "Measured bit-identical to Q4_SRI here, "
                                    "confirming AGENTS.md 4.14."),
    ("C7", "CPE4IH"): (5.68e-02, "DEFECT, NEW (found by this suite): CPE4IH is "
                       "anisotropic at the same order as the SRI family, while "
                       "its siblings CPE4I/CPE4H are at 2e-14. Not previously "
                       "reported. Stage 3 owns it."),

    # ---- C8: K is the FD Jacobian of the element's own residual -------
    ("C8", "Q4"): (9.46e-01, "BY DESIGN: as C2 -- a small-strain tangent "
                   "evaluated at finite strain."),
    ("C8", "Q4_COROTATIONAL"): (1.26e-01, "BY DESIGN: modified Newton. The "
                                "analytic material-only tangent deliberately "
                                "drops the dT8/du rotational geometric term "
                                "(AGENTS.md 4.4 -- the full autodiff tangent "
                                "never finished compiling). Costs Newton "
                                "iterations, not correctness."),
    ("C8", "Q4_SRI"): (1.13e-01, "BY DESIGN: modified Newton, as "
                       "Q4_COROTATIONAL."),
    ("C8", "Q4_COROTATIONAL_SRI"): (1.13e-01, "BY DESIGN: modified Newton, as "
                                    "Q4_COROTATIONAL."),
    ("C8", "CPE4IH"): (3.44e-03, "DEFECT, NEW: CPE4IH's tangent is not the "
                       "Jacobian of its own residual, while CPE4I/CPE4H are at "
                       "1.3e-09 on the identical probe -- so this is NOT the "
                       "modified-Newton exemption, it is the B1/B2/B3 class "
                       "surviving in the one kernel B2's fix touched last. "
                       "Stage 3 owns it."),

    # ---- C9: condensation symmetry -----------------------------------
    ("C9", "Q4_VISCO_SIMO"): (7.67e-03, "BY DESIGN: F-bar. The J_bar/J coupling "
                              "makes every Gauss point's stress depend on every "
                              "node, and the resulting tangent is genuinely "
                              "unsymmetric -- a documented property of the "
                              "method (de Souza Neto et al. 1996), not a lost "
                              "K_au = K_ua^T. The EAS/hybrid siblings, which "
                              "cure locking differently, are at 1e-16."),

    # ---- C10: JAX == Numba -------------------------------------------
    # FIXED 2026-09-12. Three entries stood here at 1.54e-03 / 1.61e-03 /
    # 1.75e-03. All three were root-caused to the SAME sentence -- "which
    # deformation gradient does the B-operator differentiate?" -- i.e. the
    # F4/F6/B1/B2/B3 class, once more:
    #   Q4_COROTATIONAL_SRI  two different co-rotational frames. JAX takes the
    #                        xi-edge pair alone; the Numba copy averaged both
    #                        edge rotations. Invisible on any full-integration
    #                        Green-Lagrange element (the frame cancels -- F1's
    #                        "bitwise TL" result), but SRI samples shear in
    #                        fixed Cartesian components, so the frame choice
    #                        changes the answer.        -> 4.67e-14
    #   CPE4I                the Numba mirror was never ported onto B2: it
    #                        built B_L from the COMPATIBLE gradient, dropping
    #                        the alpha-dependent half of dE_inc/du, and
    #                        contracted f_a against the un-pushed S_v. Neither
    #                        vanishes at F_n = I.        -> 1.70e-15
    #   Q4_VISCO_SIMO        B_L built from `Fbar` instead of the real F. In
    #                        F-bar the modified gradient enters the
    #                        CONSTITUTIVE call only.     -> 2.00e-15
    # Residual K errors (2.3e-06 / 8.0e-08 / 6.8e-08) are the FD-vs-autodiff
    # floor the C10 tangent tolerance already allows, not drift.

    # ---- C11: the FD step must be mesh-invariant ----------------------
    # FIXED 2026-09-12. `("C11", "CPE4H")` stood here at ratio 8.0 (2.61e-06 /
    # 1.31e-06 / 6.54e-07 / 3.27e-07 across x1/x2/x4/x8 -- error exactly
    # inverse in element size, B4's h/L signature). `q4_visco_hybrid_up_numba`
    # now takes the per-column Dennis & Schnabel step; measured 4.11e-08 flat,
    # ratio 1.00.
    #
    # Worth recording as a method note: CPE4I's C11 row was flat BEFORE the
    # C10 fix and grew the ratio-8.0 signature (6.63e-06 ... 8.28e-07) the
    # moment its force agreed with JAX. C11 measures |K_numba - K_jax|, so a
    # large force-formulation disagreement was swamping the FD term and making
    # a real B4 defect look absent. A contract can be masked by another
    # contract's failure; a flat C11 row is only evidence once C10 passes.
    # `q4_visco_eas_numba` was fixed in the same pass -> 1.01.
}


def _check(contract: str, name: str, measured: float, tol: float, what: str):
    """Assert `measured < tol`, unless this pair is a characterized failure.

    A characterized pair is `xfail`ed with its baseline number alongside the
    fresh one, so a drift in a known-bad element is visible in the test log
    rather than silently absorbed.
    """
    entry = _XFAIL.get((contract, name))
    if entry is not None:
        baseline, why = entry
        pytest.xfail(f"{contract} {name}: {measured:.3e} "
                     f"(baseline {baseline:.3e}) -- {why}")
    assert measured < tol, f"{contract} {name}: {what}, measured {measured:.3e}"


def _spec(name: str) -> ElementSpec:
    return BY_NAME[name]


def _probe(name: str, coords=UNIT_AR7, **kw) -> Probe:
    """Cached per (element, backend) -- see harness.probe_for."""
    return probe_for(_spec(name), coords, **kw)


# ======================================================================
# C1 -- rigid-rotation canary
# ======================================================================
def c1_rigid_rotation(name: str) -> float:
    """max over test angles of |f| / (|K| |u|) for a pure rigid rotation."""
    p = _probe(name)
    worst = 0.0
    for th in ROTATIONS:
        u = affine_u(p.coords, rot2(th))
        f, K = p.evaluate(u)
        scale = float(np.max(np.abs(K))) * float(np.max(np.abs(u)))
        worst = max(worst, float(np.max(np.abs(f))) / max(scale, 1e-30))
    return worst


@pytest.mark.parametrize("name", ALL_NAMES)
def test_c1_rigid_rotation(name):
    # Baseline max across the library: 2.66e-15.
    _check("C1", name, c1_rigid_rotation(name), 1e-12,
           "a pure rigid rotation produces internal force")


# ======================================================================
# C2 -- strain-then-rotate objectivity
# ======================================================================
def c2_strain_then_rotate(name: str) -> float:
    """f(R F0) must equal T8(R) f(F0) on the SAME reference.

    PK2 is invariant under a superposed rigid rotation (`C` is unchanged),
    while `f_a = integral F S grad N dV0` rotates with it -- so a
    frame-covariant element satisfies this exactly at ANY strain level,
    not only at zero strain like C1.
    """
    p = _probe(name)
    F0 = np.array([[1.08, 0.05], [0.02, 0.94]])
    f0, _ = p.evaluate(affine_u(p.coords, F0))
    worst = 0.0
    for th in ROTATIONS:
        fr, _ = p.evaluate(affine_u(p.coords, rot2(th) @ F0))
        worst = max(worst, rel(fr, rot8(th) @ f0))
    return worst


@pytest.mark.parametrize("name", ALL_NAMES)
def test_c2_strain_then_rotate(name):
    # Baseline: everything except Q4 is at or below 2.85e-12. NOTE the SRI
    # family PASSES this one (9.56e-15) -- a superposed rotation does not
    # change the element's own orientation, so its Cartesian shear sampling
    # is untouched. C7, which rotates the whole problem, is the test that
    # catches it. Keeping both is the point.
    _check("C2", name, c2_strain_then_rotate(name), 1e-11,
           "not objective under a superposed rigid rotation")


# ======================================================================
# C3 -- UL == TL consistency
# ======================================================================
_C3_CASES = (
    ("stretch_then_shear",
     np.array([[1.20, 0.0], [0.0, 0.90]]), np.array([[1.0, 0.03], [0.0, 1.0]])),
    ("rotation_then_stretch",
     rot2(1.2), np.array([[1.05, 0.0], [0.0, 1.0]])),
)


def c3_ul_equals_tl(name: str) -> float:
    p = _probe(name)
    if not p.uses_ul:
        return 0.0
    worst = 0.0
    for _, F_n, F_extra in _C3_CASES:
        F_total = F_extra @ F_n
        u_total = affine_u(p.coords, F_total)
        u_ref = affine_u(p.coords, F_n)
        f_tl, _ = p.evaluate(u_total)
        f_ul, _ = p.evaluate(u_total, u_ref=u_ref, F_n=np.stack([F_n] * 4))
        worst = max(worst, rel(f_tl, f_ul))
    return worst


@pytest.mark.parametrize("name", ALL_NAMES)
def test_c3_ul_equals_tl(name):
    p = _probe(name)
    if not p.uses_ul:
        pytest.skip(f"{name} runs Total Lagrangian under NLGEOM (_use_ul_for=False)")
    # Baseline max: 5.18e-15.
    _check("C3", name, c3_ul_equals_tl(name), 1e-11,
           "UL != TL -- a missing PK2 push-forward (F4 class)")


# ======================================================================
# C4 -- mid-step F_n idempotence
# ======================================================================
def c4_fn_idempotence(name: str) -> float:
    """Two back-to-back assemblies at the same `u` must agree exactly.

    F5 was `_ul_F_n` being written on EVERY assembly call rather than only
    on convergence, so Newton iteration 2 in the same step evaluated
    `F_inc @ (F_inc @ F_n_conv)`. Deliberately does NOT restore solver
    state between the two calls -- that is the whole point.
    """
    p = _probe(name)
    F_n = np.array([[1.10, 0.04], [-0.02, 0.95]])
    u_ref = affine_u(p.coords, F_n)
    u = affine_u(p.coords, np.array([[1.02, 0.01], [0.0, 1.01]]) @ F_n)
    s = p.solver
    s._ul_u_ref = u_ref.copy()
    s._ul_F_n[0] = np.stack([F_n] * 4)
    f1, K1, _ = s._assemble(u, 1.0)
    f2, K2, _ = s._assemble(u, 1.0)
    return max(rel(np.asarray(f1), np.asarray(f2)),
               rel(np.asarray(K1.todense()), np.asarray(K2.todense())))


@pytest.mark.parametrize("name", ALL_NAMES)
def test_c4_fn_idempotence(name):
    # Baseline max: 1.40e-13 (CPE4IH); most are exactly 0.
    _check("C4", name, c4_fn_idempotence(name), 1e-11,
           "assembly is not idempotent within a step (F5 class)")


# ======================================================================
# C5 -- constant-strain patch test on an IRREGULAR mesh
# ======================================================================
def _patch_solver(spec: ElementSpec, elem_jit="jax"):
    """2x2 quads over a unit square with the interior node pulled off-centre.

    The irregularity matters: on a parallelogram mesh a centroid-sampled
    volumetric term and an element-average one coincide exactly (AGENTS.md
    4.14's recurring lesson), so a regular patch test cannot see the
    difference.
    """
    xs = [0.0, 0.55, 1.0]
    ys = [0.0, 0.42, 1.0]
    mesh = Mesh()
    nid = {}
    k = 1
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            mesh.add_node(k, x, y)
            nid[(i, j)] = k
            k += 1
    eid = 1
    for j in range(2):
        for i in range(2):
            mesh.add_element(eid, [nid[(i, j)], nid[(i + 1, j)],
                                   nid[(i + 1, j + 1)], nid[(i, j + 1)]], "Q4", 0)
            eid += 1
    mat, params = make_material(spec.material)
    s = DynamicSolver(mesh, {0: mat}, rho=1.0,
                      material_params=({0: params} if params is not None else None),
                      element_type={0: spec.name}, nlgeom=True, elem_jit=elem_jit,
                      mode="quasistatic", verbose=False)
    s.sta_status = False
    coords = np.array([[mesh.get_node(n).x, mesh.get_node(n).y]
                       for n in s.sorted_nids])
    interior = [i for i, n in enumerate(s.sorted_nids) if n == nid[(1, 1)]]
    return s, coords, interior


def c5_patch(name: str) -> float:
    """Residual at the interior node / max boundary reaction, under an
    exactly affine displacement imposed at EVERY node."""
    s, coords, interior = _patch_solver(_spec(name))
    F = np.array([[1.002, 0.0015], [0.0, 0.9985]])
    u = (coords @ (F - np.eye(2)).T).reshape(-1)
    f, _, _ = s._assemble(u, 1.0)
    f = np.asarray(f)
    idofs = np.array([[2 * i, 2 * i + 1] for i in interior]).reshape(-1)
    return float(np.max(np.abs(f[idofs]))) / max(float(np.max(np.abs(f))), 1e-30)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_c5_patch(name):
    # Baseline max: 5.33e-13.
    _check("C5", name, c5_patch(name), 1e-10,
           "interior residual under an exactly affine displacement")


# ======================================================================
# C6 -- rotated-reference aspect-ratio sweep (bending stiffness ratio)
# ======================================================================
def c6_bending_ratio(name: str, ar: float, ref_rot_deg: float,
                     kappa: float = 1e-4) -> float:
    """U_element / U_Euler-Bernoulli for a pure-bending field.

    1.000 = exact. > 1 is artificial bending stiffness (shear locking);
    a value that MOVES with `ref_rot_deg` is a frame-covariance defect --
    the two are independent failure modes and this sweep separates them,
    which is exactly what AGENTS.md 4.1's axis-aligned-only sweep could
    not do (4.14).
    """
    h = 1.0 / ar
    base = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, h], [0.0, h]])
    R = rot2(np.deg2rad(ref_rot_deg))
    p = _probe(name, coords=base @ R.T)
    u = (bending_u(base, kappa, 1.0, h).reshape(4, 2) @ R.T).reshape(-1)
    _, K = p.evaluate(u)
    U = 0.5 * float(u @ K @ u)
    U_exact = 0.5 * (E_REF / (1.0 - NU_REF ** 2)) * (h ** 3 / 12.0) * kappa ** 2 * 1.0
    return U / U_exact


# Production column aspect ratios. `gen_ex12_inp.py`'s `_graded_display_x()`
# grades the display's x-columns to 0.25mm at the four transition locations,
# 0.5mm through the free hinge span and 1.0mm under the rigid plate body;
# over a PET row of roughly 0.033mm that is a width/height ratio of about
# 7.5 / 15 / 30 depending on where in the mesh the element sits.
PRODUCTION_AR = (7.5, 15.0, 30.0)
_AR_PROVENANCE = {
    7.5: "0.25mm column -- graded transition bands (free tips, plate/hinge boundary)",
    15.0: "0.5mm column -- the free hinge span, where the fold curvature develops",
    30.0: "1.0mm column -- under the rigid plate body",
}


@pytest.mark.parametrize("name", ALL_NAMES)
def test_c6_frame_covariance(name):
    """The RATIO must not depend on how the element is oriented.

    This is the frame-covariance half of C6. Whether the ratio is 1.000 at
    0 degrees (i.e. whether the element shear-locks at all) is a separate,
    documented property of each element, reported by the `__main__` table
    and deliberately NOT asserted here -- CPE4/CPE4H are SUPPOSED to lock
    in bending, that is why CPE4I exists.
    """
    ratios = [c6_bending_ratio(name, 7.0, a) for a in REF_ROT_DEG]
    spread = (max(ratios) - min(ratios)) / max(abs(ratios[0]), 1e-30)
    _check("C6", name, spread, 1e-6,
           f"bending stiffness depends on element orientation "
           f"(ratios {['%.3f' % r for r in ratios]} over 0-90 deg)")


# ======================================================================
# C7 -- global-axis isotropy (reference AND displacement rotated)
# ======================================================================
def c7_isotropy(name: str) -> float:
    p0 = _probe(name)
    f0, _ = p0.evaluate(U_SMALL)
    worst = 0.0
    for th in ROTATIONS:
        R = rot2(th)
        pr = _probe(name, coords=UNIT_AR7 @ R.T)
        ur = (U_SMALL.reshape(4, 2) @ R.T).reshape(-1)
        fr, _ = pr.evaluate(ur)
        worst = max(worst, rel(fr, rot8(th) @ f0))
    return worst


@pytest.mark.parametrize("name", ALL_NAMES)
def test_c7_isotropy(name):
    # Baseline: 7.14e-14 for everything that passes; the three failures are
    # in _XFAIL with their numbers.
    _check("C7", name, c7_isotropy(name), 1e-11,
           "global-axis isotropy violated -- f(rotated problem) != T8 f")


# ======================================================================
# C8 -- K IS the FD Jacobian of the element's own residual (rotated ref)
# ======================================================================
def c8_fd_jacobian(name: str, ref_rot_deg: float = 39.0) -> float:
    R = rot2(np.deg2rad(ref_rot_deg))
    p = _probe(name, coords=UNIT_AR7 @ R.T)
    u = (U_SMALL.reshape(4, 2) @ R.T).reshape(-1) * 25.0
    _, K = p.evaluate(u)
    K_fd = p.fd_tangent(u)
    return rel(K, K_fd)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_c8_fd_jacobian(name):
    # Baseline for the consistent-tangent elements: 1.1e-09 to 6.1e-09, i.e.
    # the central-difference floor. The modified-Newton elements sit at
    # 1.1e-01 to 1.3e-01 BY DESIGN and are in _XFAIL with that reason -- the
    # tolerance below is not what separates them, the table is.
    _check("C8", name, c8_fd_jacobian(name), 1e-7,
           "tangent is not the Jacobian of its own residual on a "
           "39-degree-rotated reference (B1/B2/B3 class)")


# ======================================================================
# C9 -- condensation symmetry
# ======================================================================
def c9_symmetry(name: str, ref_rot_deg: float = 39.0) -> float:
    """|K - K^T| / |K| for the CONDENSED element tangent.

    The observable consequence of `K_au = K_ua^T` (and `K_qu = K_uq^T`):
    if `f_u` and `f_alpha` are gradients of ONE potential, the condensed
    `K_uu - K_ua K_aa^-1 K_au` is symmetric. B2/B3 were a partial
    push-forward that made them gradients of two different functionals --
    measured 4.0e-2 (TL) / 7.8e-2 (UL) on the EAS path and 54% on CPE4IH.
    Asserted here rather than claimed in a comment, per AGENTS.md 4.15.
    """
    R = rot2(np.deg2rad(ref_rot_deg))
    p = _probe(name, coords=UNIT_AR7 @ R.T)
    u = (U_SMALL.reshape(4, 2) @ R.T).reshape(-1) * 25.0
    _, K = p.evaluate(u)
    return float(np.max(np.abs(K - K.T))) / max(float(np.max(np.abs(K))), 1e-30)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_c9_condensation_symmetry(name):
    # Baseline max among the symmetric-by-construction elements: 1.74e-12.
    _check("C9", name, c9_symmetry(name), 1e-10,
           "condensed tangent is asymmetric -- the K_au = K_ua^T the "
           "condensation assumes does not hold (B2/B3 class)")


# ======================================================================
# C10 -- JAX == Numba on a tilted + distorted probe
# ======================================================================
def c10_jax_numba(name: str) -> tuple:
    R = rot2(np.deg2rad(23.0))
    coords = (UNIT_AR7 @ R.T) + np.array([[0.0, 0.0], [0.0, 0.012],
                                          [-0.03, 0.0], [0.0, -0.008]])
    u = (U_SMALL.reshape(4, 2) @ R.T).reshape(-1) * 10.0
    f_j, K_j = _probe(name, coords=coords, elem_jit="jax").evaluate(u)
    f_n, K_n = _probe(name, coords=coords, elem_jit="numba").evaluate(u)
    return rel(f_j, f_n), rel(K_j, K_n)


@pytest.mark.parametrize("name", NUMBA_NAMES)
def test_c10_jax_numba_agreement(name):
    ef, eK = c10_jax_numba(name)
    if ef == 0.0 and eK == 0.0:
        # Bit-identical to the JAX path means the Numba branch was never
        # taken, not that the lowering agrees -- `Q4_EAS`'s Numba branch is
        # gated on `not self.ul_mode`, so it is unreachable under NLGEOM,
        # which is how every probe here (and production) runs. Reporting
        # that as a pass would be the vacuous-green failure this whole
        # suite exists to prevent.
        pytest.skip(f"{name}: Numba branch not reachable under NLGEOM "
                    f"(gated on `not ul_mode`) -- nothing was compared")
    _check("C10", name, ef, 1e-12, "JAX and Numba internal forces differ")
    # Several Numba lowerings build K by finite difference where the JAX one
    # autodiffs, so the tangent tolerance is the FD floor, not machine eps.
    # Baseline for the one lowering that agrees on force (CPE4H): 2.44e-06.
    _check("C10", name, eK, 1e-5, "JAX and Numba tangents differ")


# ======================================================================
# C11 -- the FD step is mesh-invariant
# ======================================================================
def c11_mesh_invariant_fd(name: str) -> list:
    """|K_numba - K_jax| / |K_jax| across a factor-8 range of element sizes.

    B4: a Numba kernel differencing with a FIXED ABSOLUTE `h = 1e-6` has a
    truncation error scaling as `h / L_elem`, so the error DOUBLES on every
    uniform refinement -- measured 9.3e-8 / 1.7e-7 / 3.3e-7 / 6.5e-7 at
    n=4/8/16/32 of Cook's membrane. A flat row here is the fix.
    """
    out = []
    for scale in (1.0, 2.0, 4.0, 8.0):
        coords = UNIT_AR7 * scale
        u = U_SMALL * scale
        _, K_j = _probe(name, coords=coords, elem_jit="jax").evaluate(u)
        _, K_n = _probe(name, coords=coords, elem_jit="numba").evaluate(u)
        out.append(rel(K_j, K_n))
    return out


@pytest.mark.parametrize("name", NUMBA_NAMES)
def test_c11_mesh_invariant_fd(name):
    errs = c11_mesh_invariant_fd(name)
    if max(errs) == 0.0:
        pytest.skip(f"{name}: Numba branch not reachable under NLGEOM -- "
                    f"nothing was compared (see C10)")
    ratio = max(errs) / max(min(errs), 1e-16)
    # A correct, relative FD step gives a FLAT row (measured: exactly flat
    # for Q4_COROTATIONAL_SRI / CPE4I / Q4_VISCO_SIMO). A fixed absolute
    # step gives error proportional to h/L, i.e. a factor-8 spread across
    # this factor-8 size sweep.
    _check("C11", name, ratio, 4.0,
           f"tangent error scales with element size ({['%.2e' % e for e in errs]}) "
           f"-- a fixed ABSOLUTE FD step (B4 class)")


# ======================================================================
# Dispatch integrity -- the tripwire for the CPE4 defect class
# ======================================================================
def test_element_large_def_table_matches_dispatch():
    """Every `_ELEMENT_LARGE_DEF` key must actually be dispatched.

    `"CPE4"` was declared `supports_ul=True` and reported as
    "UL (rotated reference)" by `element_large_deformation_report()` while
    having NO dispatch branch at all, so it assembled as plain
    small-strain B-bar Q4. AGENTS.md 4.2/4.8's silent-fallthrough class.
    The solver now asserts this at construction; this test is the
    permanent guard on the assert itself.
    """
    from dispsolver.solver.dynamic import _ELEMENT_LARGE_DEF, _dispatched_element_types
    undispatched = sorted(set(_ELEMENT_LARGE_DEF) - _dispatched_element_types())
    assert not undispatched, (
        f"declared in _ELEMENT_LARGE_DEF but never dispatched: {undispatched} "
        f"-- element_large_deformation_report() would lie about these")


@pytest.mark.parametrize("bad", ["CPE4S", "CPE4SH", "CPE4S_COR", "CPE4SH_COR"])
def test_invented_abaqus_names_are_rejected(bad):
    """`CPE4S`/`CPE4SH` are not Abaqus elements and must not resolve.

    No Abaqus element applies selective reduced integration to the SHEAR
    term (SRI in Abaqus is volumetric-only; the complete plane-strain list
    is CPE4/CPE4R/CPE4H/CPE4RH/CPE4I/CPE4IH plus the CPE3/CPE6/CPE8
    families -- AUG 28.1.3). Presenting this repo's invented SRI device
    under an Abaqus-shaped name in a `.inp` deck asserts an element that
    does not exist, so the deck must ERROR rather than silently resolve.
    """
    from dispsolver.solver.dynamic import _ELEMENT_TYPE_ALIASES
    from dispsolver.io.model_builder import ABAQUS_ELEMENT_MAP
    assert bad not in _ELEMENT_TYPE_ALIASES
    assert bad not in ABAQUS_ELEMENT_MAP


def test_hybrid_eas_kernel_is_dead():
    """Characterization: `Q4_HYBRID_EAS` assembles an all-zero element.

    A 2026-07-30 dev_log found `q4_hybrid_jax.py`'s CR-hybrid-EAS kernel was
    "a functional duplicate of CR-EAS ... zero actual pressure/hybrid
    content". Re-checked 2026-09-11 (Stage 0 item 4) and it is WORSE than a
    duplicate: the kernel NaNs internally and its NaN guard zeroes the
    output, so `compute_corotational_hybrid_eas_contributions_jax` returns
    `f_int == 0`, `K_e == 0` and a NaN `alpha`, against CR-EAS's
    `|f| = 8.28`, `|K| = 1.16e4` on the same input.

    An element that contributes nothing while Newton converges happily is
    the AGENTS.md 4.8 signature. Recorded here so the state is a documented
    fact rather than a surprise; `Q4_HYBRID_EAS` /
    `Q4_COROTATIONAL_HYBRID_EAS` must not be used until this is fixed or
    the type removed (both are DELETE candidates, refactor plan 2.2).
    """
    import jax.numpy as jnp
    from dispsolver.element.q4_hybrid_jax import (
        compute_corotational_hybrid_eas_contributions_jax as hyb_eas)
    from dispsolver.element.q4_corotational_eas_jax import (
        compute_corotational_eas_j2_contributions_jax as cr_eas)
    from dispsolver.material.plastic import J2Plasticity

    m = J2Plasticity(E=E_REF, nu=NU_REF, sigma_y0=1e12, H=0.0)
    coords = jnp.asarray(UNIT_AR7)
    u = jnp.asarray(U_SMALL)
    state = jnp.tile(jnp.array([1.0, 0.0, 0.0, 1.0, 0.0]), (4, 1))
    a0 = jnp.zeros(4)
    args = (coords, u, a0, state, m.lam, m.mu, m.sigma_y0, m.H, 1.0)
    f_h, K_h = (np.asarray(x) for x in hyb_eas(*args)[:2])
    f_e, K_e = (np.asarray(x) for x in cr_eas(*args)[:2])

    assert np.max(np.abs(f_e)) > 1.0, "CR-EAS control produced no force"
    assert np.max(np.abs(f_h)) == 0.0, (
        "Q4_HYBRID_EAS now produces a force -- if it was fixed, delete this "
        "characterization test and give it real contracts")
    assert np.max(np.abs(K_h)) == 0.0
    assert np.max(np.abs(K_e)) > 1.0


@pytest.mark.parametrize(
    "name,reason",
    sorted(KNOWN_BROKEN_BATCH.items()),
    ids=sorted(KNOWN_BROKEN_BATCH),   # the reason strings are paragraphs
)
def test_known_broken_batch_dispatch(name, reason):
    """Characterization: these types are reachable but fail on the batch path.

    Recorded rather than skipped so the baseline states what is broken.
    Fixing one turns its test into an `xpass`, which is the signal to
    remove it from `KNOWN_BROKEN_BATCH`.
    """
    from .harness import ElementSpec
    mat_kind = "visco" if name == "CPE4RH" else "j2"
    spec = ElementSpec(name, mat_kind, True, False, reason)
    with pytest.raises(Exception):
        Probe(spec, UNIT_AR7).evaluate(U_SMALL)


# ======================================================================
# Characterization table
# ======================================================================
def _table():
    def safe(fn, *a):
        try:
            return fn(*a)
        except Exception as e:                                   # noqa: BLE001
            return f"ERR:{type(e).__name__}"

    def fmt(v):
        return f"{v:9.2e}" if isinstance(v, float) else f"{str(v)[:9]:>9s}"

    print(f"\n{'element':24s} {'C1':>9s} {'C2':>9s} {'C3':>9s} {'C4':>9s} "
          f"{'C5':>9s} {'C7':>9s} {'C8':>9s} {'C9':>9s}")
    print("-" * 104)
    for s in REGISTRY:
        row = [safe(f, s.name) for f in (c1_rigid_rotation, c2_strain_then_rotate,
                                         c3_ul_equals_tl, c4_fn_idempotence,
                                         c5_patch, c7_isotropy, c8_fd_jacobian,
                                         c9_symmetry)]
        print(f"{s.name:24s} " + " ".join(fmt(v) for v in row))

    print(f"\nC6 -- bending stiffness ratio (1.000 = exact), "
          f"reference rotated in-plane")
    for ar in PRODUCTION_AR:
        print(f"\n  AR = {ar}  ({_AR_PROVENANCE[ar]})")
        print(f"  {'element':24s} " + "".join(f"{a:8.0f}d" for a in REF_ROT_DEG))
        for s in REGISTRY:
            vals = [safe(c6_bending_ratio, s.name, ar, a) for a in REF_ROT_DEG]
            print(f"  {s.name:24s} " + "".join(
                f"{v:9.3f}" if isinstance(v, float) else f"{str(v)[:8]:>9s}"
                for v in vals))

    print(f"\nC10 / C11 -- Numba lowerings")
    print(f"  {'element':24s} {'f err':>11s} {'K err':>11s}   C11 across x1..x8")
    for s in REGISTRY:
        if not s.numba:
            continue
        r = safe(c10_jax_numba, s.name)
        c11 = safe(c11_mesh_invariant_fd, s.name)
        if isinstance(r, tuple):
            head = f"{r[0]:11.2e} {r[1]:11.2e}"
        else:
            head = f"{str(r)[:23]:>23s}"
        tail = ("  ".join(f"{e:.2e}" for e in c11)
                if isinstance(c11, list) else str(c11))
        print(f"  {s.name:24s} {head}   {tail}")


if __name__ == "__main__":
    _table()
