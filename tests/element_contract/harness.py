"""
tests/element_contract/harness.py
==================================
The element contract suite's probe layer.

Why this is driven through `DynamicSolver` rather than the element kernels
-------------------------------------------------------------------------
Every earlier element check in this repo pokes a kernel function directly
(`tests/test_ul_tl_consistency.py`, `tests/test_cpe4_element.py`,
`scratch/verify_eas_transpose.py`, ...). That is fine for one element but
it cannot be parametrized across the library, because each kernel has its
own hand-rolled signature -- and, more importantly, it tests something the
solver may not actually be running. Two live defects found while building
this suite are invisible to a kernel-level probe and obvious to a
solver-level one:

* `element_type="CPE4"` has a `_ELEMENT_LARGE_DEF` entry (so
  `element_large_deformation_report()` prints "UL") but **no dispatch
  branch anywhere**, so it silently assembles as plain small-strain B-bar
  `Q4`. AGENTS.md 4.2 / 4.8, verbatim.
* `element_type="Q4_EAS"` passed as a bare STRING (rather than the
  `{pid: name}` dict form) with a single J2 material takes the
  `use_j2_batch` path, which is also plain B-bar `Q4` with no element-type
  awareness at all.

So every probe here goes through `DynamicSolver._assemble()`, with
`element_type` in the **dict form** -- the form `fold_model_config.py` and
both `verification/abaqus_benchmarks/` scripts use, i.e. the one
production actually runs.

Configuration bookkeeping
-------------------------
The solver's Updated-Lagrangian state is two arrays: `_ul_u_ref` (total
displacement at the last converged step) and `_ul_F_n` (total F per
element per Gauss point). An element in UL mode is handed
`coords + u_ref` as its reference and `u - u_ref` as its displacement.
`Probe.evaluate()` takes those explicitly, which is what makes C3
(UL == TL) expressible: the same total deformation reached in one shot
(`u_ref = 0`, `F_n = I`) or in two (`u_ref`/`F_n` carrying the first part).

Purity
------
`_assemble()` MUTATES `eas_alpha`, `hybrid_q`, `state` and
`_eas_local_status`. Every probe call snapshots and restores them, so a
finite-difference sweep re-solves each element-local Newton from the same
starting point -- which is what makes the FD derivative the *condensed*
Jacobian (C8) rather than a path-dependent one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

from dispsolver.mesh import Mesh
from dispsolver.solver.dynamic import DynamicSolver
from dispsolver.material.plastic import J2Plasticity
from dispsolver.material.viscoelastic import ViscoelasticMaterial
from dispsolver.material.arruda_boyce import ArrudaBoyce


# ----------------------------------------------------------------------
# Materials
# ----------------------------------------------------------------------
# Elastic-range J2 (yield set absurdly high) so what these tests measure
# is the ELEMENT, not a return map. Same recipe as tests/test_cpe4_element.py.
E_REF, NU_REF = 4000.0, 0.3

# Arruda-Boyce constants matched to (E_REF, NU_REF) to O(strain), using this
# codebase's own PSA derivation (`mu` is NOT the initial shear modulus --
# see tests/test_cpe4rh_patch.py and AGENTS.md 1.4).
_LAMBDA_M = 3.0
_BETA = (1.0 + 3.0 / (5.0 * _LAMBDA_M ** 2) + 99.0 / (175.0 * _LAMBDA_M ** 4)
         + 513.0 / (875.0 * _LAMBDA_M ** 6) + 42039.0 / (67375.0 * _LAMBDA_M ** 8))
VISCO_PARAMS = {
    "mu": (E_REF / (2.0 * (1.0 + NU_REF))) / _BETA,
    "lambda_m": _LAMBDA_M,
    "K": E_REF / (3.0 * (1.0 - 2.0 * NU_REF)),
}


def make_material(kind: str):
    if kind == "j2":
        return J2Plasticity(E=E_REF, nu=NU_REF, sigma_y0=1e12, H=0.0), None
    if kind == "visco":
        # g_i=0 -> no relaxation, so the response is the pure Arruda-Boyce
        # hyperelastic ground state and the element is what is measured.
        return (ViscoelasticMaterial(ArrudaBoyce(), g_i=[0.0], tau_i=[1.0]),
                dict(VISCO_PARAMS))
    raise ValueError(kind)


# ----------------------------------------------------------------------
# Element registry -- every type this solver actually dispatches
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ElementSpec:
    name: str
    material: str              # "j2" | "visco"
    has_internal_dofs: bool    # EAS alpha / hybrid pressure -> static condensation
    numba: bool                # a Numba lowering exists and is reachable
    note: str = ""


REGISTRY = (
    # --- J2-plasticity family -----------------------------------------
    ElementSpec("Q4", "j2", False, False,
                "small-strain B-bar baseline (rigid BC-driven plates)"),
    ElementSpec("Q4_EAS", "j2", True, True, "Simo-Rifai EAS-4; CPE4I-equivalent"),
    ElementSpec("Q4_COROTATIONAL", "j2", False, False, "whole-element T8 frame"),
    ElementSpec("Q4_COROTATIONAL_EAS", "j2", True, False, "T8 wrapper around Q4_EAS"),
    ElementSpec("Q4_SRI", "j2", False, False, "shear-selective reduced integration"),
    ElementSpec("Q4_COROTATIONAL_SRI", "j2", False, True,
                "PRODUCTION: PET + GLASS layers"),
    # --- finite-strain viscoelastic family ----------------------------
    ElementSpec("Q4_VISCO_SIMO", "visco", False, True, "full integration + F-bar"),
    ElementSpec("CPE4I", "visco", True, True, "PRODUCTION: PSA layers"),
    ElementSpec("CPE4H", "visco", True, True, "element-constant hybrid pressure"),
    ElementSpec("CPE4IH", "visco", True, False, "CPE4I + CPE4H"),
    # --- not dispatched yet; Stage 1 wires it --------------------------
    ElementSpec("CPE4", "visco", False, False,
                "cpe4_jax.py exists but is unwired; dispatch lands in Stage 1"),
)

BY_NAME = {s.name: s for s in REGISTRY}

# Registered so the suite already covers them the moment they are wired,
# but not yet constructible: `DynamicSolver.__init__` now REFUSES an
# element name nothing dispatches rather than letting it fall through to
# plain B-bar Q4 (see dynamic.py::_DISPATCHED_ELEMENT_TYPES). Remove a
# name from this set in the same commit that adds its dispatch branch.
NOT_YET_DISPATCHED = frozenset({"CPE4"})

# Types that are dispatchable but currently broken on the batch path --
# kept OUT of the parametrized suite and characterized by their own
# explicit xfail tests, so the baseline records the breakage instead of
# hiding it behind a skip. See test_contract.py::test_known_broken_*.
# Verified 2026-09-11: the SRI-hybrid pair is NOT in here. Its constructor
# wrapper (`sri_hybrid_wrapper` in dynamic.py) already drops the kernel's
# extra return value to match the 3-value unpack, the same repair the plain
# SRI wrapper carries -- so those two dispatch fine. Only the kernels with
# no such wrapper fail.
KNOWN_BROKEN_BATCH = {
    "Q4_HYBRID": "ValueError: too many values to unpack -- "
                 "compute_corotational_hybrid_j2_contributions_jax returns a "
                 "4-tuple and the batch dispatch unpacks 3. Identical to the "
                 "latent bug dynamic.py's own comment describes fixing for the "
                 "SRI wrapper (`never actually exercised end-to-end`), left "
                 "unfixed here: this element has never run.",
    "Q4_COROTATIONAL_HYBRID": "as Q4_HYBRID (same kernel, same wrapper-less "
                              "dispatch branch)",
    "CPE4RH": "jax.jit static-argname hashing failure under vmap "
              "(TypeError: unhashable type: BatchTracer). Reachable only via "
              "the dict/batch form of element_type; the uniform-string form "
              "takes the sequential path, which is what "
              "tests/test_cpe4rh_patch.py exercises -- so the element is "
              "verified on a path production would not use.",
}


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------
UNIT_AR1 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
# The real free-hinge-span shape: 1.0 x 0.15 -> aspect ratio ~6.7. Matches
# scratch/verify_eas_transpose.py so C6's numbers are comparable to the
# F6 investigation's.
UNIT_AR7 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.15], [0.0, 0.15]])


def rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def rot8(theta: float) -> np.ndarray:
    R = rot2(theta)
    T = np.zeros((8, 8))
    for i in range(4):
        T[2 * i:2 * i + 2, 2 * i:2 * i + 2] = R
    return T


def affine_u(coords: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Nodal displacement realising the homogeneous deformation `F`."""
    H = np.asarray(F) - np.eye(2)
    return (np.asarray(coords) @ H.T).reshape(-1)


def bending_u(coords: np.ndarray, kappa: float, w: float, h: float) -> np.ndarray:
    """Pure-bending displacement field about the element centre.

    `ux = kappa*x*y`, `uy = -kappa*x^2/2` -- the Euler-Bernoulli field whose
    exact strain energy is `0.5 * E' * (h^3/12) * kappa^2 * w` in plane
    strain (`E' = E/(1-nu^2)`).
    """
    cx = coords[:, 0] - w / 2.0
    cy = coords[:, 1] - h / 2.0
    u = np.zeros(8)
    u[0::2] = kappa * cx * cy
    u[1::2] = -0.5 * kappa * cx ** 2
    return u


# ----------------------------------------------------------------------
# The probe
# ----------------------------------------------------------------------
_PROBE_CACHE: dict = {}


def probe_for(spec: ElementSpec, coords: np.ndarray, nlgeom: bool = True,
              elem_jit: str = "jax", thickness: float = 1.0) -> "Probe":
    """A `Probe` for this element, REUSED across geometries.

    Building a fresh `DynamicSolver` per geometry costs a fresh JAX
    compile every time -- the per-pid vmap caches in `DynamicSolver.__init__`
    are local to that call, so nothing is shared between solvers. A C6
    sweep alone is 6 rotations x 3 aspect ratios x 11 elements = ~200
    compiles of a finite-strain kernel, which does not finish in any
    useful time. Since the element's geometry enters only through
    `elem_coords` (and the arrays precomputed from it), moving the nodes
    of one cached solver is equivalent and keeps compiles at one per
    (element, backend) -- JAX's own cache then hits, because the traced
    shapes never change.
    """
    key = (spec.name, nlgeom, elem_jit, float(thickness))
    p = _PROBE_CACHE.get(key)
    if p is None:
        p = Probe(spec, coords, nlgeom=nlgeom, elem_jit=elem_jit,
                  thickness=thickness)
        _PROBE_CACHE[key] = p
    else:
        p.set_coords(coords)
    return p


class Probe:
    """One element, evaluated as a pure function of (u, u_ref, F_n)."""

    def __init__(self, spec: ElementSpec, coords: np.ndarray,
                 nlgeom: bool = True, elem_jit: str = "jax",
                 thickness: float = 1.0):
        self.spec = spec
        self.coords = np.asarray(coords, dtype=np.float64)
        mat, params = make_material(spec.material)
        mesh = Mesh()
        for i, xy in enumerate(self.coords):
            mesh.add_node(i + 1, float(xy[0]), float(xy[1]))
        mesh.add_element(1, [1, 2, 3, 4], "Q4", 0)
        solver = DynamicSolver(
            mesh, {0: mat}, rho=1.0,
            material_params=({0: params} if params is not None else None),
            element_type={0: spec.name},
            nlgeom=nlgeom, elem_jit=elem_jit,
            section_thickness=thickness, mode="quasistatic", verbose=False,
        )
        solver.sta_status = False
        if elem_jit == "numba":
            solver.enable_new_numba_elements = True
        self.solver = solver
        self._pristine = self._snapshot()

    def set_coords(self, coords: np.ndarray) -> "Probe":
        """Move this element's nodes, keeping the compiled kernels.

        Everything geometry-dependent that assembly reads is `elem_coords`
        plus the three arrays `_precompute_reference_geometry()` derives
        from it. `dof_indices`, `K_rows`/`K_cols` and the state arrays are
        topological and unchanged. The lumped mass is not rebuilt: every
        probe runs `mode="quasistatic"`, where it is not assembled.
        """
        self.coords = np.asarray(coords, dtype=np.float64)
        self.solver.elem_coords[0] = self.coords
        (self.solver._B_bar_all, self.solver._weights_all,
         self.solver._dN_dX_all) = self.solver._precompute_reference_geometry()
        return self

    # -- internal mutable state -----------------------------------------
    def _snapshot(self):
        s = self.solver
        return {
            k: (None if getattr(s, k, None) is None else np.copy(getattr(s, k)))
            for k in ("state", "eas_alpha", "hybrid_q", "_eas_local_status",
                      "_ul_F_n", "_ul_u_ref")
        }

    def _restore(self, snap):
        for k, v in snap.items():
            if v is not None:
                setattr(self.solver, k, np.copy(v))

    @property
    def uses_ul(self) -> bool:
        return bool(self.solver._use_ul_for(0))

    def evaluate(self, u, u_ref=None, F_n=None, dt: float = 1.0):
        """(f_int(8,), K(8,8)) for this element. Leaves the solver unchanged.

        `u_ref` (8,) and `F_n` (4,2,2) set the Updated-Lagrangian reference;
        omitted means "reference = original configuration" (Total Lagrangian
        evaluation, even for a UL-capable element).
        """
        s = self.solver
        self._restore(self._pristine)
        if u_ref is not None:
            s._ul_u_ref = np.asarray(u_ref, dtype=np.float64).copy()
        if F_n is not None:
            s._ul_F_n[0] = np.asarray(F_n, dtype=np.float64)
        f, K, _ = s._assemble(np.asarray(u, dtype=np.float64), dt)
        f = np.asarray(f, dtype=np.float64).copy()
        K = np.asarray(K.todense(), dtype=np.float64)
        self._restore(self._pristine)
        return f, K

    def fd_tangent(self, u, u_ref=None, F_n=None, dt: float = 1.0,
                   h: Optional[float] = None) -> np.ndarray:
        """Central-difference Jacobian `df_int/du` of this element's own residual.

        The standing test of AGENTS.md 4.15: the element's tangent must BE
        the Jacobian of the element's own residual -- including through any
        static condensation, which is why the internal DOFs are re-solved
        from the same starting point at every perturbed `u` (see the module
        docstring's purity note).
        """
        u = np.asarray(u, dtype=np.float64)
        if h is None:
            # Mesh-scale-relative step (Dennis & Schnabel 1983, 5.4): a fixed
            # absolute step is itself the B4 defect this suite tests for.
            L = float(np.max(np.abs(self.coords)) or 1.0)
            h = np.sqrt(np.finfo(float).eps) * max(float(np.max(np.abs(u))), L)
        K = np.zeros((8, 8))
        for j in range(8):
            up = u.copy(); up[j] += h
            um = u.copy(); um[j] -= h
            fp, _ = self.evaluate(up, u_ref, F_n, dt)
            fm, _ = self.evaluate(um, u_ref, F_n, dt)
            K[:, j] = (fp - fm) / (2.0 * h)
        return K


def rel(a, b, floor: float = 1e-30) -> float:
    """Max-norm relative difference, scaled by the larger operand."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), floor)
    return float(np.max(np.abs(a - b))) / scale
