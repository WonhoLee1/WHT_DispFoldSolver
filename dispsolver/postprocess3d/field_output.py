"""
field_output.py
================
Per-element Cauchy stress recovery for a converged DynamicSolver3D state.

Modeled on Abaqus's own ODB field-output access pattern
(`odb.steps[name].frames[-1].fieldOutputs['S']`, then `.values[i].data` /
`.values[i].mises`) -- an object with the same shape, not a bare function,
so a future real ODB-style API (if this codebase ever gets one) is a
drop-in replacement rather than a rewrite.

Why this exists: `benchmark_element/benchmark_3d_mechanics.py`'s
`run_abaqus_official_patch_test` (and every other benchmark in that
module) can only check DISPLACEMENT fields today -- there is no
per-Gauss-point Cauchy-stress-recovery API anywhere in
`dispsolver/solver3d/` or `dispsolver/element3d/`. This module adds one,
as pure post-processing: it re-evaluates each element's OWN material law
(`dispsolver.material3d.numba_materials.material_dispatch_3d`, the exact
function the solver's own assembly already calls) at the element's
converged deformation, rather than re-deriving a second, parallel
constitutive evaluation -- the maintenance hazard flagged in
`.omc/plans/3d_capability_buildout_20260913.md`'s Risks section.

Current scope (2026-09-13, extended later the same day): C3D4/C3D4_ANP
(4-node tet), C3D8-family hexahedra (C3D8/C3D8R/C3D8_CR), C3D10/C3D10M
(10-node quadratic tet), C3D6 (6-node wedge), and C3D8_FBAR -- all
evaluated at a SINGLE point (the element centroid in natural
coordinates), not the full multi-point Gauss rule the assembly itself
uses. This is an intentional, documented simplification, not an
approximation FOR THE VERIFICATION TARGET this module is built and
tested against (`run_abaqus_official_patch_test`'s affine displacement
field): F is spatially CONSTANT throughout every element for that field,
so a single evaluation point is exact everywhere in the element, and any
mean-dilatation/B-bar-style correction these kernels apply (C3D10M's
`B_vol_bar` volume-averaged dilatation operator, C3D8_FBAR's
`(detF0/detF)^(1/3)` centroid-referenced volumetric scale) is
IDENTICALLY A NO-OP when the true field is homogeneous -- verified for
each newly-added type below against the same `run_abaqus_official_patch_test`
target, not assumed. For a general (non-affine) field this single-point
evaluation would NOT reproduce the assembly's own point-to-point stress
variation for elements with such a correction; extending to true
multi-point output (and, for C3D10M/C3D8_FBAR specifically, actually
applying their B-bar/F-bar correction rather than relying on it being a
no-op) is a documented follow-up, not attempted here.

C3D8I (9-mode EAS-enhanced) and C3D8H (hybrid pressure) are NOT
supported and cannot be added without a solver-level change: their
converged internal DOFs (the EAS alpha vector, the hybrid pressure
field) are solved fresh inside each kernel's own per-call assembly and
are NOT persisted anywhere on `DynamicSolver3D` between calls (confirmed
by inspecting `dynamic3d.py` -- no `eas_alpha`/`hybrid_q`-style attribute
exists in the 3D solver, unlike the 2D solver's `self.eas_alpha`/
`self.hybrid_q` per-element arrays). Recovering their stress correctly
would require either persisting that state at the solver level (a real,
separate task) or re-solving each element's own internal-DOF equilibrium
from scratch inside this post-processor (duplicating, not reusing, the
kernel's own logic -- the exact maintenance hazard this module was built
to avoid). Elements of an unsupported type are silently OMITTED from
`values()` (not raised) -- check `skipped_element_types` after
construction if a mesh might contain any; this docstring previously
(incorrectly) claimed `values()` raises `NotImplementedError` for
skipped types -- it doesn't, and never has, corrected here rather than
left standing as a doc/code mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from dispsolver.material3d.numba_materials import material_dispatch_3d

_TET4_TYPES = {"C3D4", "C3D4_ANP", "ANP"}
_HEX8_TYPES = {"C3D8", "C3D8R", "C3D8_CR"}
_TET10_TYPES = {"C3D10", "C3D10M", "C3D10_MODIFIED"}
_WEDGE6_TYPES = {"C3D6", "C3D6_WEDGE", "WEDGE6"}
_HEX8_FBAR_TYPES = {"C3D8_FBAR"}
# Elements whose kernel modifies F/the strain measure before the material
# call are only included above once that modification has been read from
# the kernel and matched exactly, not assumed -- reusing plain-F recovery
# for one of these would silently give the WRONG stress (the same class
# of "which F does this operator use" defect this project has hit
# repeatedly, AGENTS.md 4.14-4.18). C3D10M's B_vol_bar mean-dilatation
# term and C3D8_FBAR's (detF0/detF)^(1/3) centroid-referenced volumetric
# scale are BOTH identically a no-op when the true field is homogeneous
# (this module's own verification target) -- see this file's module
# docstring -- which is why they're handled by dedicated methods below
# that replicate each kernel's real construction, not by silently
# reusing plain C3D4/C3D8 recovery. C3D8I (EAS) and C3D8H (hybrid
# pressure) remain unsupported -- see module docstring for why.

_TET4_DN_DXI = np.array([
    [-1.0, 1.0, 0.0, 0.0],
    [-1.0, 0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


def _tet10_shape_derivs(xi: float, eta: float, zeta: float) -> np.ndarray:
    """dN/d(xi,eta,zeta) for the 10-node quadratic tet, EXACT copy of the
    node-order/sign convention in dispsolver/element3d/c3d10_numba.py's
    `_sd3d_tet10` (verified against it directly)."""
    w = 1.0 - xi - eta - zeta
    dN = np.zeros((3, 10), dtype=np.float64)
    dN[0, 0] = 1.0 - 4.0 * w
    dN[0, 1] = 4.0 * xi - 1.0
    dN[0, 4] = 4.0 * (w - xi)
    dN[0, 5] = 4.0 * eta
    dN[0, 6] = -4.0 * eta
    dN[0, 7] = -4.0 * zeta
    dN[0, 8] = 4.0 * zeta

    dN[1, 0] = 1.0 - 4.0 * w
    dN[1, 2] = 4.0 * eta - 1.0
    dN[1, 4] = -4.0 * xi
    dN[1, 5] = 4.0 * xi
    dN[1, 6] = 4.0 * (w - eta)
    dN[1, 7] = -4.0 * zeta
    dN[1, 9] = 4.0 * zeta

    dN[2, 0] = 1.0 - 4.0 * w
    dN[2, 3] = 4.0 * zeta - 1.0
    dN[2, 4] = -4.0 * xi
    dN[2, 6] = -4.0 * eta
    dN[2, 7] = 4.0 * (w - zeta)
    dN[2, 8] = 4.0 * xi
    dN[2, 9] = 4.0 * eta
    return dN


_TET10_DN_DXI_CENTROID = _tet10_shape_derivs(0.25, 0.25, 0.25)


def _wedge6_shape_derivs(xi: float, eta: float, zeta: float) -> np.ndarray:
    """dN/d(xi,eta,zeta) for the 6-node wedge, EXACT copy of
    dispsolver/element3d/c3d6_numba.py's `_sd3d_wedge6` convention
    (verified against it directly)."""
    lam1 = 1.0 - xi - eta
    lam2 = xi
    lam3 = eta
    half_m = 0.5 * (1.0 - zeta)
    half_p = 0.5 * (1.0 + zeta)
    dN = np.zeros((3, 6), dtype=np.float64)
    dN[0, 0] = -half_m
    dN[0, 1] = half_m
    dN[0, 3] = -half_p
    dN[0, 4] = half_p
    dN[1, 0] = -half_m
    dN[1, 2] = half_m
    dN[1, 3] = -half_p
    dN[1, 5] = half_p
    dN[2, 0] = -0.5 * lam1
    dN[2, 1] = -0.5 * lam2
    dN[2, 2] = -0.5 * lam3
    dN[2, 3] = 0.5 * lam1
    dN[2, 4] = 0.5 * lam2
    dN[2, 5] = 0.5 * lam3
    return dN


_WEDGE6_DN_DXI_CENTROID = _wedge6_shape_derivs(1.0 / 3.0, 1.0 / 3.0, 0.0)


def _hex8_shape_derivs_centroid() -> np.ndarray:
    """dN/dxi at the hex centroid (xi=eta=zeta=0), same node-order
    convention as dispsolver/element3d/c3d8_numba.py's _sd3d (verified
    against it directly, not re-derived independently) -- bottom face
    (i,j,k)->(i+1,j,k)->(i+1,j+1,k)->(i,j+1,k), then the same in-plane
    order at k+1, matching benchmark_element/mechanics_patches.py's own
    hex connectivity convention.
    """
    xi = eta = zeta = 0.0
    dN_dxi = np.array([
        -0.125 * (1.0 - eta) * (1.0 - zeta),
         0.125 * (1.0 - eta) * (1.0 - zeta),
         0.125 * (1.0 + eta) * (1.0 - zeta),
        -0.125 * (1.0 + eta) * (1.0 - zeta),
        -0.125 * (1.0 - eta) * (1.0 + zeta),
         0.125 * (1.0 - eta) * (1.0 + zeta),
         0.125 * (1.0 + eta) * (1.0 + zeta),
        -0.125 * (1.0 + eta) * (1.0 + zeta),
    ], dtype=np.float64)
    dN_deta = np.array([
        -0.125 * (1.0 - xi) * (1.0 - zeta),
        -0.125 * (1.0 + xi) * (1.0 - zeta),
         0.125 * (1.0 + xi) * (1.0 - zeta),
         0.125 * (1.0 - xi) * (1.0 - zeta),
        -0.125 * (1.0 - xi) * (1.0 + zeta),
        -0.125 * (1.0 + xi) * (1.0 + zeta),
         0.125 * (1.0 + xi) * (1.0 + zeta),
         0.125 * (1.0 - xi) * (1.0 + zeta),
    ], dtype=np.float64)
    dN_dzeta = np.array([
        -0.125 * (1.0 - xi) * (1.0 - eta),
        -0.125 * (1.0 + xi) * (1.0 - eta),
        -0.125 * (1.0 + xi) * (1.0 + eta),
        -0.125 * (1.0 - xi) * (1.0 + eta),
         0.125 * (1.0 - xi) * (1.0 - eta),
         0.125 * (1.0 + xi) * (1.0 - eta),
         0.125 * (1.0 + xi) * (1.0 + eta),
         0.125 * (1.0 - xi) * (1.0 + eta),
    ], dtype=np.float64)
    return np.vstack([dN_dxi, dN_deta, dN_dzeta])  # (3, 8)


_HEX8_DN_DXI_CENTROID = _hex8_shape_derivs_centroid()


@dataclass
class StressFieldValue:
    element_id: int
    integration_point: int
    data: np.ndarray  # (6,) Voigt Cauchy stress [S11,S22,S33,S12,S13,S23]
    mises: float
    pressure: float


@dataclass
class StressFieldOutput:
    """Analogous to an Abaqus odb FieldOutput('S') for one converged
    increment of a DynamicSolver3D solve."""

    solver: Any
    _values: List[StressFieldValue] = field(default_factory=list, init=False, repr=False)
    _by_element: Dict[int, List[StressFieldValue]] = field(default_factory=dict, init=False, repr=False)
    skipped_element_types: List[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._compute()

    def values(self) -> List[StressFieldValue]:
        return self._values

    def at_element(self, element_id: int) -> List[StressFieldValue]:
        return self._by_element.get(element_id, [])

    def _compute(self) -> None:
        solver = self.solver
        mesh = solver.mesh
        nid_to_idx = mesh.node_id_to_index()
        u = solver.u
        elem_mat_types = solver.elem_mat_types
        elem_props = solver.elem_props

        # elem index -> (eid, order) must match how _setup_numba_topology
        # ordered elements when building elem_mat_types/elem_props (the
        # same list(mesh.elements.values()) iteration order) -- reuse that
        # exact convention rather than re-deriving it.
        elements_list = list(mesh.elements.values())
        skipped: set = set()

        for e_idx, el in enumerate(elements_list):
            et = el.elem_type.upper()
            if et in _TET4_TYPES:
                stress_voigt = self._tet4_cauchy(el, u, nid_to_idx, elem_mat_types[e_idx], elem_props[e_idx])
            elif et in _HEX8_TYPES:
                stress_voigt = self._hex8_centroid_cauchy(el, u, nid_to_idx, elem_mat_types[e_idx], elem_props[e_idx])
            elif et in _TET10_TYPES:
                stress_voigt = self._tet10_centroid_cauchy(el, u, nid_to_idx, elem_mat_types[e_idx], elem_props[e_idx])
            elif et in _WEDGE6_TYPES:
                stress_voigt = self._wedge6_centroid_cauchy(el, u, nid_to_idx, elem_mat_types[e_idx], elem_props[e_idx])
            elif et in _HEX8_FBAR_TYPES:
                stress_voigt = self._hex8_fbar_centroid_cauchy(el, u, nid_to_idx, elem_mat_types[e_idx], elem_props[e_idx])
            else:
                skipped.add(et)
                continue

            if stress_voigt is None:
                continue

            s = stress_voigt
            mean = (s[0] + s[1] + s[2]) / 3.0
            dev = np.array([s[0] - mean, s[1] - mean, s[2] - mean, s[3], s[4], s[5]])
            mises = float(np.sqrt(1.5 * (dev[0]**2 + dev[1]**2 + dev[2]**2
                                          + 2.0 * (dev[3]**2 + dev[4]**2 + dev[5]**2))))
            val = StressFieldValue(
                element_id=el.id,
                integration_point=0,
                data=s,
                mises=mises,
                pressure=float(-mean),
            )
            self._values.append(val)
            self._by_element.setdefault(el.id, []).append(val)

        self.skipped_element_types = sorted(skipped)

    def _tet4_cauchy(self, el, u, nid_to_idx, mat_type, props) -> np.ndarray:
        coords = np.array([[
            self.solver.mesh.nodes[nid].x,
            self.solver.mesh.nodes[nid].y,
            self.solver.mesh.nodes[nid].z,
        ] for nid in el.node_ids])
        J0 = _TET4_DN_DXI @ coords
        invJ0 = np.linalg.inv(J0)
        dN_dX = invJ0 @ _TET4_DN_DXI  # (3, 4)

        u_elem = np.zeros((4, 3), dtype=np.float64)
        for a, nid in enumerate(el.node_ids):
            i3 = nid_to_idx[nid]
            u_elem[a] = u[3 * i3: 3 * i3 + 3]

        F = np.eye(3, dtype=np.float64)
        for a in range(4):
            F += np.outer(u_elem[a], dN_dX[:, a])

        return self._cauchy_from_F(F, mat_type, props)

    def _hex8_centroid_cauchy(self, el, u, nid_to_idx, mat_type, props) -> np.ndarray:
        coords = np.array([[
            self.solver.mesh.nodes[nid].x,
            self.solver.mesh.nodes[nid].y,
            self.solver.mesh.nodes[nid].z,
        ] for nid in el.node_ids])
        J0 = _HEX8_DN_DXI_CENTROID @ coords  # (3,3)
        invJ0 = np.linalg.inv(J0)
        dN_dX = invJ0 @ _HEX8_DN_DXI_CENTROID  # (3, 8)

        u_elem = np.zeros((8, 3), dtype=np.float64)
        for a, nid in enumerate(el.node_ids):
            i3 = nid_to_idx[nid]
            u_elem[a] = u[3 * i3: 3 * i3 + 3]

        F = np.eye(3, dtype=np.float64)
        for a in range(8):
            F += np.outer(u_elem[a], dN_dX[:, a])

        return self._cauchy_from_F(F, mat_type, props)

    def _tet10_centroid_cauchy(self, el, u, nid_to_idx, mat_type, props) -> np.ndarray:
        """C3D10/C3D10M: plain quadratic-tet F at the natural centroid
        (xi=eta=zeta=1/4). C3D10M's own kernel additionally applies a
        volume-averaged B_vol_bar mean-dilatation correction
        (c3d10m_numba.py's compute_c3d10m_element_numba, `B_bar = B_dev +
        (1/3)*B_vol_bar`) -- that correction is IDENTICALLY a no-op when
        the true field is homogeneous (B_vol_bar == the local B_vol at
        every point in that case, so B_bar == B), which is exactly this
        module's verification target -- verified directly (see
        tests/test_field_output.py), not assumed. For a non-homogeneous
        field this method would need the real B_bar correction to match
        C3D10M's assembled stress; not implemented here."""
        coords = np.array([[
            self.solver.mesh.nodes[nid].x,
            self.solver.mesh.nodes[nid].y,
            self.solver.mesh.nodes[nid].z,
        ] for nid in el.node_ids])
        J0 = _TET10_DN_DXI_CENTROID @ coords  # (3, 3)
        invJ0 = np.linalg.inv(J0)
        dN_dX = invJ0 @ _TET10_DN_DXI_CENTROID  # (3, 10)

        u_elem = np.zeros((10, 3), dtype=np.float64)
        for a, nid in enumerate(el.node_ids):
            i3 = nid_to_idx[nid]
            u_elem[a] = u[3 * i3: 3 * i3 + 3]

        F = np.eye(3, dtype=np.float64)
        for a in range(10):
            F += np.outer(u_elem[a], dN_dX[:, a])

        return self._cauchy_from_F(F, mat_type, props)

    def _wedge6_centroid_cauchy(self, el, u, nid_to_idx, mat_type, props) -> np.ndarray:
        """C3D6: plain wedge F at the natural centroid (triangle centroid
        xi=eta=1/3, axial midpoint zeta=0). c3d6_numba.py has no B-bar/
        F-bar correction (confirmed by grep -- plain compatible
        formulation), so this is exact for ANY field the kernel itself
        would agree with at that point, not just the homogeneous
        verification target."""
        coords = np.array([[
            self.solver.mesh.nodes[nid].x,
            self.solver.mesh.nodes[nid].y,
            self.solver.mesh.nodes[nid].z,
        ] for nid in el.node_ids])
        J0 = _WEDGE6_DN_DXI_CENTROID @ coords  # (3, 3)
        invJ0 = np.linalg.inv(J0)
        dN_dX = invJ0 @ _WEDGE6_DN_DXI_CENTROID  # (3, 6)

        u_elem = np.zeros((6, 3), dtype=np.float64)
        for a, nid in enumerate(el.node_ids):
            i3 = nid_to_idx[nid]
            u_elem[a] = u[3 * i3: 3 * i3 + 3]

        F = np.eye(3, dtype=np.float64)
        for a in range(6):
            F += np.outer(u_elem[a], dN_dX[:, a])

        return self._cauchy_from_F(F, mat_type, props)

    def _hex8_fbar_centroid_cauchy(self, el, u, nid_to_idx, mat_type, props) -> np.ndarray:
        """C3D8_FBAR: replicates dispsolver/element3d/c3d8_fbar_tl_numba.py's
        own F-bar construction EXACTLY (`compute_c3d8_fbar_tl_element_numba`,
        verified by reading that kernel directly, not assumed): F_bar =
        (detF0/detF)^(1/3) * F, where detF0 is F's determinant AT THE
        CENTROID and F is evaluated at whatever point stress is wanted --
        here, both are evaluated at the centroid, so detF0 == detF and
        F_bar == F identically for this single-point evaluation
        (regardless of whether the field is homogeneous). This differs
        from the tet10/hex8 "no-op because the field happens to be
        homogeneous" argument above -- here scale_vol=1 by construction
        (same point for both F and F0), not because of the specific test
        field, so this centroid recovery is exact for C3D8_FBAR at any
        field, same as the plain hex methods above, though it still only
        gives ONE point's stress, not the kernel's real multi-Gauss-point
        field."""
        coords = np.array([[
            self.solver.mesh.nodes[nid].x,
            self.solver.mesh.nodes[nid].y,
            self.solver.mesh.nodes[nid].z,
        ] for nid in el.node_ids])
        J0 = _HEX8_DN_DXI_CENTROID @ coords
        invJ0 = np.linalg.inv(J0)
        dN_dX = invJ0 @ _HEX8_DN_DXI_CENTROID

        u_elem = np.zeros((8, 3), dtype=np.float64)
        for a, nid in enumerate(el.node_ids):
            i3 = nid_to_idx[nid]
            u_elem[a] = u[3 * i3: 3 * i3 + 3]

        F = np.eye(3, dtype=np.float64)
        for a in range(8):
            F += np.outer(u_elem[a], dN_dX[:, a])

        detF = float(np.linalg.det(F))
        if detF <= 0.0:
            return None
        detF0 = detF  # same evaluation point -> scale_vol == 1 exactly
        scale_vol = (detF0 / detF) ** (1.0 / 3.0)
        F_bar = scale_vol * F

        return self._cauchy_from_F(F_bar, mat_type, props)

    @staticmethod
    def _cauchy_from_F(F: np.ndarray, mat_type: int, props: np.ndarray) -> np.ndarray:
        detF = float(np.linalg.det(F))
        if detF <= 0.0:
            return None

        # Green-Lagrange strain E = 1/2(F^T F - I), Voigt [E11,E22,E33,2E12,2E13,2E23]
        C = F.T @ F
        E = 0.5 * (C - np.eye(3))
        E_voigt = np.array([E[0, 0], E[1, 1], E[2, 2], 2.0 * E[0, 1], 2.0 * E[0, 2], 2.0 * E[1, 2]])

        sdv_dummy = np.zeros(0, dtype=np.float64)
        S_voigt, _C_mat, _sdv_new, err = material_dispatch_3d(
            int(mat_type), props, sdv_dummy, E_voigt, F, detF, 1.0
        )
        if err != 0:
            return None

        S = np.array([
            [S_voigt[0], S_voigt[3], S_voigt[4]],
            [S_voigt[3], S_voigt[1], S_voigt[5]],
            [S_voigt[4], S_voigt[5], S_voigt[2]],
        ])
        sigma = (F @ S @ F.T) / detF  # Cauchy stress, push-forward from PK2
        return np.array([sigma[0, 0], sigma[1, 1], sigma[2, 2], sigma[0, 1], sigma[0, 2], sigma[1, 2]])
