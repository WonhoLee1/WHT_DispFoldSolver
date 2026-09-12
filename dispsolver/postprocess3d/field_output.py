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

Current scope (2026-09-13): C3D4/C3D4_ANP (4-node tet, single
integration point) and C3D8-family hexahedra (8-node, evaluated at the
element CENTROID only, single point -- not the full 2x2x2 Gauss rule the
assembly itself uses). The centroid-only hex evaluation is an
intentional, documented simplification: for the verification target this
module was built against (`run_abaqus_official_patch_test`'s affine
displacement field), the deformation gradient F is spatially CONSTANT
throughout every element, so a single evaluation point is exact -- not
an approximation -- for that case. For a general (non-affine) hex field,
a full 8-point Gauss stress field would differ point-to-point; extending
to real multi-point hex output is a documented follow-up, not attempted
here. C3D10/C3D10M (quadratic tet), C3D6 (wedge), and the hybrid/EAS/
F-bar-enhanced hex variants are NOT YET supported (`values()` raises
NotImplementedError naming which element types were skipped).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from dispsolver.material3d.numba_materials import material_dispatch_3d

_TET4_TYPES = {"C3D4", "C3D4_ANP", "ANP"}
_HEX8_TYPES = {"C3D8", "C3D8R", "C3D8_CR"}
# Elements whose kernel modifies F/the strain measure before the material
# call (F-bar, incompatible modes, hybrid pressure) are deliberately NOT
# included here yet -- reusing this module's plain-F recovery for them
# would silently give the WRONG stress (the same class of "which F does
# this operator use" defect this project has hit repeatedly, AGENTS.md
# 4.14-4.18). Add them only by reading and matching each kernel's own
# modified-F/modified-strain construction, not by assuming plain F works.

_TET4_DN_DXI = np.array([
    [-1.0, 1.0, 0.0, 0.0],
    [-1.0, 0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


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
