"""
nitsche_contact3d.py
====================
3D Multiplier-Free Symmetric Nitsche Contact Discretization.
Commercial-grade implementation based on Chouly, Fabre, Hild & Pousin (2017)
and Wriggers & Zavarise (2008).

Key Features:
1. Multiplier-Free (Zero Additional DOFs):
   Total system size remains exactly 3 * N_nodes, avoiding indefinite saddle-point systems.
2. Exact Symmetry & Positive Semi-Definiteness (SPD Preservation):
   Tangent stiffness matrix is assembled from point-wise dyadic outer products:
     K_Nitsche = sum_{qp} gamma * w_qp * (c_qp (x) c_qp^T)
   with upper-triangular accumulation ensuring |K - K^T| = 0.0 bit-level symmetry.
   Directly compatible with high-speed PARDISO Cholesky (LLT) factorizations.
3. Accurate Contact Enforcement without Penalty Drift:
   Consistent boundary weighting converges to exact non-penetration without requiring
   artificially extreme penalty values that cause ill-conditioning.
4. Seamless DynamicSolver3D Integration:
   Implements assemble(u), get_active_set(u), and update_augmented_multipliers(u, omega).
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Set, Sequence
import numpy as np

from dispsolver.constraint3d.surface_tie3d import _project_point_to_quad4


def _quad4_shape_functions(xi: float, eta: float) -> Tuple[float, float, float, float]:
    """Evaluate standard bilinear Quad4 shape functions on [-1, 1]^2."""
    N1 = 0.25 * (1.0 - xi) * (1.0 - eta)
    N2 = 0.25 * (1.0 + xi) * (1.0 - eta)
    N3 = 0.25 * (1.0 + xi) * (1.0 + eta)
    N4 = 0.25 * (1.0 - xi) * (1.0 + eta)
    return N1, N2, N3, N4


def _quad4_tangents(q1: np.ndarray, q2: np.ndarray, q3: np.ndarray, q4: np.ndarray, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute dx/dxi and dx/deta at (xi, eta)."""
    dN1_dxi = -0.25 * (1.0 - eta)
    dN2_dxi =  0.25 * (1.0 - eta)
    dN3_dxi =  0.25 * (1.0 + eta)
    dN4_dxi = -0.25 * (1.0 + eta)

    dN1_deta = -0.25 * (1.0 - xi)
    dN2_deta = -0.25 * (1.0 + xi)
    dN3_deta =  0.25 * (1.0 + xi)
    dN4_deta =  0.25 * (1.0 - xi)

    t_xi = dN1_dxi * q1 + dN2_dxi * q2 + dN3_dxi * q3 + dN4_dxi * q4
    t_eta = dN1_deta * q1 + dN2_deta * q2 + dN3_deta * q3 + dN4_deta * q4
    return t_xi, t_eta


def _quad4_normal(q1: np.ndarray, q2: np.ndarray, q3: np.ndarray, q4: np.ndarray, xi: float, eta: float) -> np.ndarray:
    """Compute outward unit normal vector at (xi, eta)."""
    t_xi, t_eta = _quad4_tangents(q1, q2, q3, q4, xi, eta)
    n = np.cross(t_xi, t_eta)
    norm = float(np.linalg.norm(n))
    if norm < 1e-14:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return n / norm


def _quad4_area_jacobian(q1: np.ndarray, q2: np.ndarray, q3: np.ndarray, q4: np.ndarray, xi: float, eta: float) -> float:
    """Compute physical differential area dA = ||dx/dxi x dx/deta||."""
    t_xi, t_eta = _quad4_tangents(q1, q2, q3, q4, xi, eta)
    return float(np.linalg.norm(np.cross(t_xi, t_eta)))


# 2x2 Gauss-Legendre quadrature on [-1, 1]^2
_HEX8_NAT = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.float64)


def _hex8_shape(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Trilinear shape functions N (8,) and dN/dxi (8, 3) at natural point xi."""
    t = 1.0 + _HEX8_NAT * xi
    N = 0.125 * t[:, 0] * t[:, 1] * t[:, 2]
    dN = 0.125 * np.stack([_HEX8_NAT[:, 0] * t[:, 1] * t[:, 2],
                           t[:, 0] * _HEX8_NAT[:, 1] * t[:, 2],
                           t[:, 0] * t[:, 1] * _HEX8_NAT[:, 2]], axis=1)
    return N, dN


def _b_std(dNdX: np.ndarray) -> np.ndarray:
    """Engineering-strain B (6 x 24) in c3d8_corotational_numba's Voigt order
    [xx, yy, zz, xy, yz, xz]."""
    B = np.zeros((6, 24))
    for i in range(8):
        dx, dy, dz = dNdX[i]
        B[0, 3 * i] = dx
        B[1, 3 * i + 1] = dy
        B[2, 3 * i + 2] = dz
        B[3, 3 * i] = dy
        B[3, 3 * i + 1] = dx
        B[4, 3 * i + 1] = dz
        B[4, 3 * i + 2] = dy
        B[5, 3 * i] = dz
        B[5, 3 * i + 2] = dx
    return B


def _bbar_vol_row(dNdX: np.ndarray) -> np.ndarray:
    B = _b_std(dNdX)
    return (B[0] + B[1] + B[2]) / 3.0


def _hex8_inverse_map(X_target: np.ndarray, X_e: np.ndarray) -> np.ndarray:
    """Natural coordinates of a reference point inside / on a hex8."""
    xi = np.zeros(3)
    for _ in range(30):
        N, dN = _hex8_shape(xi)
        r = X_target - N @ X_e
        if np.linalg.norm(r) < 1e-13 * max(1.0, np.abs(X_e).max()):
            break
        xi = xi + np.linalg.solve(X_e.T @ dN, r)
    return xi


_INV_SQRT3 = 1.0 / np.sqrt(3.0)
_GAUSS2X2 = [
    (-_INV_SQRT3, -_INV_SQRT3, 1.0),
    (+_INV_SQRT3, -_INV_SQRT3, 1.0),
    (+_INV_SQRT3, +_INV_SQRT3, 1.0),
    (-_INV_SQRT3, +_INV_SQRT3, 1.0),
]


class NitscheContactConstraint3D:
    """3D Multiplier-Free Symmetric Nitsche Surface-to-Surface Contact Constraint.

    Parameters
    ----------
    slave_faces : List[Tuple[int, int, int, int]]
        List of 4-node slave element face node IDs.
    master_faces : List[Tuple[int, int, int, int]]
        List of 4-node master element face node IDs.
    nid_to_idx : Dict[int, int]
        Mapping from node ID to contiguous 0-based node index.
    coords : np.ndarray
        Array of reference nodal coordinates (N, 3).
    gamma : float
        Nitsche stabilization parameter [force / length^3]. Defaults to 1.0e8.
    position_tolerance : float
        Maximum separation distance between face centroids for pairing.
    name : str
        Constraint identifier.
    """

    def __init__(
        self,
        slave_faces: Sequence[Tuple[int, int, int, int]],
        master_faces: Sequence[Tuple[int, int, int, int]],
        nid_to_idx: Dict[int, int],
        coords: np.ndarray,
        gamma: float = 1.0e8,
        position_tolerance: float = 1.0,
        name: str = "NITSCHE_CONTACT_3D",
        master_interior: Optional[Dict[tuple, np.ndarray]] = None,
        slave_owner: Optional[Dict[tuple, int]] = None,
        gamma0: Optional[float] = None,
        consistency: bool = True,
    ):
        self.master_interior = dict(master_interior or {})
        # Element owning each slave face: the bulk stress sigma_nn that makes
        # this Nitsche (not penalty) is evaluated in it. Bound to the solver's
        # materials by bind_solver(), called from DynamicSolver3D.assemble_system.
        self.slave_owner = dict(slave_owner or {})
        # Nitsche stabilization: gamma = gamma0 * M / h per slave element
        # (M = initial P-wave modulus of its material, h = volume / face area).
        # theta = 0 Nitsche is only stable for gamma0 large enough -- measured on
        # the contact patch test: gamma0 = 2.5 diverged, >= 25 converged. When
        # gamma0 is None the fixed `gamma` (penalty-style) is used as before.
        self.gamma0 = None if gamma0 is None else float(gamma0)
        # consistency=False drops the bulk-stress term: p = gamma * g_n, a
        # surface-to-surface PENALTY (what this class was before 2026-10-04,
        # while its docstring called it Nitsche). Opt-in only; it needs no
        # solver binding and is symmetric, but its error scales as 1/gamma.
        self.consistency = bool(consistency)
        self._solver = None
        self._dt = 1.0
        self.slave_faces = list(slave_faces)
        self.master_faces = list(master_faces)
        self.nid_to_idx = dict(nid_to_idx)
        self.coords = np.asarray(coords, dtype=np.float64)
        self.gamma = float(gamma)
        self.position_tolerance = float(position_tolerance)
        self.name = str(name)
        self.augmented_lagrange = False  # Nitsche is multiplier-free

        # Build contact face pairs with frozen reference projection (small-sliding)
        self.pairs: List[Dict[str, Any]] = []
        self._build_pairs()

    def _build_pairs(self) -> None:
        """Pair slave faces with closest master faces and precompute frozen quadrature projections."""
        self.pairs = []
        for s_idx, face in enumerate(self.slave_faces):
            idx = [self.nid_to_idx[nid] for nid in face]
            q_s = [self.coords[i] for i in idx]
            centroid_s = 0.25 * (q_s[0] + q_s[1] + q_s[2] + q_s[3])

            best_m = None
            min_dist = float("inf")
            for m_face in self.master_faces:
                m_idx = [self.nid_to_idx[nid] for nid in m_face]
                q_m = [self.coords[i] for i in m_idx]
                xi_c, eta_c, dist, _ = _project_point_to_quad4(centroid_s, *q_m)
                if dist < min_dist:
                    min_dist = dist
                    best_m = (m_face, q_m, xi_c, eta_c)

            if best_m is None or min_dist > self.position_tolerance:
                continue

            m_face, q_m, xi_c, eta_c = best_m

            # Sign orientation from the CENTROID gap, matching sibling contact convention
            n_c_raw = _quad4_normal(q_m[0], q_m[1], q_m[2], q_m[3], xi_c, eta_c)
            Nm_c = _quad4_shape_functions(xi_c, eta_c)
            xm_c = sum(Nm_c[k] * q_m[k] for k in range(4))
            interior = self.master_interior.get(tuple(m_face))
            if interior is not None:
                # Outward from the master body, whatever the initial gap.
                sign = 1.0 if np.dot(n_c_raw, xm_c - interior) >= 0.0 else -1.0
            else:
                gap0 = float(np.dot(centroid_s - xm_c, n_c_raw))
                sign = 1.0 if (abs(gap0) < 1e-5 or gap0 >= 0.0) else -1.0

            # Quadrature points on slave face
            quad_points = []
            for xi_s, eta_s, w_gauss in _GAUSS2X2:
                Ns = _quad4_shape_functions(xi_s, eta_s)
                x_qp = Ns[0] * q_s[0] + Ns[1] * q_s[1] + Ns[2] * q_s[2] + Ns[3] * q_s[3]
                xi_m, eta_m, _, _ = _project_point_to_quad4(x_qp, *q_m)
                dA = _quad4_area_jacobian(q_s[0], q_s[1], q_s[2], q_s[3], xi_s, eta_s) * w_gauss

                n_m = _quad4_normal(q_m[0], q_m[1], q_m[2], q_m[3], xi_m, eta_m)
                n_contact = sign * n_m

                quad_points.append({
                    "xi_s": float(xi_s),
                    "eta_s": float(eta_s),
                    "xi_m": float(xi_m),
                    "eta_m": float(eta_m),
                    "w": float(dA),
                    "n": n_contact.copy(),
                    "sign": float(sign),
                })

            self.pairs.append({
                "slave_face_idx": s_idx,
                "slave_face": face,
                "master_face": m_face,
                "quad_points": quad_points,
            })

    def get_active_set(self, u: np.ndarray) -> frozenset:
        """Return frozenset of slave node IDs that participate in active penetrating contact."""
        active_nodes: Set[int] = set()
        for pair in self.pairs:
            s_face = pair["slave_face"]
            m_face = pair["master_face"]
            idx_s = [self.nid_to_idx[nid] for nid in s_face]
            idx_m = [self.nid_to_idx[nid] for nid in m_face]

            for qp in pair["quad_points"]:
                Ns = _quad4_shape_functions(qp["xi_s"], qp["eta_s"])
                Nm = _quad4_shape_functions(qp["xi_m"], qp["eta_m"])

                xs = sum(Ns[a] * (self.coords[idx_s[a]] + u[3 * idx_s[a] : 3 * idx_s[a] + 3]) for a in range(4))
                xm = sum(Nm[b] * (self.coords[idx_m[b]] + u[3 * idx_m[b] : 3 * idx_m[b] + 3]) for b in range(4))

                n = self._current_normal(u, idx_m, qp)
                g_n = -float(np.dot(xs - xm, n))
                # Same activity rule as assemble(): p = gamma*g_n - sigma_nn > 0.
                sig = (self._sigma_nn_cr(u[qp["e_dofs"]].reshape(8, 3), qp, n)
                       if (self.consistency and self._solver is not None) else 0.0)
                gam = qp.get("gamma", self.gamma)
                if gam * g_n - (sig if sig is not None else 0.0) > 0.0:
                    active_nodes.update(s_face)
                    break

        return frozenset(active_nodes)

    # ------------------------------------------------------------------
    # Bulk normal stress on the slave side (the Nitsche consistency term)
    # ------------------------------------------------------------------
    def bind_solver(self, solver: Any, dt: float = 1.0) -> None:
        """Attach the solver whose element materials give sigma(u) on the
        slave side, and precompute, per quadrature point, the owning element
        and its shape-function gradients there."""
        self._dt = float(dt)
        if not self.consistency or self._solver is solver:
            return
        self._solver = solver
        eid_to_row = {eid: k for k, eid in enumerate(solver.mesh.elements)}
        if not self.slave_owner:
            from dispsolver.model.interaction import _face_owner_elements
            self.slave_owner = _face_owner_elements([tuple(p["slave_face"]) for p in self.pairs], solver.mesh)
        for pair in self.pairs:
            face = tuple(pair["slave_face"])
            eid = self.slave_owner.get(face)
            if eid is None:
                raise ValueError(f"Nitsche contact '{self.name}': no owning element for slave face {face}")
            elem = solver.mesh.elements[eid]
            nids = list(elem.node_ids)
            if len(nids) != 8:
                raise NotImplementedError(
                    f"Nitsche contact '{self.name}': slave element {eid} has {len(nids)} nodes; "
                    f"the bulk-stress term is implemented for 8-node hexahedra only")
            e_idx = [self.nid_to_idx[n] for n in nids]
            X_e = self.coords[e_idx]
            idx_s = [self.nid_to_idx[n] for n in face]
            kern = getattr(solver, "_row_kernel", {}).get(eid_to_row[eid])
            if kern not in getattr(solver, "_POINT_STRESS_KERNELS", {}):
                raise NotImplementedError(
                    f"Nitsche contact '{self.name}': slave element {eid} runs solver kernel {kern}; "
                    f"the solver cannot evaluate that kernel's stress at a contact point "
                    f"(DynamicSolver3D.element_point_stress), and any other stress definition "
                    f"would make the method inconsistent.")
            # Volume-averaged shape-function gradients (reference), exactly as
            # c3d8_corotational_numba builds B-bar.
            g = 1.0 / np.sqrt(3.0)
            dN_mean = np.zeros((8, 3))
            V0 = 0.0
            for gx in (-g, g):
                for gy in (-g, g):
                    for gz in (-g, g):
                        _, dNg = _hex8_shape(np.array([gx, gy, gz]))
                        Jg = X_e.T @ dNg
                        detg = float(np.linalg.det(Jg))
                        dN_mean += (dNg @ np.linalg.inv(Jg)) * detg
                        V0 += detg
            vol_mean = _bbar_vol_row(dN_mean / V0)
            h_fd = 1e-7 * float(np.max(np.ptp(X_e, axis=0)))
            gamma_e = self.gamma
            if self.gamma0 is not None:
                from dispsolver.material3d.numba_materials import material_dispatch_3d
                row = eid_to_row[eid]
                sdv0 = np.zeros(getattr(solver, "elem_sdvs", np.zeros((1, 0))).shape[-1] if getattr(solver, "elem_sdvs", None) is not None else 0)
                _, C0, _, _ = material_dispatch_3d(int(solver.elem_mat_types[row]), solver.elem_props[row],
                                                   sdv0, np.zeros(6), np.eye(3), 1.0, 1.0)
                q4 = [self.coords[i] for i in idx_s]
                area = sum(_quad4_area_jacobian(q4[0], q4[1], q4[2], q4[3], gx, gy) * gw for gx, gy, gw in _GAUSS2X2)
                _, dN0 = _hex8_shape(np.zeros(3))
                vol = 8.0 * abs(np.linalg.det(X_e.T @ dN0))
                gamma_e = self.gamma0 * float(C0[0, 0]) / (vol / area)
            for qp in pair["quad_points"]:
                Ns = _quad4_shape_functions(qp["xi_s"], qp["eta_s"])
                X_qp = sum(Ns[a] * self.coords[idx_s[a]] for a in range(4))
                xi_e = _hex8_inverse_map(X_qp, X_e)
                _, dN = _hex8_shape(xi_e)
                qp["e_row"] = eid_to_row[eid]
                qp["e_dofs"] = np.array([3 * i + d for i in e_idx for d in range(3)], dtype=np.int64)
                qp["dNdX"] = dN @ np.linalg.inv(X_e.T @ dN).T   # (8, 3): dN_a/dX
                qp["h_fd"] = h_fd
                qp["gamma"] = gamma_e
                qp["X_e"] = X_e
                qp["vol_mean"] = vol_mean

    def _sigma_nn_cr(self, u_e: np.ndarray, qp: Dict[str, Any], n: Optional[np.ndarray] = None) -> Optional[float]:
        """n . sigma . n at the quadrature point, sigma being the stress the
        slave element itself produces there: delegated to
        DynamicSolver3D.element_point_stress, which dispatches on the kernel the
        solver picked for that element (NLGEOM-dependent). An earlier version
        hard-coded Green-Lagrange/PK2 at the compatible F, i.e. a strain
        measure the element does not use."""
        sig = self._solver.element_point_stress(qp["e_row"], qp["X_e"], qp["dNdX"], qp["vol_mean"],
                                                u_e, self._dt)
        if sig is None:
            return None
        if n is None:
            n = qp["n"]
        return float(n @ sig @ n)

    def _sigma_nn(self, u_e: np.ndarray, qp: Dict[str, Any]) -> Optional[float]:
        return self._sigma_nn_cr(u_e, qp)

    def _sigma_nn_and_grad_u(self, u: np.ndarray, qp: Dict[str, Any], n: Optional[np.ndarray] = None):
        """sigma_nn and d sigma_nn / d u_e (24,) by central differences on the
        element DOFs (R depends on all of them, so no cheaper chain rule)."""
        u_e = u[qp["e_dofs"]].reshape(8, 3)
        sig = self._sigma_nn_cr(u_e, qp, n)
        if sig is None:
            return None, None
        flat = u_e.ravel()
        h = qp["h_fd"]
        grad = np.zeros(24)
        for j in range(24):
            up = flat.copy(); up[j] += h
            um = flat.copy(); um[j] -= h
            sp = self._sigma_nn_cr(up.reshape(8, 3), qp, n)
            sm = self._sigma_nn_cr(um.reshape(8, 3), qp, n)
            if sp is None or sm is None:
                return sig, None
            grad[j] = (sp - sm) / (2.0 * h)
        return sig, grad

    def _current_normal(self, u: np.ndarray, idx_m: List[int], qp: Dict[str, Any]) -> np.ndarray:
        """Master normal at the paired master point in the CURRENT geometry.
        It used to be frozen at build time (qp["n"]); when the master surface
        rotates (deck #06: the flange bends) while the slave slides along it
        (the gasket bulges ~0.3 mm), a frozen normal mis-measures the gap by
        about slip x rotation, and the contact force acts in a stale
        direction. The node-to-surface penalty and the mortar already use the
        current normal."""
        q = [self.coords[i] + u[3 * i: 3 * i + 3] for i in idx_m]
        return qp.get("sign", 1.0) * _quad4_normal(q[0], q[1], q[2], q[3], qp["xi_m"], qp["eta_m"])

    def _require_bound(self) -> None:
        if self._solver is None:
            raise RuntimeError(
                f"Nitsche contact '{self.name}' is not bound to a solver: its bulk-stress term "
                f"needs the slave elements' materials (DynamicSolver3D.assemble_system binds it)")

    def assemble(
        self, u: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, float]]:
        """Nitsche contact (theta = 0, Chouly et al. 2017) on the slave surface.

        Per quadrature point, with g_n the penetration and sigma_nn the slave
        element's normal Cauchy stress (negative in compression):

            p = [gamma * g_n - sigma_nn]_+        (contact pressure)
            f = p * w * c,   c = (-N_s n, +N_m n)

        At the exact solution g_n = 0 and p = -sigma_nn, so the method is
        consistent: unlike a penalty (the previous p = gamma * g_n, whose error
        scaled exactly as 1/gamma), a uniform-stress patch is reproduced for
        any stable gamma. Tangent: K = w c (x) (gamma c - d sigma_nn / du_e),
        UNSYMMETRIC through the stress term (the solver factors it with LU);
        d sigma_nn / du_e by central differences over the slave element's 24
        DOFs. The normal n is held at its build-time value, as before.
        """
        if self.consistency:
            self._require_bound()
        n_dof = len(u)
        f_contact = np.zeros(n_dof, dtype=np.float64)
        rows: List[np.ndarray] = []
        cols: List[np.ndarray] = []
        data: List[np.ndarray] = []
        max_penetration = 0.0
        n_active = 0
        total_normal_force = 0.0

        for pair in self.pairs:
            idx_s = [self.nid_to_idx[nid] for nid in pair["slave_face"]]
            idx_m = [self.nid_to_idx[nid] for nid in pair["master_face"]]
            for qp in pair["quad_points"]:
                Ns = _quad4_shape_functions(qp["xi_s"], qp["eta_s"])
                Nm = _quad4_shape_functions(qp["xi_m"], qp["eta_m"])
                xs = sum(Ns[a] * (self.coords[idx_s[a]] + u[3 * idx_s[a]: 3 * idx_s[a] + 3]) for a in range(4))
                xm = sum(Nm[b] * (self.coords[idx_m[b]] + u[3 * idx_m[b]: 3 * idx_m[b] + 3]) for b in range(4))
                n = self._current_normal(u, idx_m, qp)
                g_n = -float(np.dot(xs - xm, n))
                if self.consistency:
                    sig = self._sigma_nn_cr(u[qp["e_dofs"]].reshape(8, 3), qp, n)
                    if sig is None:
                        continue
                    gam = qp["gamma"]
                else:
                    sig, gam = 0.0, self.gamma
                p_n = gam * g_n - sig
                if p_n <= 0.0:
                    continue
                dsig = self._sigma_nn_and_grad_u(u, qp, n)[1] if self.consistency else None
                n_active += 1
                max_penetration = max(max_penetration, g_n)
                w = qp["w"]
                total_normal_force += p_n * w

                c_dofs = np.array([3 * i + d for i in idx_s + idx_m for d in range(3)], dtype=np.int64)
                c_vec = np.concatenate([-Ns[a] * n for a in range(4)] + [Nm[b] * n for b in range(4)])
                np.add.at(f_contact, c_dofs, w * p_n * c_vec)

                rows.append(np.repeat(c_dofs, len(c_dofs)))
                cols.append(np.tile(c_dofs, len(c_dofs)))
                data.append((w * gam * np.outer(c_vec, c_vec)).ravel())
                if dsig is not None:
                    e_dofs = qp["e_dofs"]
                    rows.append(np.repeat(c_dofs, len(e_dofs)))
                    cols.append(np.tile(e_dofs, len(c_dofs)))
                    data.append((-w * np.outer(c_vec, dsig)).ravel())
                if self.consistency:
                    # Rotation of the current master normal n(q_m): it enters the
                    # gap, sigma_nn = n.sigma.n and the force direction.
                    from dispsolver.constraint3d.surface_contact3d_deformable import _quad4_normal_derivatives
                    q_m = [self.coords[i] + u[3 * i: 3 * i + 3] for i in idx_m]
                    Dn = _quad4_normal_derivatives(q_m, qp["xi_m"], qp["eta_m"], n)
                    sig_t = self._solver.element_point_stress(
                        qp["e_row"], qp["X_e"], qp["dNdX"], qp["vol_mean"],
                        u[qp["e_dofs"]].reshape(8, 3), self._dt)
                    d_vec = xs - xm
                    sn2 = 2.0 * (sig_t @ n)
                    blk = np.zeros((24, 12))
                    for bb in range(4):
                        dp_b = -gam * (d_vec @ Dn[bb]) - sn2 @ Dn[bb]          # (3,)
                        geo = np.vstack([-Ns[a] * Dn[bb] for a in range(4)] + [Nm[b2] * Dn[bb] for b2 in range(4)])
                        blk[:, 3 * bb: 3 * bb + 3] = w * np.outer(c_vec, dp_b) + w * p_n * geo
                    m_dofs = np.array([3 * i + dd for i in idx_m for dd in range(3)], dtype=np.int64)
                    rows.append(np.repeat(c_dofs, 12))
                    cols.append(np.tile(m_dofs, len(c_dofs)))
                    data.append(blk.ravel())

        stats = {
            "max_penetration": max_penetration,
            "n_active": n_active,
            "n_pairs": len(self.pairs),
            "total_normal_force": total_normal_force,
        }
        if rows:
            return f_contact, (np.concatenate(rows).astype(np.int32), np.concatenate(cols).astype(np.int32),
                               np.concatenate(data)), stats
        return f_contact, (np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0)), stats

    def update_augmented_multipliers(self, u: np.ndarray, omega: float = 1.0) -> Dict[str, float]:
        """Nitsche method is multiplier-free; returns dummy stats for solver API compatibility."""
        return {"max_penetration": 0.0, "max_lambda_change": 0.0}
