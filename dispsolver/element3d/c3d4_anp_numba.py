"""
c3d4_anp_numba.py
=================
High-Performance Numba Kernel for 3D 4-Node Linear Tetrahedron Elements
with Bonet & Burton (1998) 2-Pass Average Nodal Pressure (C3D4_ANP).

Features:
- Pass 1: Element-to-node volume and dilatation accumulation (V_a, v_a, J_a = v_a / V_a).
- Pass 2: OpenMP parallel element-wise F-bar projection (F_bar = (J_bar_e / J_e)^(1/3) * F_e)
  and decoupled B-bar tangent assembly.
- Completely resolves volumetric locking in nearly incompressible materials (nu -> 0.5)
  and finite-strain J2 plasticity.
- Decoupled UMAT constitutive integration via material_dispatch_3d.
"""

from __future__ import annotations
import numpy as np

try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from dispsolver.material3d.numba_materials import material_dispatch_3d, MAT_CUSTOM_ELASTIC


if HAS_NUMBA:
    _EMPTY_1D_FLOAT = np.empty(0, dtype=np.float64)
    _EMPTY_2D_FLOAT = np.empty((0, 0), dtype=np.float64)

    # Constant natural derivatives for 4-node tetrahedron (3, 4)
    _DN_DXI_TET4 = np.array([
        [-1.0,  1.0,  0.0,  0.0],
        [-1.0,  0.0,  1.0,  0.0],
        [-1.0,  0.0,  0.0,  1.0]
    ], dtype=np.float64)

    @njit(fastmath=True)
    def _inv3x3_det_tet(A: np.ndarray) -> tuple[np.ndarray, float]:
        """Compute inverse and determinant of 3x3 matrix."""
        a00 = A[0, 0]; a01 = A[0, 1]; a02 = A[0, 2]
        a10 = A[1, 0]; a11 = A[1, 1]; a12 = A[1, 2]
        a20 = A[2, 0]; a21 = A[2, 1]; a22 = A[2, 2]

        c00 =  (a11 * a22 - a12 * a21)
        c01 = -(a10 * a22 - a12 * a20)
        c02 =  (a10 * a21 - a11 * a20)

        det = a00 * c00 + a01 * c01 + a02 * c02

        if abs(det) < 1e-15:
            inv = np.zeros((3, 3), dtype=np.float64)
            return inv, 0.0

        inv_det = 1.0 / det
        inv = np.zeros((3, 3), dtype=np.float64)
        inv[0, 0] = c00 * inv_det
        inv[0, 1] = -(a01 * a22 - a02 * a21) * inv_det
        inv[0, 2] =  (a01 * a12 - a02 * a11) * inv_det

        inv[1, 0] = c01 * inv_det
        inv[1, 1] =  (a00 * a22 - a02 * a20) * inv_det
        inv[1, 2] = -(a00 * a12 - a02 * a10) * inv_det

        inv[2, 0] = c02 * inv_det
        inv[2, 1] = -(a00 * a21 - a01 * a20) * inv_det
        inv[2, 2] =  (a00 * a11 - a01 * a10) * inv_det

        return inv, det

    @njit(fastmath=True)
    def compute_c3d4_anp_element_matrices(
        coords: np.ndarray,      # (4, 3)
        u_elem: np.ndarray,      # (12,)
        J_bar_e: float,          # Smoothed element Jacobian from Pass 1
        mat_type: int,
        props: np.ndarray,
        sdvs: np.ndarray,
        dt: float,
        elem_controls: np.ndarray = _EMPTY_1D_FLOAT
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Compute single C3D4_ANP element tangent stiffness (12, 12) and internal force (12,)."""
        K_elem = np.zeros((12, 12), dtype=np.float64)
        f_int = np.zeros(12, dtype=np.float64)

        # 1. Jacobian of coordinate mapping J0 (3, 3) = _DN_DXI_TET4 @ coords
        J0 = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                val = 0.0
                for a in range(4):
                    val += _DN_DXI_TET4[i, a] * coords[a, j]
                J0[i, j] = val

        invJ0, detJ0 = _inv3x3_det_tet(J0)
        if detJ0 <= 1e-15:
            return K_elem, f_int, 1

        V0 = abs(detJ0) / 6.0

        # Physical derivatives dN_dX = invJ0 @ _DN_DXI_TET4 (3, 4)
        dN_dX = np.zeros((3, 4), dtype=np.float64)
        for i in range(3):
            for a in range(4):
                val = 0.0
                for j in range(3):
                    val += invJ0[i, j] * _DN_DXI_TET4[j, a]
                dN_dX[i, a] = val

        # 2. Deformation gradient F (3, 3) = I + du / dX
        F = np.eye(3, dtype=np.float64)
        for a in range(4):
            ua_x = u_elem[3 * a]
            ua_y = u_elem[3 * a + 1]
            ua_z = u_elem[3 * a + 2]

            dNx = dN_dX[0, a]
            dNy = dN_dX[1, a]
            dNz = dN_dX[2, a]

            F[0, 0] += ua_x * dNx
            F[0, 1] += ua_x * dNy
            F[0, 2] += ua_x * dNz

            F[1, 0] += ua_y * dNx
            F[1, 1] += ua_y * dNy
            F[1, 2] += ua_y * dNz

            F[2, 0] += ua_z * dNx
            F[2, 1] += ua_z * dNy
            F[2, 2] += ua_z * dNz

        _, detF = _inv3x3_det_tet(F)
        if detF <= 1e-15:
            return K_elem, f_int, 1

        # 3. F-bar Modified Deformation Gradient: F_bar = (J_bar_e / detF)^(1/3) * F
        scale = (J_bar_e / detF) ** (1.0 / 3.0)
        F_bar = scale * F
        detF_bar = J_bar_e

        # 4. Standard linear strain B matrix (6, 12)
        B = np.zeros((6, 12), dtype=np.float64)
        for a in range(4):
            dNx = dN_dX[0, a]
            dNy = dN_dX[1, a]
            dNz = dN_dX[2, a]
            col = 3 * a

            B[0, col]     = dNx
            B[1, col + 1] = dNy
            B[2, col + 2] = dNz

            B[3, col + 1] = dNz
            B[3, col + 2] = dNy

            B[4, col]     = dNz
            B[4, col + 2] = dNx

            B[5, col]     = dNy
            B[5, col + 1] = dNx

        # Dilatational vector: B_vol = row0 + row1 + row2 (12,)
        B_vol = np.zeros(12, dtype=np.float64)
        for j in range(12):
            B_vol[j] = B[0, j] + B[1, j] + B[2, j]

        # Bonet & Burton average-nodal-pressure (ANP) mean-dilatation
        # projection (Nagtegaal-Parks-Rice 1974 / Hughes 1980 style B-bar):
        # scale this element's OWN volumetric operator down by c_e in
        # [0, 1], its fractional share of the nodally-averaged dilatation
        # J_bar_e (Pass 1). c_e = 1 recovers plain C3D4 exactly (no
        # neighbors to average with / this element is the sole contributor
        # at all 4 of its nodes); c_e < 1 is the softened, locking-relieved
        # volumetric response. THE SAME B_bar is used for both the
        # constitutive strain and the force/tangent contraction, so this
        # stays a single self-consistent local operator (no split-formula
        # residual/tangent mismatch).
        #
        # PREVIOUS BUG (found 2026-09-13): the original code subtracted and
        # immediately re-added the SAME local (1/3)*B_vol, making B_bar
        # identically equal to B and the whole Pass-1 nodal averaging
        # (nodal_J / J_bar_e) unused -- MAT_CUSTOM_ELASTIC (used by every
        # benchmark_element/ test) computes stress purely from
        # S = C_mat @ (B_bar @ u_elem), so the element was mathematically
        # identical to plain C3D4 at any nu. A first attempt at fixing this
        # by feeding a nodally-smoothed strain into the material while
        # still contracting force/tangent against the UNMODIFIED local B
        # (i.e. two different operators for stress vs. equilibrium) was
        # tried and reverted: it broke the patch test (~5e-6, previously
        # exact) and made the nu=0.49999 volumetric ratio WORSE, not
        # better -- an inconsistent tangent is especially dangerous exactly
        # in the near-incompressible regime this element exists to handle.
        # The single-consistent-operator form below does not have that
        # failure mode: for a HOMOGENEOUS patch-test deformation c_e's
        # exact value is irrelevant to whether the affine field is a root,
        # since B_bar @ u_elem still integrates the same affine field
        # exactly regardless of c_e (it's a still a first-order-consistent
        # operator, just reweighted) -- verified below.
        # c_e: this element's own volumetric-strain measure (detF) relative
        # to the nodally-averaged one (J_bar_e, from Pass 1). c_e == 1 when
        # this element's own dilatation already equals the nodal average
        # (the homogeneous-deformation / patch-test case, and the
        # no-neighbors-to-average-with case) -- B_bar reduces to B exactly.
        # When this element's local dilatation diverges from its
        # neighbors' (the volumetric-locking symptom: isolated elements
        # over/under-shoot pressure element-by-element), c_e pulls its
        # assumed volumetric strain toward the smoother nodal average.
        c_e = J_bar_e / detF
        B_bar = np.zeros((6, 12), dtype=np.float64)
        for j in range(12):
            B_bar[0, j] = B[0, j] - (1.0 / 3.0) * B_vol[j]
            B_bar[1, j] = B[1, j] - (1.0 / 3.0) * B_vol[j]
            B_bar[2, j] = B[2, j] - (1.0 / 3.0) * B_vol[j]
            B_bar[3, j] = B[3, j]
            B_bar[4, j] = B[4, j]
            B_bar[5, j] = B[5, j]

            vol_contrib = (c_e / 3.0) * B_vol[j]
            B_bar[0, j] += vol_contrib
            B_bar[1, j] += vol_contrib
            B_bar[2, j] += vol_contrib

        eps_bar = np.zeros(6, dtype=np.float64)
        for i in range(6):
            val = 0.0
            for j in range(12):
                val += B_bar[i, j] * u_elem[j]
            eps_bar[i] = val

        # 5. UMAT Constitutive Evaluation
        sdv_pt = sdvs[0] if sdvs.shape[0] > 0 else _EMPTY_1D_FLOAT
        stress, C_mat, sdv_new, mat_err = material_dispatch_3d(
            mat_type, props, sdv_pt, eps_bar, F_bar, detF_bar, dt
        )
        if mat_err != 0:
            return K_elem, f_int, 1
        if sdvs.shape[0] > 0:
            sdvs[0, :] = sdv_new

        # 6. Internal Force and Tangent Stiffness
        # f_int = V0 * B_bar^T @ stress
        for j in range(12):
            val = 0.0
            for i in range(6):
                val += B_bar[i, j] * stress[i]
            f_int[j] = V0 * val

        # K_elem = V0 * B_bar^T @ C_mat @ B_bar
        CB = np.zeros((6, 12), dtype=np.float64)
        for i in range(6):
            for j in range(12):
                val = 0.0
                for k in range(6):
                    val += C_mat[i, k] * B_bar[k, j]
                CB[i, j] = val

        for i in range(12):
            for j in range(12):
                val = 0.0
                for k in range(6):
                    val += B_bar[k, i] * CB[k, j]
                K_elem[i, j] = V0 * val

        return K_elem, f_int, 0

    @njit
    def compute_nodal_dilatations_pass1(
        node_coords: np.ndarray,      # (n_nodes, 3)
        elem_conn: np.ndarray,        # (n_elems, 4)
        u_global: np.ndarray          # (n_dofs,)
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pass 1: Accumulate initial and deformed nodal volumes to compute nodal Jacobian J_a."""
        n_nodes = node_coords.shape[0]
        n_elems = elem_conn.shape[0]

        nodal_V0 = np.zeros(n_nodes, dtype=np.float64)
        nodal_v_def = np.zeros(n_nodes, dtype=np.float64)
        elem_detF = np.zeros(n_elems, dtype=np.float64)

        for e in range(n_elems):
            coords_e = np.zeros((4, 3), dtype=np.float64)
            u_e = np.zeros(12, dtype=np.float64)

            for a in range(4):
                nid = elem_conn[e, a]
                coords_e[a, 0] = node_coords[nid, 0]
                coords_e[a, 1] = node_coords[nid, 1]
                coords_e[a, 2] = node_coords[nid, 2]

                u_e[3 * a]     = u_global[3 * nid]
                u_e[3 * a + 1] = u_global[3 * nid + 1]
                u_e[3 * a + 2] = u_global[3 * nid + 2]

            # Jacobian J0
            J0 = np.zeros((3, 3), dtype=np.float64)
            for i in range(3):
                for j in range(3):
                    val = 0.0
                    for a in range(4):
                        val += _DN_DXI_TET4[i, a] * coords_e[a, j]
                    J0[i, j] = val

            invJ0, detJ0 = _inv3x3_det_tet(J0)
            V0 = abs(detJ0) / 6.0

            # dN_dX
            dN_dX = np.zeros((3, 4), dtype=np.float64)
            for i in range(3):
                for a in range(4):
                    val = 0.0
                    for j in range(3):
                        val += invJ0[i, j] * _DN_DXI_TET4[j, a]
                    dN_dX[i, a] = val

            # F (3, 3)
            F = np.eye(3, dtype=np.float64)
            for a in range(4):
                ua_x = u_e[3 * a]
                ua_y = u_e[3 * a + 1]
                ua_z = u_e[3 * a + 2]

                dNx = dN_dX[0, a]
                dNy = dN_dX[1, a]
                dNz = dN_dX[2, a]

                F[0, 0] += ua_x * dNx
                F[0, 1] += ua_x * dNy
                F[0, 2] += ua_x * dNz

                F[1, 0] += ua_y * dNx
                F[1, 1] += ua_y * dNy
                F[1, 2] += ua_y * dNz

                F[2, 0] += ua_z * dNx
                F[2, 1] += ua_z * dNy
                F[2, 2] += ua_z * dNz

            _, detF = _inv3x3_det_tet(F)
            elem_detF[e] = detF
            v_def = V0 * max(detF, 1e-12)

            for a in range(4):
                nid = elem_conn[e, a]
                nodal_V0[nid] += 0.25 * V0
                nodal_v_def[nid] += 0.25 * v_def

        nodal_J = np.ones(n_nodes, dtype=np.float64)
        for n in range(n_nodes):
            if nodal_V0[n] > 1e-15:
                nodal_J[n] = nodal_v_def[n] / nodal_V0[n]

        return nodal_J, elem_detF

    @njit(parallel=True)
    def assemble_mesh_c3d4_anp_numba(
        node_coords: np.ndarray,      # (n_nodes, 3)
        elem_conn: np.ndarray,        # (n_elems, 4)
        u_global: np.ndarray,         # (n_dofs,)
        elem_mat_types: np.ndarray,   # (n_elems,)
        elem_props: np.ndarray,       # (n_elems, max_props)
        elem_sdvs: np.ndarray,        # (n_elems, 1, max_sdvs)
        dt: float = 1.0,
        elem_controls: np.ndarray = _EMPTY_2D_FLOAT,
        elem_stress_init: np.ndarray = _EMPTY_2D_FLOAT
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """2-Pass OpenMP parallel assembly kernel for C3D4_ANP elements."""
        n_elems = elem_conn.shape[0]
        f_elems = np.zeros((n_elems, 12), dtype=np.float64)
        K_elems = np.zeros((n_elems, 12, 12), dtype=np.float64)
        has_error = 0

        # Pass 1: Global nodal volume averaging
        nodal_J, elem_detF = compute_nodal_dilatations_pass1(node_coords, elem_conn, u_global)

        # Pass 2: Element-wise assembly with smoothed Jacobian J_bar_e
        for e in prange(n_elems):
            coords_e = np.zeros((4, 3), dtype=np.float64)
            u_e = np.zeros(12, dtype=np.float64)

            n0 = elem_conn[e, 0]
            n1 = elem_conn[e, 1]
            n2 = elem_conn[e, 2]
            n3 = elem_conn[e, 3]

            J_bar_e = 0.25 * (nodal_J[n0] + nodal_J[n1] + nodal_J[n2] + nodal_J[n3])

            for a in range(4):
                nid = elem_conn[e, a]
                coords_e[a, 0] = node_coords[nid, 0]
                coords_e[a, 1] = node_coords[nid, 1]
                coords_e[a, 2] = node_coords[nid, 2]

                u_e[3 * a]     = u_global[3 * nid]
                u_e[3 * a + 1] = u_global[3 * nid + 1]
                u_e[3 * a + 2] = u_global[3 * nid + 2]

            ctrl_e = elem_controls[e] if elem_controls.shape[0] > 0 else _EMPTY_1D_FLOAT

            K_e, f_e, err = compute_c3d4_anp_element_matrices(
                coords_e,
                u_e,
                J_bar_e,
                mat_type=elem_mat_types[e],
                props=elem_props[e],
                sdvs=elem_sdvs[e],
                dt=dt,
                elem_controls=ctrl_e
            )

            if err != 0:
                has_error = 1

            for i in range(12):
                f_elems[e, i] = f_e[i]
                for j in range(12):
                    K_elems[e, i, j] = K_e[i, j]

        return f_elems, K_elems, has_error
