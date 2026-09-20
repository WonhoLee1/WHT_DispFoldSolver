import numpy as np

try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from .c3d8_numba import _sd3d, _jacobian3d, _GP_GAUSS

if HAS_NUMBA:
    @njit(fastmath=True)
    def _compute_invT0_T_numba(J0: np.ndarray) -> np.ndarray:
        j11, j12, j13 = J0[0, 0], J0[0, 1], J0[0, 2]
        j21, j22, j23 = J0[1, 0], J0[1, 1], J0[1, 2]
        j31, j32, j33 = J0[2, 0], J0[2, 1], J0[2, 2]

        T0 = np.array([
            [j11*j11, j12*j12, j13*j13, 2.0*j11*j12, 2.0*j12*j13, 2.0*j13*j11],
            [j21*j21, j22*j22, j23*j23, 2.0*j21*j22, 2.0*j22*j23, 2.0*j23*j21],
            [j31*j31, j32*j32, j33*j33, 2.0*j31*j32, 2.0*j32*j33, 2.0*j33*j31],
            [j11*j21, j12*j22, j13*j23, j11*j22+j12*j21, j12*j23+j13*j22, j13*j21+j11*j23],
            [j21*j31, j22*j32, j23*j33, j21*j32+j22*j31, j22*j33+j23*j32, j23*j31+j21*j33],
            [j31*j11, j32*j12, j33*j13, j31*j12+j32*j11, j32*j13+j33*j12, j33*j11+j31*j13]
        ], dtype=np.float64)
        return np.linalg.inv(T0).T

    @njit(fastmath=True)
    def _add_geometric_stiffness_3d(
        K_local: np.ndarray,
        dN_dX: np.ndarray,
        dN_dY: np.ndarray,
        dN_dZ: np.ndarray,
        stress_voigt: np.ndarray,
        dV: float
    ):
        """Accumulate 3D Geometric Stiffness (Initial Stress Matrix K_sigma) into 24x24 K_local."""
        sig0, sig1, sig2 = stress_voigt[0], stress_voigt[1], stress_voigt[2]
        sig3, sig4, sig5 = stress_voigt[3], stress_voigt[4], stress_voigt[5]

        for a in range(8):
            ga_x = dN_dX[a]
            ga_y = dN_dY[a]
            ga_z = dN_dZ[a]

            s_ga_x = sig0 * ga_x + sig3 * ga_y + sig5 * ga_z
            s_ga_y = sig3 * ga_x + sig1 * ga_y + sig4 * ga_z
            s_ga_z = sig5 * ga_x + sig4 * ga_y + sig2 * ga_z

            for b in range(8):
                gb_x = dN_dX[b]
                gb_y = dN_dY[b]
                gb_z = dN_dZ[b]

                s_ab = (s_ga_x * gb_x + s_ga_y * gb_y + s_ga_z * gb_z) * dV

                K_local[3 * a + 0, 3 * b + 0] += s_ab
                K_local[3 * a + 1, 3 * b + 1] += s_ab
                K_local[3 * a + 2, 3 * b + 2] += s_ab

    @njit(fastmath=True)
    def compute_solid_shell_element_numba(coords: np.ndarray, u_elem: np.ndarray, mat_type: int, props: np.ndarray, sdvs: np.ndarray, dt: float, elem_controls: np.ndarray, elem_stress_init: np.ndarray):
        J0, detJ0, _ = _jacobian3d(0.0, 0.0, 0.0, coords)
        invT0_T = _compute_invT0_T_numba(J0)

        K_uu = np.zeros((24, 24), dtype=np.float64)
        K_ua = np.zeros((24, 9), dtype=np.float64)
        K_aa = np.zeros((9, 9), dtype=np.float64)
        f_int = np.zeros(24, dtype=np.float64)

        # Build material matrix based on mat_type and props
        C_mat = np.zeros((6, 6), dtype=np.float64)
        if mat_type == 99:  # MAT_CUSTOM_ELASTIC
            for i in range(6):
                for j in range(6):
                    C_mat[i, j] = props[6 * i + j]
        else:
            E = props[0]
            nu = props[1]
            lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
            mu = E / (2.0 * (1.0 + nu))
            C_mat[0, 0] = lam + 2*mu; C_mat[0, 1] = lam;        C_mat[0, 2] = lam
            C_mat[1, 0] = lam;        C_mat[1, 1] = lam + 2*mu; C_mat[1, 2] = lam
            C_mat[2, 0] = lam;        C_mat[2, 1] = lam;        C_mat[2, 2] = lam + 2*mu
            C_mat[3, 3] = mu
            C_mat[4, 4] = mu
            C_mat[5, 5] = mu

        # Evaluate tying points for ANS
        # A(0,-1,0), B(1,0,0), C(0,1,0), D(-1,0,0)
        # Compute B matrices at tying points to replace shear components at GP
        def get_B_nat_shear(xi, eta, zeta):
            dN_dxi, dN_deta, dN_dzeta = _sd3d(xi, eta, zeta)
            J_mat, _, invJ = _jacobian3d(xi, eta, zeta, coords)
            T_mat = _compute_invT0_T_numba(J_mat)  # Actually this returns inv(T).T = T_inv^T.
            # Wait, the helper _compute_invT0_T_numba returns np.linalg.inv(T0).T.
            # If we want T0, we should just write a helper for T0. Let's just build it inline.
            
            j11, j12, j13 = J_mat[0, 0], J_mat[0, 1], J_mat[0, 2]
            j21, j22, j23 = J_mat[1, 0], J_mat[1, 1], J_mat[1, 2]
            j31, j32, j33 = J_mat[2, 0], J_mat[2, 1], J_mat[2, 2]

            T0 = np.array([
                [j11*j11, j12*j12, j13*j13, 2.0*j11*j12, 2.0*j12*j13, 2.0*j13*j11],
                [j21*j21, j22*j22, j23*j23, 2.0*j21*j22, 2.0*j22*j23, 2.0*j23*j21],
                [j31*j31, j32*j32, j33*j33, 2.0*j31*j32, 2.0*j32*j33, 2.0*j33*j31],
                [j11*j21, j12*j22, j13*j23, j11*j22+j12*j21, j12*j23+j13*j22, j13*j21+j11*j23],
                [j21*j31, j22*j32, j23*j33, j21*j32+j22*j31, j22*j33+j23*j32, j23*j31+j21*j33],
                [j31*j11, j32*j12, j33*j13, j31*j12+j32*j11, j32*j13+j33*j12, j33*j11+j31*j13]
            ], dtype=np.float64)

            B_cart = np.zeros((6, 24), dtype=np.float64)
            for i in range(8):
                dx = invJ[0, 0] * dN_dxi[i] + invJ[0, 1] * dN_deta[i] + invJ[0, 2] * dN_dzeta[i]
                dy = invJ[1, 0] * dN_dxi[i] + invJ[1, 1] * dN_deta[i] + invJ[1, 2] * dN_dzeta[i]
                dz = invJ[2, 0] * dN_dxi[i] + invJ[2, 1] * dN_deta[i] + invJ[2, 2] * dN_dzeta[i]
                B_cart[0, 3*i] = dx
                B_cart[1, 3*i+1] = dy
                B_cart[2, 3*i+2] = dz
                B_cart[3, 3*i] = dy
                B_cart[3, 3*i+1] = dx
                B_cart[4, 3*i+1] = dz
                B_cart[4, 3*i+2] = dy
                B_cart[5, 3*i] = dz
                B_cart[5, 3*i+2] = dx
            
            B_nat = T0 @ B_cart
            B_s = np.zeros((2, 24), dtype=np.float64)
            B_s[0, :] = B_nat[5, :]  # xi_zeta
            B_s[1, :] = B_nat[4, :]  # eta_zeta
            return B_s
        
        B_nat_A = get_B_nat_shear(0.0, -1.0, 0.0)
        B_nat_B = get_B_nat_shear(1.0, 0.0, 0.0)
        B_nat_C = get_B_nat_shear(0.0, 1.0, 0.0)
        B_nat_D = get_B_nat_shear(-1.0, 0.0, 0.0)

        for gi in range(2):
            xi = _GP_GAUSS[gi]
            for gj in range(2):
                eta = _GP_GAUSS[gj]
                for gk in range(2):
                    zeta = _GP_GAUSS[gk]

                    dN_dxi, dN_deta, dN_dzeta = _sd3d(xi, eta, zeta)
                    J_gp, detJ, invJ = _jacobian3d(xi, eta, zeta, coords)

                    B_cart = np.zeros((6, 24), dtype=np.float64)
                    dN_dX = np.zeros(8, dtype=np.float64)
                    dN_dY = np.zeros(8, dtype=np.float64)
                    dN_dZ = np.zeros(8, dtype=np.float64)
                    for i in range(8):
                        dx = invJ[0, 0] * dN_dxi[i] + invJ[0, 1] * dN_deta[i] + invJ[0, 2] * dN_dzeta[i]
                        dy = invJ[1, 0] * dN_dxi[i] + invJ[1, 1] * dN_deta[i] + invJ[1, 2] * dN_dzeta[i]
                        dz = invJ[2, 0] * dN_dxi[i] + invJ[2, 1] * dN_deta[i] + invJ[2, 2] * dN_dzeta[i]
                        dN_dX[i] = dx
                        dN_dY[i] = dy
                        dN_dZ[i] = dz

                        B_cart[0, 3*i] = dx
                        B_cart[1, 3*i+1] = dy
                        B_cart[2, 3*i+2] = dz
                        B_cart[3, 3*i] = dy
                        B_cart[3, 3*i+1] = dx
                        B_cart[4, 3*i+1] = dz
                        B_cart[4, 3*i+2] = dy
                        B_cart[5, 3*i] = dz
                        B_cart[5, 3*i+2] = dx

                    j11, j12, j13 = J_gp[0, 0], J_gp[0, 1], J_gp[0, 2]
                    j21, j22, j23 = J_gp[1, 0], J_gp[1, 1], J_gp[1, 2]
                    j31, j32, j33 = J_gp[2, 0], J_gp[2, 1], J_gp[2, 2]

                    T0_gp = np.array([
                        [j11*j11, j12*j12, j13*j13, 2.0*j11*j12, 2.0*j12*j13, 2.0*j13*j11],
                        [j21*j21, j22*j22, j23*j23, 2.0*j21*j22, 2.0*j22*j23, 2.0*j23*j21],
                        [j31*j31, j32*j32, j33*j33, 2.0*j31*j32, 2.0*j32*j33, 2.0*j33*j31],
                        [j11*j21, j12*j22, j13*j23, j11*j22+j12*j21, j12*j23+j13*j22, j13*j21+j11*j23],
                        [j21*j31, j22*j32, j23*j33, j21*j32+j22*j31, j22*j33+j23*j32, j23*j31+j21*j33],
                        [j31*j11, j32*j12, j33*j13, j31*j12+j32*j11, j32*j13+j33*j12, j33*j11+j31*j13]
                    ], dtype=np.float64)

                    B_nat_gp = T0_gp @ B_cart

                    # Replace transverse shear with ANS MITC Covariant Strains
                    B_nat_xz = 0.5 * (1 - eta) * B_nat_A[0] + 0.5 * (1 + eta) * B_nat_C[0]
                    B_nat_yz = 0.5 * (1 - xi) * B_nat_D[1] + 0.5 * (1 + xi) * B_nat_B[1]
                    B_nat_gp[4, :] = B_nat_yz
                    B_nat_gp[5, :] = B_nat_xz

                    # Map back to physical Cartesian strains
                    B_trans = np.linalg.solve(T0_gp, B_nat_gp)
                    B = B_cart.copy()
                    B[4, :] = B_trans[4, :]
                    B[5, :] = B_trans[5, :]

                    # EAS M matrix (9 parameters)
                    M = np.zeros((6, 9), dtype=np.float64)
                    M[0, 0] = xi
                    M[1, 1] = eta
                    M[2, 2] = zeta
                    M[3, 3] = xi
                    M[3, 4] = eta
                    M[4, 5] = eta
                    M[4, 6] = zeta
                    M[5, 7] = zeta
                    M[5, 8] = xi
                    B_tilde = (detJ0 / detJ) * (invT0_T @ M)

                    dV = detJ
                    K_uu += (B.T @ C_mat @ B) * dV
                    K_ua += (B.T @ C_mat @ B_tilde) * dV
                    K_aa += (B_tilde.T @ C_mat @ B_tilde) * dV
                    
                    strain = B @ u_elem
                    stress = C_mat @ strain
                    f_int += (B.T @ stress) * dV

        inv_K_aa = np.linalg.inv(K_aa)
        K_cond = K_uu - K_ua @ inv_K_aa @ K_ua.T
        alpha_opt = -inv_K_aa @ (K_ua.T @ u_elem)
        f_cond = f_int + K_ua @ alpha_opt

        return K_cond, f_cond, 0

    @njit(parallel=True, fastmath=True, nogil=True)
    def assemble_mesh_solid_shell_numba(node_coords, elem_conn, u_global, mat_types, props, sdvs, dt, elem_controls=None, elem_stress_init=None):
        n_elems = elem_conn.shape[0]
        f_elems = np.zeros((n_elems, 24), dtype=np.float64)
        K_elems = np.zeros((n_elems, 24, 24), dtype=np.float64)
        err_out = 0

        for e in prange(n_elems):
            nodes_e = elem_conn[e]
            coords_e = np.zeros((8, 3), dtype=np.float64)
            u_e = np.zeros(24, dtype=np.float64)

            for i in range(8):
                nid = nodes_e[i]
                coords_e[i, 0] = node_coords[nid, 0]
                coords_e[i, 1] = node_coords[nid, 1]
                coords_e[i, 2] = node_coords[nid, 2]
                u_e[3*i] = u_global[3*nid]
                u_e[3*i+1] = u_global[3*nid+1]
                u_e[3*i+2] = u_global[3*nid+2]

            ctrls = np.zeros(8, dtype=np.float64)
            s_init = np.zeros((8,6), dtype=np.float64)
            Ke, fe, err = compute_solid_shell_element_numba(coords_e, u_e, mat_types[e], props[e], sdvs[e], dt, ctrls, s_init)
            K_elems[e] = Ke
            f_elems[e] = fe
            if err > 0:
                err_out = 1

        return f_elems, K_elems, err_out
