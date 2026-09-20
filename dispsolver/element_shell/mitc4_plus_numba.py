"""
mitc4_plus_numba.py
====================
MITC4+ 사각형 쉘 요소 — Bathe & Dvorkin (1985) 공변 횡전단 보간 완전 구현.

참조 문헌:
- Bathe, K. J., & Dvorkin, E. N. (1985). "A four-node plate bending element based
  on Mindlin/Reissner plate theory and a mixed interpolation of transverse shear."
  International Journal for Numerical Methods in Engineering, 21, 367-383.
- Lee, P. S., & Bathe, K. J. (2004). "Development of MITC isotropic triangular
  shell finite elements." Computers & Structures, 82, 945-962.
- Hughes, T. J. R., & Brezzi, F. (1989). "On drilling degrees of freedom."
  Computer Methods in Applied Mechanics and Engineering, 72, 105-121.
"""

import numpy as np
from numba import njit
from .shell_kinematics import calc_drilling_penalty_stiffness


@njit(fastmath=True)
def get_local_frame_quad(coords):
    """4절점 사각형 요소의 국소 좌표계 계산.

    Parameters
    ----------
    coords : ndarray (4, 3)
        전역 절점 좌표.

    Returns
    -------
    local_coords : ndarray (4, 2)
        국소 평면 (e1, e2) 기저의 절점 좌표.
    T_sys : ndarray (24, 24)
        전역 → 국소 변환 행렬 (6-DOF × 4절점 = 24×24).
    """
    center = (coords[0] + coords[1] + coords[2] + coords[3]) / 4.0
    v1 = coords[1] - coords[0]
    v2 = coords[2] - coords[0]
    e1 = v1 / np.linalg.norm(v1)
    e3_temp = np.cross(e1, v2)
    e3 = e3_temp / np.linalg.norm(e3_temp)
    e2 = np.cross(e3, e1)

    T_node = np.zeros((3, 3))
    T_node[0, :] = e1
    T_node[1, :] = e2
    T_node[2, :] = e3

    local_coords = np.zeros((4, 2))
    for i in range(4):
        vec = coords[i] - center
        local_coords[i, 0] = np.dot(vec, e1)
        local_coords[i, 1] = np.dot(vec, e2)

    T_sys = np.zeros((24, 24))
    for i in range(8):
        T_sys[i*3:i*3+3, i*3:i*3+3] = T_node

    return local_coords, T_sys


@njit(fastmath=True)
def shape_funcs_quad(xi, eta):
    """4절점 사각형 쌍선형 형상함수 및 편미분.

    Parameters
    ----------
    xi, eta : float
        자연 좌표 (각각 [-1, 1]).

    Returns
    -------
    N : ndarray (4,)
    dN_dxi : ndarray (4,)
    dN_deta : ndarray (4,)
    """
    N = np.array([
        0.25 * (1 - xi) * (1 - eta),
        0.25 * (1 + xi) * (1 - eta),
        0.25 * (1 + xi) * (1 + eta),
        0.25 * (1 - xi) * (1 + eta)
    ])
    dN_dxi = np.array([
        -0.25 * (1 - eta),
         0.25 * (1 - eta),
         0.25 * (1 + eta),
        -0.25 * (1 + eta)
    ])
    dN_deta = np.array([
        -0.25 * (1 - xi),
        -0.25 * (1 + xi),
         0.25 * (1 + xi),
         0.25 * (1 - xi)
    ])
    return N, dN_dxi, dN_deta


@njit(fastmath=True)
def _mitc4_shear_B_at_tying_point(xi_t, eta_t, local_coords):
    """타이포인트 (xi_t, eta_t)에서 공변 횡전단 B 행렬 계산 (2×24).

    Bathe-Dvorkin 공변 변형률 관례 (공학전단 스케일 — D_s = kappa*G*t와 정합):
        행 0: e_xi_zeta  = sum_i[dN_i/dxi  * w_i] + J11*θy_i - J12*θx_i
        행 1: e_eta_zeta = sum_i[dN_i/deta * w_i] + J21*θy_i - J22*θx_i

    DOF 순서: [u, v, w, θx, θy, θz] × 4절점 (인덱스 0~23)

    Parameters
    ----------
    xi_t, eta_t : float
        타이포인트 자연 좌표.
    local_coords : ndarray (4, 2)
        국소 평면 절점 좌표.

    Returns
    -------
    Bs_cov : ndarray (2, 24)
        공변 횡전단 B 행렬.
    """
    N_t, dN_dxi_t, dN_deta_t = shape_funcs_quad(xi_t, eta_t)
    J11_t = np.sum(dN_dxi_t  * local_coords[:, 0])
    J12_t = np.sum(dN_dxi_t  * local_coords[:, 1])
    J21_t = np.sum(dN_deta_t * local_coords[:, 0])
    J22_t = np.sum(dN_deta_t * local_coords[:, 1])

    Bs_cov = np.zeros((2, 24))
    for i in range(4):
        idxi = i * 6
        # e_xi_zeta 행 (행 0) — 공학전단(engineering shear) 관례:
        # w 항 dN/dxi와 동일한 스케일. D_s = kappa*G*t가 공학전단 기준이므로
        # 회전 항에도 1/2을 붙이면(텐서 관례 혼합) 전단 에너지가 1/4로 축소되어
        # 조립 굽힘 응답이 정확히 4배 뻣뻣해짐 (Scordelis-Lo 63% 오차 원인).
        Bs_cov[0, idxi + 2] = dN_dxi_t[i]        # w
        Bs_cov[0, idxi + 4] = J11_t * N_t[i]     # θy
        Bs_cov[0, idxi + 3] = -J12_t * N_t[i]    # θx
        # e_eta_zeta 행 (행 1)
        Bs_cov[1, idxi + 2] = dN_deta_t[i]         # w
        Bs_cov[1, idxi + 4] = J21_t * N_t[i]       # θy
        Bs_cov[1, idxi + 3] = -J22_t * N_t[i]      # θx

    return Bs_cov


@njit(fastmath=True)
def compute_mitc4_plus_stiffness(coords, E, nu, thickness):
    """MITC4+ 국소 강성 행렬 계산 후 전역 좌표계로 변환.

    Parameters
    ----------
    coords : ndarray (4, 3)
        전역 절점 좌표.
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    thickness : float
        쉘 두께.

    Returns
    -------
    K_global : ndarray (24, 24)
        전역 좌표계 요소 강성 행렬.
    """
    local_coords, T_sys = get_local_frame_quad(coords)

    K_loc = np.zeros((24, 24))

    # ── 재료 행렬 ──────────────────────────────────────────────────────
    # 막 (Membrane)
    D_m = E * thickness / (1 - nu**2) * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0]
    ])
    # 굽힘 (Bending)
    D_b = (E * thickness**3 / 12.0) / (1 - nu**2) * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0]
    ])
    # 횡전단 (Transverse Shear) — Reissner-Mindlin kappa=5/6
    kappa = 5.0 / 6.0
    G = E / (2.0 * (1.0 + nu))
    D_s = kappa * G * thickness * np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    gauss_pts = np.array([-0.577350269, 0.577350269])

    # ── 막 + 굽힘 강성 (표준 2×2 Gauss 적분) ────────────────────────────
    for xi in gauss_pts:
        for eta in gauss_pts:
            N_gp, dN_dxi_gp, dN_deta_gp = shape_funcs_quad(xi, eta)

            # Jacobian
            J11 = np.sum(dN_dxi_gp  * local_coords[:, 0])
            J12 = np.sum(dN_dxi_gp  * local_coords[:, 1])
            J21 = np.sum(dN_deta_gp * local_coords[:, 0])
            J22 = np.sum(dN_deta_gp * local_coords[:, 1])
            detJ = J11 * J22 - J12 * J21
            invJ = np.array([[J22, -J12], [-J21, J11]]) / detJ

            dN_dx = invJ[0, 0] * dN_dxi_gp + invJ[0, 1] * dN_deta_gp
            dN_dy = invJ[1, 0] * dN_dxi_gp + invJ[1, 1] * dN_deta_gp

            Bm = np.zeros((3, 24))
            Bb = np.zeros((3, 24))

            for i in range(4):
                idx = i * 6
                # 막: u → d[0], v → d[1]
                Bm[0, idx]   = dN_dx[i]
                Bm[1, idx+1] = dN_dy[i]
                Bm[2, idx]   = dN_dy[i]
                Bm[2, idx+1] = dN_dx[i]

                # 굽힘: w → d[2], θx → d[3], θy → d[4]
                Bb[0, idx+4] = dN_dx[i]
                Bb[1, idx+3] = -dN_dy[i]
                Bb[2, idx+4] = dN_dy[i]
                Bb[2, idx+3] = -dN_dx[i]

            weight = detJ
            K_loc += weight * (Bm.T @ D_m @ Bm)
            K_loc += weight * (Bb.T @ D_b @ Bb)

    # ── MITC4 공변 횡전단 강성 — Bathe & Dvorkin (1985) ──────────────────
    #
    # 타이포인트:
    #   A(xi=0, eta=-1) → e_xi_zeta  보간 기저
    #   B(xi=+1, eta=0) → e_eta_zeta 보간 기저
    #   C(xi=0, eta=+1) → e_xi_zeta  보간 기저
    #   D(xi=-1, eta=0) → e_eta_zeta 보간 기저
    #
    # MITC4 이방성 선형 보간:
    #   e_xi_zeta(xi,eta)  = 0.5*(1-eta)*Bs_A[0] + 0.5*(1+eta)*Bs_C[0]
    #   e_eta_zeta(xi,eta) = 0.5*(1-xi)*Bs_D[1]  + 0.5*(1+xi)*Bs_B[1]
    # ─────────────────────────────────────────────────────────────────

    Bs_A = _mitc4_shear_B_at_tying_point(0.0, -1.0, local_coords)
    Bs_B = _mitc4_shear_B_at_tying_point(1.0,  0.0, local_coords)
    Bs_C = _mitc4_shear_B_at_tying_point(0.0,  1.0, local_coords)
    Bs_D = _mitc4_shear_B_at_tying_point(-1.0, 0.0, local_coords)

    for xi in gauss_pts:
        for eta in gauss_pts:
            _, dN_dxi_gp, dN_deta_gp = shape_funcs_quad(xi, eta)

            # Jacobian at Gauss point (공변 → Cartesian 변환용)
            J11_gp = np.sum(dN_dxi_gp  * local_coords[:, 0])
            J12_gp = np.sum(dN_dxi_gp  * local_coords[:, 1])
            J21_gp = np.sum(dN_deta_gp * local_coords[:, 0])
            J22_gp = np.sum(dN_deta_gp * local_coords[:, 1])
            detJ_gp = J11_gp * J22_gp - J12_gp * J21_gp

            # MITC4 이방성 선형 보간 (각각 2×24 행 선택)
            Bs_xi_row  = 0.5 * (1.0 - eta) * Bs_A[0:1, :] + 0.5 * (1.0 + eta) * Bs_C[0:1, :]
            Bs_eta_row = 0.5 * (1.0 - xi)  * Bs_D[1:2, :] + 0.5 * (1.0 + xi)  * Bs_B[1:2, :]

            # 공변 전단 B 행렬 (2×24): [e_xi_zeta; e_eta_zeta]
            Bs_cov_gp = np.zeros((2, 24))
            Bs_cov_gp[0, :] = Bs_xi_row[0, :]
            Bs_cov_gp[1, :] = Bs_eta_row[0, :]

            # 공변 → Cartesian 변환: {e_x_zeta, e_y_zeta} = J^{-T} {e_xi_zeta, e_eta_zeta}
            invJ_gp = np.array([[J22_gp, -J12_gp], [-J21_gp, J11_gp]]) / detJ_gp
            Bs_cart_gp = invJ_gp @ Bs_cov_gp  # (2, 24)

            weight = detJ_gp
            K_loc += weight * (Bs_cart_gp.T @ D_s @ Bs_cart_gp)

    # ── Hughes-Brezzi 드릴링 패널티 ─────────────────────────────────────
    area = 0.5 * np.linalg.norm(np.cross(coords[2] - coords[0], coords[3] - coords[1]))
    K_drill = calc_drilling_penalty_stiffness(area, E, nu, thickness, penalty_factor=1e-4)
    for i in range(4):
        K_loc[i*6+5, i*6+5] += K_drill

    return T_sys.T @ K_loc @ T_sys


@njit(fastmath=True)
def assemble_mitc4_plus(nodes, elements, E, nu, thickness):
    """MITC4+ 요소 전체 전역 강성 행렬 조립.

    Parameters
    ----------
    nodes : ndarray (n_nodes, 3)
        전역 절점 좌표 배열.
    elements : ndarray (n_elems, 4)
        요소 절점 인덱스 배열.
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    thickness : float
        쉘 두께.

    Returns
    -------
    K_global : ndarray (n_nodes*6, n_nodes*6)
        전역 강성 행렬.
    """
    n_dofs = nodes.shape[0] * 6
    K_global = np.zeros((n_dofs, n_dofs))

    for i in range(elements.shape[0]):
        n1, n2, n3, n4 = elements[i]
        coords = np.zeros((4, 3))
        coords[0] = nodes[n1]
        coords[1] = nodes[n2]
        coords[2] = nodes[n3]
        coords[3] = nodes[n4]
        Ke = compute_mitc4_plus_stiffness(coords, E, nu, thickness)

        idx = np.array([
            n1*6, n1*6+1, n1*6+2, n1*6+3, n1*6+4, n1*6+5,
            n2*6, n2*6+1, n2*6+2, n2*6+3, n2*6+4, n2*6+5,
            n3*6, n3*6+1, n3*6+2, n3*6+3, n3*6+4, n3*6+5,
            n4*6, n4*6+1, n4*6+2, n4*6+3, n4*6+4, n4*6+5
        ])

        for r in range(24):
            for c in range(24):
                K_global[idx[r], idx[c]] += Ke[r, c]

    return K_global
