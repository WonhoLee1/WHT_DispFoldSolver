"""
coh2d4_numba.py
===============
Numba JIT accelerated 4-node 2D Cohesive Element (COH2D4).
Implements standard reference mid-surface kinematics with Bilinear Traction-Separation Law:
- Linear elasticity in normal (K_nn) and shear (K_ss) modes
- Penetration resistance (compressive contact penalty, no damage in compression)
- Damage initiation: Maximum nominal stress (MAXS) or Quadratic nominal stress (QUADS)
- Damage evolution: Linear softening with critical fracture energy G_c or failure displacement delta_f
- Monotonic irreversible damage accumulation (D in [0, 1])
- Exact algorithmic consistent tangent stiffness Ke (8x8)
- 2-point Gauss quadrature along interface length xi in [-1, 1]

Equivalents: Abaqus COH2D4 / Ansys INTER202
"""

import numpy as np
from numba import njit, prange

# 2 Gauss points in [-1, 1]
_GP1D = 1.0 / np.sqrt(3.0)
_GPS = np.array([-_GP1D, _GP1D], dtype=np.float64)
_WTS = np.array([1.0, 1.0], dtype=np.float64)


@njit(fastmath=True, cache=True)
def compute_coh2d4_element_numba(
    coords: np.ndarray,      # (4, 2) [node0=(X_bot1), node1=(X_bot2), node2=(X_top2), node3=(X_top1)]
    u_elem: np.ndarray,      # (8,) [u0x, u0y, u1x, u1y, u2x, u2y, u3x, u3y]
    props: np.ndarray,       # (64,) material and section properties
    sdvs: np.ndarray,        # (2, 9) state variables per Gauss point
    dt: float = 0.0,
    elem_controls: np.ndarray = None,
):
    """
    Computes 8x8 tangent stiffness Ke and 8-element internal force vector fe
    for a single 4-node 2D cohesive element (COH2D4).

    Parameters
    ----------
    coords : (4, 2) float64 array
        Initial coordinates of nodes:
        0: Bottom face, xi = -1
        1: Bottom face, xi = +1
        2: Top face,    xi = +1
        3: Top face,    xi = -1
    u_elem : (8,) float64 array
        Nodal displacements: [u0x, u0y, u1x, u1y, u2x, u2y, u3x, u3y]
    props : (64,) float64 array
        Constitutive properties:
        props[0]: K_nn (normal elastic stiffness, penalty for penetration)
        props[1]: K_ss (shear elastic stiffness)
        props[2]: N_max (peak normal traction / tensile strength)
        props[3]: S_max (peak shear traction / shear strength)
        props[4]: G_c (critical fracture energy)
        props[5]: init_criterion (0=MAXS, 1=QUADS)
        props[6]: evol_type (0=linear softening, 1=exponential)
        props[7]: delta_f (optional explicit displacement at total failure)
        props[8]: viscosity (viscous regularization coefficient, default 0.0)
        props[9]: thickness (out-of-plane thickness, default 1.0)
    sdvs : (2, 9) float64 array
        State variables for 2 Gauss points:
        sdvs[q, 0]: delta_m_max (history maximum of effective separation)
        sdvs[q, 1]: D (damage parameter in [0, 1])
        sdvs[q, 2]: delta_n (normal separation)
        sdvs[q, 3]: delta_s (shear separation)
        sdvs[q, 4]: t_n (normal traction)
        sdvs[q, 5]: t_s (shear traction)
        sdvs[q, 6]: status (0=elastic, 1=softening, 2=failed)
        sdvs[q, 7]: delta_m_0 (effective separation at initiation)
        sdvs[q, 8]: delta_m_f (effective separation at failure)

    Returns
    -------
    fe : (8,) float64 internal force vector
    Ke : (8, 8) float64 consistent tangent stiffness matrix
    has_error : int (0 = success)
    """
    fe = np.zeros(8, dtype=np.float64)
    Ke = np.zeros((8, 8), dtype=np.float64)
    has_error = 0

    # Extract constitutive parameters
    K_nn = props[0] if props[0] > 0.0 else 1.0e6
    K_ss = props[1] if props[1] > 0.0 else 1.0e6
    N_max = props[2]
    S_max = props[3]
    G_c = props[4]
    init_crit = int(props[5])
    delta_f_param = props[7]
    thick = props[9] if props[9] > 0.0 else 1.0

    has_damage = (N_max > 0.0 or S_max > 0.0) and (G_c > 0.0 or delta_f_param > 0.0)

    # Reference mid-surface geometry and local orientation frame:
    # dX_m/dxi = 0.25 * ((X1 - X0) + (X2 - X3))
    dXm_dxi_x = 0.25 * ((coords[1, 0] - coords[0, 0]) + (coords[2, 0] - coords[3, 0]))
    dXm_dxi_y = 0.25 * ((coords[1, 1] - coords[0, 1]) + (coords[2, 1] - coords[3, 1]))
    J0 = np.sqrt(dXm_dxi_x * dXm_dxi_x + dXm_dxi_y * dXm_dxi_y)
    if J0 < 1.0e-12:
        # Fallback to bottom face
        dX_bot_x = 0.5 * (coords[1, 0] - coords[0, 0])
        dX_bot_y = 0.5 * (coords[1, 1] - coords[0, 1])
        J0 = np.sqrt(dX_bot_x * dX_bot_x + dX_bot_y * dX_bot_y)
        if J0 < 1.0e-12:
            J0 = 0.5

    if J0 > 1.0e-12:
        tx = dXm_dxi_x / J0
        ty = dXm_dxi_y / J0
    else:
        tx = 1.0
        ty = 0.0

    # Normal vector pointing from bottom to top (+90 deg CCW rotation from tangent)
    nx = -ty
    ny = tx

    # Current deformed node positions
    x_curr = np.zeros((4, 2), dtype=np.float64)
    for i in range(4):
        x_curr[i, 0] = coords[i, 0] + u_elem[2 * i + 0]
        x_curr[i, 1] = coords[i, 1] + u_elem[2 * i + 1]

    # Gauss point integration loop
    for q in range(2):
        xi = _GPS[q]
        wt = _WTS[q]

        N1 = 0.5 * (1.0 - xi)
        N2 = 0.5 * (1.0 + xi)

        # Displacement jump in Cartesian frame:
        # Delta = x_top(xi) - x_bot(xi)
        Delta_x = (N1 * x_curr[3, 0] + N2 * x_curr[2, 0]) - (N1 * x_curr[0, 0] + N2 * x_curr[1, 0])
        Delta_y = (N1 * x_curr[3, 1] + N2 * x_curr[2, 1]) - (N1 * x_curr[0, 1] + N2 * x_curr[1, 1])

        # Local separations:
        delta_n = nx * Delta_x + ny * Delta_y
        delta_s = tx * Delta_x + ty * Delta_y

        # B-operator matrix: 2x8
        B = np.zeros((2, 8), dtype=np.float64)
        # Node 0 (bottom, N1): -N1
        B[0, 0] = -N1 * nx
        B[0, 1] = -N1 * ny
        B[1, 0] = -N1 * tx
        B[1, 1] = -N1 * ty
        # Node 1 (bottom, N2): -N2
        B[0, 2] = -N2 * nx
        B[0, 3] = -N2 * ny
        B[1, 2] = -N2 * tx
        B[1, 3] = -N2 * ty
        # Node 2 (top, N2): +N2
        B[0, 4] =  N2 * nx
        B[0, 5] =  N2 * ny
        B[1, 4] =  N2 * tx
        B[1, 5] =  N2 * ty
        # Node 3 (top, N1): +N1
        B[0, 6] =  N1 * nx
        B[0, 7] =  N1 * ny
        B[1, 6] =  N1 * tx
        B[1, 7] =  N1 * ty

        # Constitutive update:
        tn = 0.0
        ts = 0.0
        Cnn = 0.0
        Css = 0.0
        Cns = 0.0
        Csn = 0.0

        if not has_damage:
            # Pure Linear Elasticity with Penetration Resistance
            tn = K_nn * delta_n
            ts = K_ss * delta_s
            Cnn = K_nn
            Css = K_ss
            Cns = 0.0
            Csn = 0.0
            D = 0.0
            status = 0
            delta_m_0 = 1.0e-12
            delta_m_f = 1.0
            delta_m_max = np.sqrt(max(0.0, delta_n)**2 + delta_s**2)
        else:
            # Traction-Separation with Progressive Damage
            delta_n_pos = delta_n if delta_n > 0.0 else 0.0
            t_n_trial_pos = K_nn * delta_n_pos
            t_s_trial = K_ss * delta_s

            # Damage initiation index f
            if init_crit == 1:
                # QUADS (Quadratic Nominal Stress)
                val_n = (t_n_trial_pos / N_max) if N_max > 0.0 else 0.0
                val_s = (t_s_trial / S_max) if S_max > 0.0 else 0.0
                f_init = np.sqrt(val_n * val_n + val_s * val_s)
            else:
                # MAXS (Maximum Nominal Stress, default)
                fn = (t_n_trial_pos / N_max) if N_max > 0.0 else 0.0
                fs = (abs(t_s_trial) / S_max) if S_max > 0.0 else 0.0
                f_init = max(fn, fs)

            delta_m = np.sqrt(delta_n_pos * delta_n_pos + delta_s * delta_s)

            # Check if initiation state variables are already committed
            has_init = (sdvs[q, 7] > 0.0)
            if not has_init:
                if f_init > 1.0e-12:
                    delta_m_0 = delta_m / f_init
                    t_n_0 = t_n_trial_pos / f_init
                    t_s_0 = t_s_trial / f_init
                    T_eff_0 = np.sqrt(t_n_0 * t_n_0 + t_s_0 * t_s_0)
                else:
                    delta_m_0 = 1.0e-12
                    T_eff_0 = min(N_max, S_max) if min(N_max, S_max) > 0.0 else 1.0

                if delta_f_param > 0.0:
                    delta_m_f = delta_f_param
                else:
                    if G_c > 0.0 and T_eff_0 > 0.0:
                        delta_m_f = 2.0 * G_c / T_eff_0
                    else:
                        delta_m_f = 2.0 * delta_m_0

                if delta_m_f <= delta_m_0:
                    delta_m_f = delta_m_0 * 1.0001 + 1.0e-12

                if f_init >= 1.0:
                    sdvs[q, 7] = delta_m_0
                    sdvs[q, 8] = delta_m_f
            else:
                delta_m_0 = sdvs[q, 7]
                delta_m_f = sdvs[q, 8]

            # Historical maximum effective separation
            delta_m_max = sdvs[q, 0]
            if delta_m > delta_m_max:
                delta_m_max = delta_m
                sdvs[q, 0] = delta_m_max

            # Damage parameter D and derivative dD/ddelta_m
            dD_ddelta = 0.0
            if delta_m_max <= delta_m_0:
                D = 0.0
                status = 0
            elif delta_m_max >= delta_m_f:
                D = 1.0
                status = 2
            else:
                status = 1
                D = (delta_m_f * (delta_m_max - delta_m_0)) / (delta_m_max * (delta_m_f - delta_m_0))
                # Loading vs unloading check
                if delta_m >= delta_m_max - 1.0e-12:
                    dD_ddelta = (delta_m_f * delta_m_0) / (delta_m_max * delta_m_max * (delta_m_f - delta_m_0))
                else:
                    dD_ddelta = 0.0

            if D < 0.0:
                D = 0.0
            elif D > 1.0:
                D = 1.0

            sdvs[q, 1] = D

            # Normal traction & tangent
            if delta_n < 0.0:
                # Compressive contact penalty: no damage, penetration prevented
                tn = K_nn * delta_n
                Cnn = K_nn
                Cns = 0.0
                Csn = 0.0
                ts = (1.0 - D) * K_ss * delta_s
                if delta_m > 1.0e-12 and dD_ddelta > 0.0:
                    dD_ds = dD_ddelta * (delta_s / delta_m)
                    Css = (1.0 - D) * K_ss - K_ss * delta_s * dD_ds
                else:
                    Css = (1.0 - D) * K_ss
            else:
                # Tensile separation: damaged
                tn = (1.0 - D) * K_nn * delta_n
                ts = (1.0 - D) * K_ss * delta_s
                if delta_m > 1.0e-12 and dD_ddelta > 0.0:
                    dD_dn = dD_ddelta * (delta_n / delta_m)
                    dD_ds = dD_ddelta * (delta_s / delta_m)
                    Cnn = (1.0 - D) * K_nn - K_nn * delta_n * dD_dn
                    Cns = -K_nn * delta_n * dD_ds
                    Csn = -K_ss * delta_s * dD_dn
                    Css = (1.0 - D) * K_ss - K_ss * delta_s * dD_ds
                else:
                    Cnn = (1.0 - D) * K_nn
                    Cns = 0.0
                    Csn = 0.0
                    Css = (1.0 - D) * K_ss

        # Update SDVs
        sdvs[q, 2] = delta_n
        sdvs[q, 3] = delta_s
        sdvs[q, 4] = tn
        sdvs[q, 5] = ts
        sdvs[q, 6] = float(status)

        # Numerical integration weight: dGamma = J0 * wt * thick
        dGamma = J0 * wt * thick

        # Internal force: fe += B^T * t * dGamma
        for a in range(8):
            fe[a] += (B[0, a] * tn + B[1, a] * ts) * dGamma

        # Stiffness matrix: Ke += B^T * C * B * dGamma
        for a in range(8):
            for b in range(8):
                Kab = (
                    B[0, a] * (Cnn * B[0, b] + Cns * B[1, b]) +
                    B[1, a] * (Csn * B[0, b] + Css * B[1, b])
                ) * dGamma
                Ke[a, b] += Kab

    return fe, Ke, has_error


@njit(parallel=True, fastmath=True)
def assemble_mesh_coh2d4_numba(
    node_coords: np.ndarray,      # (N_nodes, 2)
    elem_conn: np.ndarray,        # (N_elems, 4)
    u: np.ndarray,                # (N_dofs,)
    elem_mat_types: np.ndarray,   # (N_elems,)
    elem_props: np.ndarray,       # (N_elems, 64)
    elem_sdvs: np.ndarray,        # (N_elems, 2, 9) or (N_elems, n_gps, max_sdvs)
    dt: float,
    elem_controls: np.ndarray = None,
    elem_stress_init: np.ndarray = None,
):
    """
    Parallel Numba assembly of all COH2D4 cohesive elements in the group.

    Returns
    -------
    f_sub : (N_elems, 8) float64
    K_sub : (N_elems, 8, 8) float64
    err_out : int (0 = success)
    """
    ne = elem_conn.shape[0]
    f_sub = np.zeros((ne, 8), dtype=np.float64)
    K_sub = np.zeros((ne, 8, 8), dtype=np.float64)
    err_out = 0

    for e in prange(ne):
        conn = elem_conn[e]
        coords = np.zeros((4, 2), dtype=np.float64)
        for i in range(4):
            nid = conn[i]
            coords[i, 0] = node_coords[nid, 0]
            coords[i, 1] = node_coords[nid, 1]

        u_elem = np.zeros(8, dtype=np.float64)
        for i in range(4):
            nid = conn[i]
            u_elem[2 * i + 0] = u[2 * nid + 0]
            u_elem[2 * i + 1] = u[2 * nid + 1]

        props_e = elem_props[e]
        sdvs_e = elem_sdvs[e]
        ctrl_e = elem_controls[e] if elem_controls is not None else None

        fe, Ke, err = compute_coh2d4_element_numba(
            coords, u_elem, props_e, sdvs_e, dt, ctrl_e
        )

        f_sub[e] = fe
        K_sub[e] = Ke
        if err != 0:
            err_out = err

    return f_sub, K_sub, err_out
