"""Granular EAS block-tangent FD check (alpha frozen): isolate the wrong block."""
import numpy as np
from dispsolver.material.plastic import J2Plasticity
import dispsolver.element.q4_eas as eas
from dispsolver.element import q4

coords = np.array([[0.0, 0.0], [1.7, -0.1], [1.9, 1.2], [0.2, 1.0]])
mat = J2Plasticity(E=2000.0, nu=0.3, sigma_y0=1e9, H=50.0)  # elastic
state = np.tile(mat.initial_internal_vars(), (4, 1))

rng = np.random.default_rng(1)
u = 0.03 * rng.standard_normal(8)
alpha = 0.02 * rng.standard_normal(4)

def blocks(u, alpha):
    """Return f_u, f_a, K_uu, K_ua, K_aa at FIXED alpha (no local solve)."""
    n_gp = len(q4._GP2)
    J0, detJ0, _ = q4.jacobian(0.0, 0.0, coords)
    f_u = np.zeros(8); f_a = np.zeros(4)
    K_uu = np.zeros((8, 8)); K_ua = np.zeros((8, 4)); K_aa = np.zeros((4, 4))
    ux, uy = u[0::2], u[1::2]
    for k in range(n_gp):
        xi, et = q4._GP2[k]
        _, detJ, invJ = q4.jacobian(xi, et, coords)
        dxi, det_ = q4.shape_derivatives(xi, et)
        gX = invJ[0, 0] * dxi + invJ[0, 1] * det_
        gY = invJ[1, 0] * dxi + invJ[1, 1] * det_
        Fenh = eas._enhanced_grad_modes(xi, et, detJ, J0, detJ0)
        w = detJ * q4._W2[k]
        Fc = np.eye(2) + np.array([[ux @ gX, ux @ gY], [uy @ gX, uy @ gY]])
        Ft = Fc + sum(alpha[j] * Fenh[j] for j in range(4))
        S_v, C_v, _ = mat.pk2_voigt(Ft, {}, state[k])
        St = np.array([[S_v[0], S_v[2]], [S_v[2], S_v[1]]])
        BL = eas._BL_columns(Ft, gX, gY)
        G = np.stack([eas._voigt_sym(Ft.T @ Fenh[j]) for j in range(4)], axis=1)
        f_u += BL.T @ S_v * w
        f_a += G.T @ S_v * w
        grad_N = np.stack([gX, gY], axis=1)
        gamma = grad_N @ St @ grad_N.T
        Kuu = np.zeros((8, 8)); Kuu[0::2, 0::2] = gamma; Kuu[1::2, 1::2] = gamma
        Kua = np.zeros((8, 4))
        for a in range(4):
            for i in range(2):
                for kk in range(4):
                    Pr = np.outer(grad_N[a], Fenh[kk][i, :])
                    Ps = 0.5 * (Pr + Pr.T)
                    Kua[2*a+i, kk] = St[0,0]*Ps[0,0]+St[1,1]*Ps[1,1]+2*St[0,1]*Ps[0,1]
        Kaa = np.zeros((4, 4))
        for a in range(4):
            for b in range(4):
                P = 0.5*(Fenh[a].T@Fenh[b]+Fenh[b].T@Fenh[a])
                Kaa[a,b] = St[0,0]*P[0,0]+St[1,1]*P[1,1]+2*St[0,1]*P[0,1]
        K_uu += (BL.T@C_v@BL + Kuu)*w
        K_ua += (BL.T@C_v@G + Kua)*w
        K_aa += (G.T@C_v@G + Kaa)*w
    return f_u, f_a, K_uu, K_ua, K_aa

f_u, f_a, K_uu, K_ua, K_aa = blocks(u, alpha)
eps = 1e-7
# K_uu = d f_u / d u
Kuu_fd = np.zeros((8,8)); Kua_fd = np.zeros((8,4)); Kaa_fd = np.zeros((4,4))
for j in range(8):
    up=u.copy(); up[j]+=eps; um=u.copy(); um[j]-=eps
    Kuu_fd[:,j] = (blocks(up,alpha)[0]-blocks(um,alpha)[0])/(2*eps)
for j in range(4):
    ap=alpha.copy(); ap[j]+=eps; am=alpha.copy(); am[j]-=eps
    Kua_fd[:,j] = (blocks(u,ap)[0]-blocks(u,am)[0])/(2*eps)   # d f_u / d alpha
    Kaa_fd[:,j] = (blocks(u,ap)[1]-blocks(u,am)[1])/(2*eps)   # d f_a / d alpha
def rel(A,B): return np.linalg.norm(A-B)/(np.linalg.norm(B)+1e-30)
print(f"K_uu  err = {rel(K_uu, Kuu_fd):.3e}")
print(f"K_ua  err = {rel(K_ua, Kua_fd):.3e}")
print(f"K_aa  err = {rel(K_aa, Kaa_fd):.3e}")
