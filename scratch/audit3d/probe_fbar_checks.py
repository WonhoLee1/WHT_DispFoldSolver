import numpy as np
import sys
sys.path.insert(0, r"D:\PythonCodeStudy\WHT_DispFoldSolver")
sys.path.insert(0, r"D:\PythonCodeStudy\WHT_DispFoldSolver\scratch\audit3d")
import _bootstrap

fbar = _bootstrap.load_element3d_module("c3d8_fbar_tl_numba")
from dispsolver.material3d.numba_materials import MAT_CUSTOM_ELASTIC

np.random.seed(4)
coords0 = np.array([
    [0,0,0],[1,0,0],[1,1,0],[0,1,0],
    [0,0,1],[1,0,1],[1,1,1],[0,1,1],
], dtype=np.float64)
coords0 = coords0 + 0.15 * np.array([
    [0.3,-0.2,0.1],[-0.1,0.25,-0.15],[0.2,-0.1,0.05],[-0.25,0.1,0.2],
    [0.1,0.05,-0.3],[-0.2,-0.15,0.25],[0.05,0.2,-0.1],[-0.15,-0.05,0.3],
])
E, nu = 1000.0, 0.3
lam = E*nu/((1+nu)*(1-2*nu)); mu = E/(2*(1+nu))
C_mat = np.zeros((6,6))
C_mat[0,0]=C_mat[1,1]=C_mat[2,2]=lam+2*mu
C_mat[0,1]=C_mat[0,2]=C_mat[1,0]=C_mat[1,2]=C_mat[2,0]=C_mat[2,1]=lam
C_mat[3,3]=C_mat[4,4]=C_mat[5,5]=mu
props = np.zeros(36); props[:] = C_mat.ravel()
sdvs = np.zeros((8,0))

u0 = 0.05*np.random.randn(24)

# --- Check 6: C8 FD Jacobian ---
def fofu(coords, u):
    f = fbar._compute_c3d8_fbar_tl_element_umat_numba(coords, u, MAT_CUSTOM_ELASTIC, props, sdvs.copy(), 1.0)[1]
    return f

K_an, f0, err = fbar._compute_c3d8_fbar_tl_element_umat_numba(coords0, u0, MAT_CUSTOM_ELASTIC, props, sdvs.copy(), 1.0)
n=24; K_fd=np.zeros((n,n)); h=1e-6*max(np.max(np.abs(u0)),1.0)
for j in range(n):
    up=u0.copy(); up[j]+=h; um=u0.copy(); um[j]-=h
    K_fd[:,j] = (fofu(coords0,up)-fofu(coords0,um))/(2*h)
rel = np.max(np.abs(K_an-K_fd))/max(np.max(np.abs(K_fd)),1e-300)
print("=== F-bar TL: C8 FD Jacobian check ===")
print("max|K_an-K_fd| =", np.max(np.abs(K_an-K_fd)), " max|K_fd|=", np.max(np.abs(K_fd)), " rel=", rel)
print("K_an symmetric?", np.max(np.abs(K_an-K_an.T)))

# --- Check 5: global-axis isotropy (rotate reference geometry AND displacement together) ---
def rotation_matrix(axis, deg):
    axis = axis/np.linalg.norm(axis)
    th = np.deg2rad(deg)
    K_ = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3) + np.sin(th)*K_ + (1-np.cos(th))*(K_@K_)

print()
print("=== F-bar TL: global-axis isotropy (rotate reference+displacement together) ===")
for axis in [np.array([0,0,1.0]), np.array([1,1,1.0])]:
    for deg in [10,30,45,60,90]:
        R = rotation_matrix(axis, deg)
        coords_rot = coords0 @ R.T
        # current physical positions rotate too: x_rot = R @ (X + u) => u_rot = R@(X+u) - X_rot = R@u (since X_rot=R@X)
        u_nodes = u0.reshape(8,3)
        u_rot_nodes = u_nodes @ R.T
        u_rot = u_rot_nodes.reshape(24)
        f_rot = fofu(coords_rot, u_rot)
        f_orig = fofu(coords0, u0)
        f_orig_nodes = f_orig.reshape(8,3)
        f_pred_nodes = f_orig_nodes @ R.T
        f_pred = f_pred_nodes.reshape(24)
        err_iso = np.max(np.abs(f_rot - f_pred)) / max(np.max(np.abs(f_pred)), 1e-300)
        print(f"axis={axis}, angle={deg:3d} deg: isotropy rel err = {err_iso:.3e}")

# --- Check 7: detF passed to material_dispatch_3d ---
import inspect
print()
print("=== F-bar TL: detF argument passed to material_dispatch_3d ===")
src = inspect.getsource(fbar._compute_c3d8_fbar_tl_element_umat_numba) if hasattr(fbar._compute_c3d8_fbar_tl_element_umat_numba,'__wrapped__') else None
