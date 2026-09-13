import numpy as np
import sys
sys.path.insert(0, r"D:\PythonCodeStudy\WHT_DispFoldSolver")
sys.path.insert(0, r"D:\PythonCodeStudy\WHT_DispFoldSolver\scratch\audit3d")
import _bootstrap

corot = _bootstrap.load_element3d_module("c3d8_corotational_numba")
from dispsolver.material3d.numba_materials import MAT_CUSTOM_ELASTIC

np.random.seed(2)

E, nu = 1000.0, 0.3
lam = E*nu/((1+nu)*(1-2*nu)); mu = E/(2*(1+nu))
C_mat = np.zeros((6,6))
C_mat[0,0]=C_mat[1,1]=C_mat[2,2]=lam+2*mu
C_mat[0,1]=C_mat[0,2]=C_mat[1,0]=C_mat[1,2]=C_mat[2,0]=C_mat[2,1]=lam
C_mat[3,3]=C_mat[4,4]=C_mat[5,5]=mu
props = np.zeros(36); props[:] = C_mat.ravel()
sdvs = np.zeros((8,0))
controls = np.array([0.0, 0.1, 0.0, 0.0, 0.02])  # distortion/anti-inversion OFF for clean physics check
stress_init = np.empty((0,0))

def base_coords(ar=1.0):
    c = np.array([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1],
    ], dtype=np.float64)
    c[:, 0] *= ar  # stretch x-direction -> aspect ratio
    return c

def distort(c, seed=0):
    rng = np.random.RandomState(seed)
    return c + 0.1 * rng.randn(*c.shape)

def call_kernel(coords, u_elem):
    return corot.compute_c3d8_corotational_element_umat_numba(
        coords, u_elem, MAT_CUSTOM_ELASTIC, props, sdvs.copy(), 1.0, controls, stress_init
    )

def rotmat(axis, deg):
    ang = np.deg2rad(deg)
    axis = np.asarray(axis, dtype=float); axis /= np.linalg.norm(axis)
    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3) + np.sin(ang)*K + (1-np.cos(ang))*(K@K)

# ---------------------------------------------------------------
# Check 4: rigid-rotation-only canary (no straining at all)
# ---------------------------------------------------------------
print("=== Check 4: rigid-rotation-only canary ===")
coords0 = distort(base_coords(3.0), seed=5)
for axis, deg in [((0,0,1), 45.0), ((1,1,1), 90.0), ((0,1,0), 120.0)]:
    R = rotmat(axis, deg)
    coords_curr = (R @ coords0.T).T
    u_elem = (coords_curr - coords0).ravel()
    f, K, err = call_kernel(coords0, u_elem)
    print(f"axis={axis} angle={deg:5.1f}  max|f_local given only rigid rotation|={np.max(np.abs(f)):.3e}  err={err}")

# ---------------------------------------------------------------
# Check 1: global-axis isotropy (rotate reference geometry AND
# displacement together, physical deformation state held fixed)
# ---------------------------------------------------------------
print()
print("=== Check 1: global-axis isotropy (C7) ===")
coords0 = distort(base_coords(3.0), seed=1)
u_phys = 0.08 * np.random.RandomState(7).randn(24)  # a generic real straining state
f0, K0, err0 = call_kernel(coords0, u_phys)

for axis, deg in [((0,0,1),10),((0,0,1),30),((1,0,0),45),((0,1,0),60),((1,1,0),90)]:
    Q = rotmat(axis, deg)
    coords_r = (Q @ coords0.T).T
    coords_curr_orig = coords0 + u_phys.reshape(8,3)
    coords_curr_r = (Q @ coords_curr_orig.T).T
    u_r = (coords_curr_r - coords_r).ravel()
    f_r, K_r, err_r = call_kernel(coords_r, u_r)

    Qblk = np.zeros((24,24))
    for i in range(8):
        Qblk[3*i:3*i+3, 3*i:3*i+3] = Q
    f_pred = Qblk @ f0
    rel_err = np.max(np.abs(f_r - f_pred)) / max(np.max(np.abs(f_pred)), 1e-300)
    print(f"axis={axis} angle={deg:5.1f}  max|f_rotated - Q@f_orig| rel = {rel_err:.4e}   err_flags=({err0},{err_r})")

# ---------------------------------------------------------------
# Check 2: rotated-reference-only sweep at several aspect ratios
# (physical current-config displacement held fixed in GLOBAL frame,
#  only the reference geometry's ORIENTATION changes)
# ---------------------------------------------------------------
print()
print("=== Check 2: rotated-reference-only sweep (bending-like probe, AR sweep) ===")
for ar in [1.0, 7.0, 15.0, 30.0]:
    base = base_coords(ar)
    # small "bending-like" nodal perturbation: linear in local x, applied in z
    u_bend = np.zeros(24)
    for i in range(8):
        x = base[i,0]
        u_bend[3*i+2] = 0.001 * x  # tip lifts proportional to x -> bending curvature
    print(f"AR={ar:5.1f}", end="  ")
    vals = []
    for deg in [0, 10, 30, 45, 60, 90]:
        R = rotmat((0,1,0), deg)
        coords_r = (R @ base.T).T
        u_r = (R @ u_bend.reshape(8,3).T).T.ravel()
        f_r, K_r, err_r = call_kernel(coords_r, u_r)
        vals.append(np.linalg.norm(f_r))
    vals = np.array(vals)
    ratio = vals / vals[0]
    print("force-norm ratio across ref-rotation 0/10/30/45/60/90 deg:", np.round(ratio, 4))

# ---------------------------------------------------------------
# Check 3: C8 FD Jacobian check on a distorted, rotated reference
# ---------------------------------------------------------------
print()
print("=== Check 3: C8 FD Jacobian check ===")
R30 = rotmat((1,1,0), 30)
coords0 = (R30 @ distort(base_coords(2.0), seed=3).T).T
u0 = 0.05 * np.random.RandomState(9).randn(24)

def f_of_u(u):
    f, K, err = call_kernel(coords0, u)
    return f

K_analytic, f0, err = call_kernel(coords0, u0)
n = 24
K_fd = np.zeros((n,n))
h = 1e-6 * max(np.max(np.abs(u0)), 1.0)
for j in range(n):
    up = u0.copy(); up[j]+=h
    um = u0.copy(); um[j]-=h
    K_fd[:,j] = (f_of_u(up) - f_of_u(um)) / (2*h)

rel = np.max(np.abs(K_analytic - K_fd)) / max(np.max(np.abs(K_fd)),1e-300)
print("max|K_analytic - K_fd| =", np.max(np.abs(K_analytic-K_fd)))
print("max|K_fd| =", np.max(np.abs(K_fd)))
print("relative error =", rel)
print("K_analytic symmetric?", np.max(np.abs(K_analytic - K_analytic.T)))
