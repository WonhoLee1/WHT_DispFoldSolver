"""Independent check of the EAS J0^-1 vs J0^-T transpose claim."""
import numpy as np, jax, jax.numpy as jnp
import dispsolver.element.q4_eas_jax as m
from dispsolver.material.plastic import J2Plasticity

E,NU,SY0,H = 4000.0,0.3,1e12,0.0
mat = J2Plasticity(E=E,nu=NU,sigma_y0=SY0,H=H); LAM,MU = mat.lam,mat.mu
S0 = np.zeros((4,5)); S0[:,0]=1.0; S0[:,3]=1.0
coords = np.array([[0.,0.],[1.,0.],[1.,0.15],[0.,0.15]])
rng = np.random.default_rng(0); d = rng.uniform(-1,1,size=8)*4e-4

_orig = m._enhanced_grad_modes
def fixed_modes(xi, eta, detJ, J0, detJ0):
    J0inv = jnp.linalg.inv(J0)
    s = detJ0/detJ
    Dk = jnp.array([[[xi,0.],[0.,0.]],[[0.,eta],[0.,0.]],
                    [[0.,0.],[xi,0.]],[[0.,0.],[0.,eta]]], dtype=jnp.float64)
    return jnp.einsum('kij,mj->kim', Dk, J0inv) * s   # J0^-T

def T8R(th):
    c,s=np.cos(th),np.sin(th); R=np.array([[c,-s],[s,c]]); T=np.zeros((8,8))
    for i in range(4): T[2*i:2*i+2,2*i:2*i+2]=R
    return T,R

def isotropy(label):
    f0 = np.asarray(m.compute_eas_j2_contributions_jax(jnp.asarray(coords), jnp.asarray(d), jnp.zeros(4), jnp.asarray(S0), LAM,MU,SY0,H)[0])
    out=[]
    for th in [0.1,0.5,1.0,np.pi/2]:
        T,R = T8R(th)
        cr = coords@R.T; dr=(d.reshape(4,2)@R.T).reshape(8)
        fr = np.asarray(m.compute_eas_j2_contributions_jax(jnp.asarray(cr), jnp.asarray(dr), jnp.zeros(4), jnp.asarray(S0), LAM,MU,SY0,H)[0])
        out.append(np.linalg.norm(fr-T@f0)/np.linalg.norm(f0))
    print(f"{label:22s} " + "  ".join(f"{e:.3e}" for e in out))

print(f"{'variant':22s} " + "  ".join(f"{np.rad2deg(t):9.1f}d" for t in [0.1,0.5,1.0,np.pi/2]))
isotropy("repo  D@J0^-1")
m._enhanced_grad_modes = fixed_modes
jax.clear_caches()
isotropy("FIX   D@J0^-T")

# bending accuracy at AR sweep, both variants, with a ROTATED reference
def bending_ratio(ref_rot_deg):
    kappa=1e-4; h=0.15; w=1.0; Eps=E/(1-NU**2)
    th=np.deg2rad(ref_rot_deg); c,s=np.cos(th),np.sin(th); R=np.array([[c,-s],[s,c]])
    base = np.array([[0.,0.],[w,0.],[w,h],[0.,h]])
    cc = base@R.T
    cx = base[:,0]-w/2; cy = base[:,1]-h/2
    u = np.zeros(8); u[0::2]=kappa*cx*cy; u[1::2]=-0.5*kappa*cx**2
    ur = (u.reshape(4,2)@R.T).reshape(8)
    f,K,_,_,_ = m.compute_eas_j2_contributions_jax(jnp.asarray(cc), jnp.asarray(ur), jnp.zeros(4), jnp.asarray(S0), LAM,MU,SY0,H)
    U = 0.5*float(np.asarray(ur) @ np.asarray(K) @ np.asarray(ur))
    U_ex = 0.5*Eps*(h**3/12)*kappa**2*w
    return U/U_ex

print()
print(f"{'ref rotation':22s} " + "  ".join(f"{a:8.0f}d" for a in [0,10,30,45,60,90]))
m._enhanced_grad_modes = _orig; jax.clear_caches()
print(f"{'repo  D@J0^-1':22s} " + "  ".join(f"{bending_ratio(a):9.3f}" for a in [0,10,30,45,60,90]))
m._enhanced_grad_modes = fixed_modes; jax.clear_caches()
print(f"{'FIX   D@J0^-T':22s} " + "  ".join(f"{bending_ratio(a):9.3f}" for a in [0,10,30,45,60,90]))
