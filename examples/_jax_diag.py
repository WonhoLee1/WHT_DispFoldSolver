"""JAX diagnostic: check x64, EAS element, and find errors."""
import os
os.environ['JAX_ENABLE_X64'] = 'True'  # must be before import jax

import jax
import jax.numpy as jnp
import numpy as np

print("=== JAX Config ===")
print(f"x64 enabled: {jax.config.read('jax_enable_x64')}")
print(f"JAX version: {jax.__version__}")

# Check float64 actually works
arr = jnp.ones(3, dtype=jnp.float64)
print(f"jnp.float64 array dtype: {arr.dtype}")
assert arr.dtype == jnp.float64, "float64 not available!"

# Test EAS element
print("\n=== EAS Element Test ===")
from dispsolver.element.q4_eas_jax import compute_eas_j2_contributions_jax

coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
u = np.zeros(8, dtype=np.float64)
alpha = np.zeros(4, dtype=np.float64)
state = np.zeros((4, 5), dtype=np.float64)
lam, mu = 769.23, 1538.46  # E=4000, nu=0.3
sigma_y0, H = 80.0, 620.0

f, K, alpha_new, state_new = compute_eas_j2_contributions_jax(
    coords, u, alpha, state, lam, mu, sigma_y0, H
)
print(f"f dtype: {f.dtype}, K dtype: {K.dtype}")
print(f"f norm: {np.linalg.norm(np.asarray(f)):.6e}")
print(f"K has NaN: {np.any(np.isnan(np.asarray(K)))}")
print(f"K sym err: {np.max(np.abs(np.asarray(K) - np.asarray(K).T)):.6e}")

# Progressive deformation test
print("\n=== Progressive Deformation ===")
for scale in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
    u2 = np.array([scale*0.01, 0.0, scale*0.01, 0.0, 
                   scale*0.02, scale*0.01, scale*0.02, scale*0.01], dtype=np.float64)
    f2, K2, a2, s2 = compute_eas_j2_contributions_jax(
        coords, u2, alpha_new, state_new, lam, mu, sigma_y0, H
    )
    f2_np = np.asarray(f2)
    K2_np = np.asarray(K2)
    has_nan = np.any(np.isnan(f2_np)) or np.any(np.isnan(K2_np))
    print(f"  scale={scale:.1f}: f_norm={np.linalg.norm(f2_np):.4e}  "
          f"NaN={'YES' if has_nan else 'no'}  "
          f"K_diag=[{K2_np[0,0]:.1f},{K2_np[1,1]:.1f},{K2_np[2,2]:.1f},{K2_np[3,3]:.1f}]")

# Test J2 plasticity stress with large deformations
print("\n=== J2 Plasticity Test ===")
from dispsolver.material.plastic_jax import pk2_voigt_jax, tangent_voigt_jax

# Severe shear test
for gamma in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]:
    F = np.array([[1.0, gamma], [0.0, 1.0]], dtype=np.float64)
    S, s_new = pk2_voigt_jax(F, state[0], lam, mu, sigma_y0, H)
    S_np = np.asarray(S)
    print(f"  gamma={gamma:.2f}: S=[{S_np[0]:.2f},{S_np[1]:.2f},{S_np[2]:.2f}]  "
          f"NaN={'YES' if np.any(np.isnan(S_np)) else 'no'}")

# Tangent FD consistency check
print("\n=== Tangent FD Check ===")
for gamma in [0.0, 0.1, 0.5]:
    F = np.array([[1.0, gamma], [0.0, 1.0]], dtype=np.float64)
    S, C, _ = tangent_voigt_jax(F, state[0], lam, mu, sigma_y0, H)
    S_np = np.asarray(S)
    C_np = np.asarray(C)
    has_nan = np.any(np.isnan(C_np))
    print(f"  gamma={gamma:.2f}: C NaN={'YES' if has_nan else 'no'}  "
          f"C_diag=[{C_np[0,0]:.2f},{C_np[1,1]:.2f},{C_np[2,2]:.2f}]")

print("\n=== DONE ===")
