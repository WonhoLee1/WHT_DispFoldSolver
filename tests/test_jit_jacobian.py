import jax
import jax.numpy as jnp
from dispsolver.element.q4_corotational_jax import compute_element_rotation, build_block_rotation
import time

def _f_global_kinematics(u_g, coords, f_l_fixed, K_l_fixed):
    c_curr = coords + u_g.reshape((4, 2))
    R = compute_element_rotation(coords, c_curr)
    T = build_block_rotation(R)
    u_l = (c_curr @ R - coords).flatten()
    
    u_l_current = jax.lax.stop_gradient(u_l)
    f_l_approx = f_l_fixed + K_l_fixed @ (u_l - u_l_current)
    
    return T @ f_l_approx

_K_global_fn = jax.jacobian(_f_global_kinematics, argnums=0)

@jax.jit
def compute_element_tangent(coords, u_elem, f_local, K_local):
    return _K_global_fn(u_elem, coords, f_local, K_local)

def test_jit_jacobian():
    coords = jnp.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    u_elem = jnp.array([0.1, 0.2, -0.1, 0.1, 0.05, -0.05, 0.0, 0.1])
    f_l = jnp.ones(8)
    K_l = jnp.eye(8)
    
    # First call (compilation)
    t0 = time.time()
    K_g = compute_element_tangent(coords, u_elem, f_l, K_l)
    t1 = time.time()
    print("First call (compile):", t1 - t0)
    
    # Second call (execution)
    t0 = time.time()
    K_g = compute_element_tangent(coords, u_elem, f_l, K_l)
    t1 = time.time()
    print("Second call (execute):", t1 - t0)
    print("Jacobian shape:", K_g.shape)

test_jit_jacobian()
