import jax
import jax.numpy as jnp
from dispsolver.element.q4_corotational_jax import compute_element_rotation, build_block_rotation

def test_jacobian_trick():
    coords = jnp.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    u_elem = jnp.array([0.1, 0.2, -0.1, 0.1, 0.05, -0.05, 0.0, 0.1])
    
    f_l_fixed = jnp.ones(8)
    K_l_fixed = jnp.eye(8)
    
    def f_global_fn(u_g, f_l, K_l):
        c_curr = coords + u_g.reshape((4, 2))
        R = compute_element_rotation(coords, c_curr)
        T = build_block_rotation(R)
        u_l = (c_curr @ R - coords).flatten()
        
        u_l_current = jax.lax.stop_gradient(u_l)
        f_l_approx = f_l + K_l @ (u_l - u_l_current)
        
        return T @ f_l_approx
        
    K_global = jax.jacobian(lambda ug: f_global_fn(ug, f_l_fixed, K_l_fixed))(u_elem)
    print("Jacobian shape:", K_global.shape)
    print("Jacobian trace:", jnp.trace(K_global))

test_jacobian_trick()
print("Success!")
