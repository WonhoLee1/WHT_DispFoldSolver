import numpy as np
import jax
import jax.numpy as jnp
from dispsolver.element.q4_up_jax import compute_hybrid_element_contributions, compute_element_energy

def test_hybrid_element_symmetry():
    """Verify that the statically condensed hybrid tangent stiffness matrix is symmetric."""
    coords = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=jnp.float64)
    
    # Arbitrary non-zero deformation state
    u_elem = jnp.array([0.05, -0.02, 0.12, 0.04, -0.03, 0.08, 0.01, -0.05], dtype=jnp.float64)
    params = {'E': 1000.0, 'nu': 0.499}  # Nearly incompressible
    
    f_e, K_e = compute_hybrid_element_contributions(coords, u_elem, params)
    
    # Convert to NumPy for assertions
    f_e_np = np.asarray(f_e)
    K_e_np = np.asarray(K_e)
    
    assert f_e_np.shape == (8,)
    assert K_e_np.shape == (8, 8)
    
    # Symmetry check
    np.testing.assert_allclose(K_e_np, K_e_np.T, rtol=1e-12, atol=1e-12)

def test_hybrid_nearly_incompressible():
    """Compare energy and contributions under large volumetric versus deviatoric strain."""
    coords = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=jnp.float64)
    
    params = {'mu': 100.0, 'lambda': 1e6}  # High bulk modulus (~1e6)
    
    # 1. Pure shear/deviatoric strain
    u_shear = jnp.array([0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    E_shear = compute_element_energy(coords, u_shear, params)
    
    # 2. Volumetric/dilatational strain
    u_vol = jnp.array([-0.05, -0.05, 0.05, -0.05, 0.05, 0.05, -0.05, 0.05], dtype=jnp.float64)
    E_vol = compute_element_energy(coords, u_vol, params)
    
    # Since lambda is high, volumetric energy should penalize heavily compared to shear energy
    assert E_vol > E_shear * 10.0
