"""is_numerically_symmetric: the RBE2-condensation path's Cholesky-vs-LU decision.

Regression for 2026-10-09: the path assumed symmetry, then a first fix normalised by the
global max entry -- which a stiff plate/penalty diagonal dominates -- and still sent the
F-bar PSA tangent (asymmetric ~1e-5 relative to its own rows) to Cholesky.
"""
import numpy as np
import scipy.sparse as sps

from dispsolver.solver.dynamic import is_numerically_symmetric


def test_exactly_symmetric():
    rng = np.random.default_rng(1)
    B = rng.normal(size=(6, 6))
    assert is_numerically_symmetric(sps.csr_matrix(B + B.T + 10 * np.eye(6)))


def test_soft_block_asymmetry_is_seen_beside_a_stiff_block():
    # stiff "plate" block 1e10, soft "adhesive" block ~1e-2 with a 1e-5 relative asymmetry
    K = np.zeros((4, 4))
    K[:2, :2] = 1e10 * np.array([[2.0, -1.0], [-1.0, 2.0]])
    K[2:, 2:] = 1e-2 * np.array([[2.0, -1.0], [-1.0 + 1e-5, 2.0]])
    Ks = sps.csr_matrix(K)
    global_ratio = abs(Ks - Ks.T).max() / abs(Ks).max()
    assert global_ratio < 1e-10                       # what the first attempt measured: "symmetric"
    assert not is_numerically_symmetric(Ks)          # entry-wise: it is not


def test_roundoff_level_asymmetry_counts_as_symmetric():
    K = np.array([[4.0, 1.0], [1.0 + 1e-14, 3.0]])
    assert is_numerically_symmetric(sps.csr_matrix(K))
