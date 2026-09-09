"""
tests/test_cpe4rh_patch.py
===========================
Theoretical-correctness verification for the CPE4RH (1-point reduced
integration + Flanagan-Belytschko hourglass + element-constant hybrid
pressure) viscoelastic element, `dispsolver/element/q4_visco_hybrid_reduced_jax.py`.

Three properties, independent of whether this element is ever selected for
a real folding mesh in this repo (see that file's module docstring):

1. Rigid-rotation canary: a pure rigid rotation must produce exactly zero
   internal force (deviatoric PK2, hybrid pressure, AND hourglass force
   all vanish for a rotation -- rotation is an affine field with F=R,
   J=1).
2. Hourglass-insensitivity under any affine field: for a homogeneous
   (non-rotation) deformation gradient, the total force must be
   INDEPENDENT of the hourglass stabilization coefficient `alpha_hg` --
   the orthogonalized hourglass generalized coordinate is exactly zero
   for any affine nodal field by construction, so scaling the
   stabilization strength by 6 orders of magnitude must not move the
   force at all.
3. Multi-element patch test (MacNeal & Harder 1985 style): 4 irregular
   quads sharing one interior node, boundary nodes driven by an affine
   field u=(F0-I)@X for a homogeneous F0, interior node's displacement
   set to the SAME affine formula (not solved for) -- the assembled
   out-of-balance force at the interior node must be exactly zero, i.e.
   the affine field is already an equilibrium solution of the patch.
   This is the standard patch test criterion and is what actually proves
   the element (kinematics + B-operator + hourglass orthogonalization)
   is consistent, not just that a single element behaves reasonably in
   isolation.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from dispsolver.element.q4_visco_hybrid_reduced_jax import (
    compute_single_reduced_hybrid_jax,
)

# Shared PSA-like material (Arruda-Boyce ground state, no Prony terms --
# M=0 keeps the test focused on the element kinematics/hourglass, not the
# overstress recurrence, which is already covered by q4_visco_simo_fs_jax's
# own tests).
_KAPPA = 8.3333
_BPARAMS = jnp.array([0.015614, 3.0])   # mu, lambda_m
_G_I = jnp.zeros(0)
_TAU_I = jnp.zeros(0)
_G_INF = 1.0
_DT = 1.0
_BASE = "arruda"
_EYE2 = jnp.eye(2, dtype=jnp.float64)
_F_N_ID = jnp.stack([_EYE2, _EYE2, _EYE2, _EYE2])


def _state_zeros():
    # M=0 -> state layout length 6*(M+1) = 6, one slot (S_iso_prev) only.
    return jnp.zeros((4, 6), dtype=jnp.float64)


def _affine_u(coords, F0):
    """Nodal displacement vector (8,) for the affine field u = (F0-I) @ X."""
    H = F0 - jnp.eye(2, dtype=jnp.float64)
    u = jnp.zeros(8, dtype=jnp.float64)
    for a in range(4):
        d = H @ coords[a]
        u = u.at[2 * a].set(d[0])
        u = u.at[2 * a + 1].set(d[1])
    return u


def _call(coords, u_elem, alpha_hg=None):
    kwargs = {}
    if alpha_hg is not None:
        kwargs["alpha_hg"] = alpha_hg
    return compute_single_reduced_hybrid_jax(
        _BASE, coords, u_elem, _state_zeros(), _KAPPA, _BPARAMS,
        _G_I, _TAU_I, _G_INF, _DT, 1.0, _F_N_ID, 0.0,
        **kwargs,
    )


_UNIT_SQUARE = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


# ---------------------------------------------------------------------
# 1. Rigid-rotation canary
# ---------------------------------------------------------------------
@pytest.mark.parametrize("theta_deg", [0.001, 1.0, 30.0, 45.0, 89.9, 90.0, 135.0])
def test_rigid_rotation_zero_force(theta_deg):
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    R = jnp.array([[c, -s], [s, c]], dtype=jnp.float64)
    u_elem = _affine_u(_UNIT_SQUARE, R)

    f_e, K_e, state_new, F_n_new = _call(_UNIT_SQUARE, u_elem)

    assert np.all(np.isfinite(np.asarray(f_e)))
    assert np.max(np.abs(np.asarray(f_e))) < 1e-9, (
        f"rigid rotation at {theta_deg} deg produced nonzero force: "
        f"max|f_e|={np.max(np.abs(np.asarray(f_e))):.3e}"
    )


def test_rigid_rotation_zero_force_off_center():
    """Same canary on a non-unit, non-axis-aligned, non-origin-centred
    quad -- the orthogonalization must not depend on the element sitting
    at a convenient location."""
    coords = jnp.array([[3.1, -0.4], [4.6, -0.1], [4.4, 1.3], [3.0, 1.1]])
    th = np.deg2rad(37.0)
    c, s = np.cos(th), np.sin(th)
    R = jnp.array([[c, -s], [s, c]], dtype=jnp.float64)
    u_elem = _affine_u(coords, R)
    f_e, K_e, _, _ = _call(coords, u_elem)
    assert np.max(np.abs(np.asarray(f_e))) < 1e-9


# ---------------------------------------------------------------------
# 2. Hourglass-insensitivity under an affine (non-rotation) field
# ---------------------------------------------------------------------
@pytest.mark.parametrize("F0", [
    jnp.array([[1.2, 0.0], [0.0, 0.9]]),      # uniaxial-ish stretch
    jnp.array([[1.1, 0.0], [0.0, 1.1]]),      # equibiaxial stretch
    jnp.array([[1.0, 0.15], [0.0, 1.0]]),     # pure shear
    jnp.array([[0.95, 0.08], [0.05, 1.08]]),  # general homogeneous F
])
def test_hourglass_insensitive_to_affine_field(F0):
    u_elem = _affine_u(_UNIT_SQUARE, F0)
    f_lo, K_lo, _, _ = _call(_UNIT_SQUARE, u_elem, alpha_hg=0.0)
    f_hi, K_hi, _, _ = _call(_UNIT_SQUARE, u_elem, alpha_hg=50.0)  # 1000x default

    f_lo = np.asarray(f_lo)
    f_hi = np.asarray(f_hi)
    scale = max(np.max(np.abs(f_lo)), 1e-12)
    assert np.max(np.abs(f_hi - f_lo)) / scale < 1e-8, (
        "hourglass stabilization changed the force under a purely affine "
        "field -- the orthogonalized hourglass coordinate should be "
        "exactly zero here regardless of alpha_hg"
    )


# ---------------------------------------------------------------------
# 3. Multi-element (irregular quad) patch test
# ---------------------------------------------------------------------
def _patch_nodes():
    """3x3 grid, node 4 (centre) perturbed off-centre for irregularity.

    Layout (node indices):
        6  7  8
        3  4  5
        0  1  2
    """
    nodes = jnp.array([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        [0.0, 1.0], [1.3, 0.9], [2.0, 1.0],
        [0.0, 2.0], [1.0, 2.0], [2.0, 2.0],
    ])
    quads = [
        (0, 1, 4, 3),
        (1, 2, 5, 4),
        (4, 5, 8, 7),
        (3, 4, 7, 6),
    ]
    return nodes, quads


@pytest.mark.parametrize("F0", [
    jnp.array([[1.2, 0.0], [0.0, 0.9]]),
    jnp.array([[1.1, 0.0], [0.0, 1.1]]),
    jnp.array([[1.0, 0.15], [0.0, 1.0]]),
    jnp.array([[0.95, 0.08], [0.05, 1.08]]),
])
def test_patch_test_affine_field_is_equilibrium(F0):
    """The affine field u=(F0-I)@X, imposed at every node (boundary AND
    the free interior node alike), must leave zero out-of-balance force
    at the interior node once all 4 surrounding elements' contributions
    are summed -- the standard patch test criterion."""
    nodes, quads = _patch_nodes()
    interior_node = 4

    u_global = jnp.stack([(F0 - jnp.eye(2)) @ nodes[n] for n in range(nodes.shape[0])])

    resid = np.zeros(2)
    for quad in quads:
        coords = jnp.stack([nodes[n] for n in quad])
        u_elem = jnp.concatenate([u_global[n] for n in quad])
        f_e, _, _, _ = _call(coords, u_elem)
        f_e = np.asarray(f_e)
        if interior_node in quad:
            local = quad.index(interior_node)
            resid += f_e[2 * local:2 * local + 2]

    assert np.all(np.isfinite(resid))
    assert np.max(np.abs(resid)) < 1e-8, (
        f"patch test failed for F0={np.asarray(F0)}: assembled residual "
        f"at the interior node is {resid}, expected ~0"
    )


def test_patch_test_rigid_rotation_is_equilibrium():
    nodes, quads = _patch_nodes()
    interior_node = 4
    th = np.deg2rad(52.0)
    c, s = np.cos(th), np.sin(th)
    R = jnp.array([[c, -s], [s, c]], dtype=jnp.float64)

    u_global = jnp.stack([(R - jnp.eye(2)) @ nodes[n] for n in range(nodes.shape[0])])

    resid = np.zeros(2)
    for quad in quads:
        coords = jnp.stack([nodes[n] for n in quad])
        u_elem = jnp.concatenate([u_global[n] for n in quad])
        f_e, _, _, _ = _call(coords, u_elem)
        f_e = np.asarray(f_e)
        if interior_node in quad:
            local = quad.index(interior_node)
            resid += f_e[2 * local:2 * local + 2]

    assert np.max(np.abs(resid)) < 1e-8


# ---------------------------------------------------------------------
# 4. Tangent sanity: K_e symmetric part matches FD of f_e (autodiff vs FD
#    cross-check, independent of the analytic derivation in the module).
# ---------------------------------------------------------------------
def test_tangent_matches_finite_difference():
    F0 = jnp.array([[1.05, 0.03], [0.02, 0.97]])
    u_elem = _affine_u(_UNIT_SQUARE, F0)
    f0, K_e, _, _ = _call(_UNIT_SQUARE, u_elem)
    f0 = np.asarray(f0)
    K_e = np.asarray(K_e)

    h = 1e-6
    K_fd = np.zeros((8, 8))
    for j in range(8):
        up = np.asarray(u_elem).copy()
        up[j] += h
        fp, _, _, _ = _call(_UNIT_SQUARE, jnp.array(up))
        K_fd[:, j] = (np.asarray(fp) - f0) / h

    denom = max(np.max(np.abs(K_e)), 1e-8)
    err = np.max(np.abs(K_fd - K_e)) / denom
    assert err < 1e-3, f"analytic (autodiff) tangent vs FD relative error {err:.3e}"


# ---------------------------------------------------------------------
# 5. Orthodox solver-level patch test: this repo's OWN standard irregular
#    patch mesh (verification/mesh_utils.py::build_patch_mesh_irregular,
#    a MacNeal & Harder 1985 style 9-node/4-quad patch with a single free
#    interior node), run through the REAL DynamicSolver end to end --
#    not the isolated element kernel above. This is the same mesh and
#    methodology `verification/benchmarks.py::patch_test_solver` already
#    uses to certify every other element in this codebase, so CPE4RH is
#    held to the identical, already-established standard rather than a
#    bespoke one invented just for this file.
# ---------------------------------------------------------------------
def test_orthodox_solver_level_patch_test():
    """Boundary nodes driven by the affine field u=(F0-I)@X; the ONE free
    interior node (node 5) must settle at the exact same affine value --
    solved by a real Newton iteration through DynamicSolver, not asserted
    algebraically. alpha=1e-3 matches this repo's own patch-test default
    (verification/benchmarks.py's patch_test_element/patch_test_solver)."""
    from dispsolver.mesh import Mesh
    from dispsolver.material import ArrudaBoyce, ViscoelasticMaterial
    from dispsolver.solver import DynamicSolver

    E, nu, alpha = 1000.0, 0.3, 1e-3
    coords = {
        1: (0.0, 0.0),  2: (0.3, 0.0),  3: (1.0, 0.0),
        4: (0.0, 0.4),  5: (0.5, 0.5),  6: (1.0, 0.6),
        7: (0.0, 1.0),  8: (0.6, 1.0),  9: (1.0, 1.0),
    }
    mesh = Mesh()
    for nid, (x, y) in coords.items():
        mesh.add_node(nid, x, y)
    mesh.add_element(1, [1, 2, 5, 4], "CPE4RH")
    mesh.add_element(2, [2, 3, 6, 5], "CPE4RH")
    mesh.add_element(3, [4, 5, 8, 7], "CPE4RH")
    mesh.add_element(4, [5, 6, 9, 8], "CPE4RH")

    boundary_nodes = [1, 2, 3, 6, 9, 8, 7, 4]
    internal_nid = 5

    # E/nu -> Arruda-Boyce (mu, K) so the small-alpha response matches
    # linear elasticity to O(alpha) -- G = E/(2(1+nu)), K = E/(3(1-2nu)),
    # mu = G / beta(lambda_m) as this codebase's own PSA config derives
    # it (AGENTS.md: "mu PARAMETER is NOT the initial shear modulus").
    lambda_m = 3.0
    beta = 1.0 + 3.0 / (5.0 * lambda_m**2) + 99.0 / (175.0 * lambda_m**4) \
        + 513.0 / (875.0 * lambda_m**6) + 42039.0 / (67375.0 * lambda_m**8)
    G = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))
    mu = G / beta
    base_mat = ArrudaBoyce()
    visco_mat = ViscoelasticMaterial(base_mat, g_i=[0.0], tau_i=[1.0])
    params = {"mu": mu, "lambda_m": lambda_m, "K": K}

    # Same tight tolerance recipe + mode='quasistatic' verification/
    # benchmarks.py's own make_solver() uses for every patch test in this
    # repo -- omitting mode='quasistatic' silently leaves inertial terms
    # in play, which is not what a patch test (a pure static-equilibrium
    # check) means to exercise.
    solver = DynamicSolver(
        mesh, visco_mat, rho=1000.0, material_params=params,
        element_type="CPE4RH", max_iter=50, tol=1e-8, atol=1e-9,
        rtol=1e-6, mode="quasistatic",
    )

    bc_dofs = []
    bc_vals = []
    for nid in boundary_nodes:
        node = mesh.get_node(nid)
        idx = int(np.where(solver.sorted_nids == nid)[0][0])
        bc_dofs += [idx * 2, idx * 2 + 1]
        bc_vals += [alpha * node.x, -nu / (1.0 - nu) * alpha * node.y]
    solver.set_prescribed_dofs(bc_dofs, bc_vals)

    conv = solver.solve_step(dt=1.0)
    assert conv > 0, f"orthodox patch test failed to converge (code {conv})"

    node5 = mesh.get_node(internal_nid)
    idx5 = int(np.where(solver.sorted_nids == internal_nid)[0][0])
    ux_exact = alpha * node5.x
    uy_exact = -nu / (1.0 - nu) * alpha * node5.y
    ux_num = solver.u[idx5 * 2]
    uy_num = solver.u[idx5 * 2 + 1]
    err = np.sqrt((ux_num - ux_exact) ** 2 + (uy_num - uy_exact) ** 2)
    err_pct = err / (abs(alpha) + 1e-30) * 100.0

    assert err_pct < 0.1, (
        f"CPE4RH orthodox patch test: internal node 5 displacement error "
        f"{err_pct:.4f}% (exact ux={ux_exact:.6e}, uy={uy_exact:.6e}; "
        f"numerical ux={ux_num:.6e}, uy={uy_num:.6e})"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
