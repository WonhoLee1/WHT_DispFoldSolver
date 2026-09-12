"""
tests/element_contract/test_material_tangent.py
================================================
C8, applied one layer down: **a material's tangent must be the derivative
of that material's own stress.**

The element contract suite found `Q4`'s element tangent failing C6
(bending stiffness swinging 69.3 -> 20.9 -> 69.3 as the element is rotated
0-45-90 degrees) while its internal force passed C7 at 3.6e-14. Force
isotropic, stiffness not, *at `u = 0`* -- where there is no stress, no
geometric stiffness and no material nonlinearity -- localises the problem
to the material tangent, and the finite-difference check below confirms it.

The defect
----------
`J2Plasticity._elastic_tangent_voigt` (`dispsolver/material/plastic.py`)
reduces the 6x6 Voigt tangent to plane-strain 3x3 by slicing the first
three rows and columns:

    result[a, b] = C_v[a, b]      for a, b in range(3)

But `_VOIGT_PAIRS = [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]`, so 6-Voigt
index **2 is the out-of-plane (3,3) normal component** and the in-plane
**shear (1,2) is at index 3**. The slice therefore puts `C_3333` where
`C_1212` belongs and `C_1133` where `C_1112` belongs, and then doubles the
latter with an "engineering shear" factor that does not apply to it. Both
wrong numbers are reproduced exactly by that reading:

    coded  C[2,2] = 5384.6 = lambda + 2*mu   (= C_3333)
    true   C[2,2] = 1538.5 = mu              (= C_1212)
    coded  C[0,2] = 4615.4 = 2*lambda        (= 2*C_1133)
    true   C[0,2] = 0                        (= C_1112, zero when isotropic)

`_plastic_tangent_voigt`, immediately below it in the same file, extracts
the same block **correctly** (`C3[2,2] = C_mat[3,3]`, and no doubling), so
the two branches of the same material disagree with each other.

Scope
-----
Confined to the NumPy `J2Plasticity` tangent, i.e. the `Q4` + `J2Plasticity`
assembly paths (`_assemble_j2_batch` and the J2 batch branch of
`_assemble_multi_material_batch`). The JAX J2 elements
(`Q4_EAS`, `Q4_COROTATIONAL*`, `Q4_SRI`) compute their own tangent through
`plastic_jax.tangent_voigt_jax` and are unaffected -- they measure 1.5e-09
on the element-level C8. A wrong tangent does not corrupt the converged
answer (the residual is still right, which is why C7 passes and why the
patch test C5 passes at 3.7e-16); it costs Newton iterations, and can stop
convergence entirely. `tests/test_solver.py`'s five standing
"NR should converge, got -40" failures are consistent with that.

**Not fixed here.** Stage 0 of the element refactor changes no formulation
and no material; this is a characterization test that records the defect
with its cause and an oracle, so that the fix -- when it is scheduled -- has
something to turn green.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

from dispsolver.material.plastic import J2Plasticity

MAT = J2Plasticity(E=4000.0, nu=0.3, sigma_y0=1e12, H=0.0)   # elastic range
LAM, MU = MAT.lam, MAT.mu
_STATE1 = np.tile(np.array([1.0, 0.0, 0.0, 1.0, 0.0]), (1, 1))   # F_p_inv = I


def _S(F):
    return np.asarray(MAT.pk2_voigt_batch(F[None], {}, _STATE1)[0])[0]


def _C(F):
    return np.asarray(MAT.pk2_voigt_batch(F[None], {}, _STATE1)[1])[0]


def _F_from_E(E):
    """The symmetric `F = U = sqrt(I + 2E)` realising Green-Lagrange `E`."""
    w, V = np.linalg.eigh(np.eye(2) + 2.0 * E)
    return V @ np.diag(np.sqrt(w)) @ V.T


def _fd_tangent(F, h=1e-6):
    """Central-difference `dS_voigt / dE_voigt`, ENGINEERING shear.

    With `gamma = 2*E12`, `S11 = ... + C_1112*gamma`, so the gamma column is
    the derivative along `E12 = gamma/2` -- hence the 0.5 in the third basis
    direction, and no extra factor anywhere.
    """
    E0 = 0.5 * (F.T @ F - np.eye(2))
    basis = [np.array([[1.0, 0.0], [0.0, 0.0]]),
             np.array([[0.0, 0.0], [0.0, 1.0]]),
             np.array([[0.0, 0.5], [0.5, 0.0]])]
    C = np.zeros((3, 3))
    for j, dE in enumerate(basis):
        C[:, j] = (_S(_F_from_E(E0 + h * dE))
                   - _S(_F_from_E(E0 - h * dE))) / (2.0 * h)
    return C


_CASES = [
    ("identity", np.eye(2)),
    ("stretch_shear", np.array([[1.03, 0.02], [0.0, 0.98]])),
]


@pytest.mark.parametrize("tag,F", _CASES)
def test_j2_elastic_tangent_is_the_derivative_of_its_own_stress(tag, F):
    C_coded = _C(F)
    C_fd = _fd_tangent(F)
    err = np.max(np.abs(C_coded - C_fd)) / max(np.max(np.abs(C_fd)), 1e-30)
    pytest.xfail(
        f"J2Plasticity tangent is not dS/dE ({tag}): rel err {err:.3e} "
        f"(baseline 8.57e-01 / 8.36e-01) -- _elastic_tangent_voigt slices "
        f"6-Voigt index 2 (the out-of-plane 33 component) where index 3 (the "
        f"in-plane 12 shear) belongs. See this module's docstring.")
    assert err < 1e-6


def test_j2_elastic_tangent_structure_at_identity():
    """At `F = I` an isotropic tangent has a known closed form.

    Stated separately from the FD test because the failure is legible
    without any numerics: an isotropic material cannot couple normal stress
    to shear strain, so `C[0,2]` must be zero, and the shear diagonal must
    be `mu`.
    """
    C = _C(np.eye(2))
    exact = np.array([[LAM + 2 * MU, LAM, 0.0],
                      [LAM, LAM + 2 * MU, 0.0],
                      [0.0, 0.0, MU]])
    pytest.xfail(
        f"isotropic structure violated at F=I: C[0,2] = {C[0, 2]:.1f} "
        f"(must be 0; equals 2*lambda = {2 * LAM:.1f}), C[2,2] = {C[2, 2]:.1f} "
        f"(must be mu = {MU:.1f}; equals lambda+2mu = {LAM + 2 * MU:.1f})")
    assert np.max(np.abs(C - exact)) / np.max(np.abs(exact)) < 1e-12


def test_the_element_is_exonerated_by_substituting_an_isotropic_tangent():
    """The C6 failure is ENTIRELY the material tangent, not `Q4`'s kinematics.

    Rebuilds the element tangent at `u = 0` from the solver's OWN precomputed
    B-bar and weights -- so the element code is held fixed and literally the
    only thing that changes is which `C` is contracted -- and measures
    frame covariance, `|K(rotated) - T K T^T| / |K|`:

        coded J2 tangent :  4.19e-01 at 10 deg,  1.35e+00 at 45 deg
        exact isotropic  :  7.0e-16  at 10 deg,  3.5e-16  at 45 deg

    Same B-bar, same weights, same quadrature. That is a proof, not a
    trace: `Q4`'s kinematics are frame-covariant to machine precision and
    the anisotropy is imported wholesale from `_elastic_tangent_voigt`.
    """
    from tests.element_contract.harness import BY_NAME, Probe, rot2, rot8

    h = 1.0 / 7.0
    base = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, h], [0.0, h]])
    probe = Probe(BY_NAME["Q4"], base)

    def K_at(C, coords):
        probe.set_coords(coords)
        B = probe.solver._B_bar_all[0]
        w = probe.solver._weights_all[0]
        return np.einsum('gki,gkl,glj,g->ij', B,
                         np.broadcast_to(C, (4, 3, 3)), B, w)

    exact = np.array([[LAM + 2 * MU, LAM, 0.0],
                      [LAM, LAM + 2 * MU, 0.0],
                      [0.0, 0.0, MU]])
    coded = _C(np.eye(2))

    def worst(C):
        K0 = K_at(C, base)
        out = 0.0
        for deg in (10.0, 45.0):
            R, T = rot2(np.deg2rad(deg)), rot8(np.deg2rad(deg))
            K = K_at(C, base @ R.T)
            out = max(out, np.max(np.abs(K - T @ K0 @ T.T))
                      / np.max(np.abs(K0)))
        return out

    assert worst(exact) < 1e-13, (
        "the ELEMENT is not frame-covariant either -- the defect is wider "
        "than the material tangent, widen the scope claim")
    assert worst(coded) > 1e-2, (
        "the coded tangent no longer breaks frame covariance -- if it was "
        "fixed, delete this characterization module")


def test_the_two_branches_of_the_same_material_disagree():
    """`_plastic_tangent_voigt` extracts the block correctly; the elastic one
    does not. Pinned so the fix can be checked against the branch that is
    already right, rather than against a rederivation."""
    C_el = MAT._elastic_tangent_voigt.__doc__
    assert C_el is not None
    import inspect
    el_src = inspect.getsource(MAT._elastic_tangent_voigt)
    pl_src = inspect.getsource(MAT._plastic_tangent_voigt)
    assert "result[a, b] = C_v[a, b]" in el_src, (
        "the elastic branch no longer slices the first 3x3 block -- if this "
        "was the fix, delete this characterization module")
    assert "C3[2, 2] = C_mat[3, 3]" in pl_src, (
        "the plastic branch's correct extraction moved; it is the reference "
        "for fixing the elastic one")


def test_stress_itself_is_correct():
    """Guard the scope claim: only the TANGENT is wrong.

    This is why the defect is invisible in every converged result and every
    patch test -- the residual the Newton loop drives to zero is right.

    The material is finite-strain multiplicative J2 on a logarithmic strain
    measure, so it agrees with the St-Venant-Kirchhoff expression only in
    the limit: the gap is O(strain), not a constant. Checked as a CONVERGENCE
    RATE rather than against one tolerance, because a fixed tolerance here
    would either pass a genuinely wrong stress at small strain or fail a
    correct one at large strain.
    """
    rates = []
    for scale in (1e-2, 1e-3, 1e-4):
        F = np.eye(2) + scale * np.array([[2.0, 1.5], [0.0, -1.5]])
        E = 0.5 * (F.T @ F - np.eye(2))
        tr = E[0, 0] + E[1, 1]
        exact = np.array([LAM * tr + 2 * MU * E[0, 0],
                          LAM * tr + 2 * MU * E[1, 1],
                          2 * MU * E[0, 1]])
        rates.append(np.max(np.abs(_S(F) - exact)) / np.max(np.abs(exact)))
    assert rates[0] > rates[1] > rates[2], (
        f"stress does not converge to the small-strain limit: {rates} -- "
        f"the scope claim (tangent only) is wrong, widen it")
    assert rates[2] < 1e-3, (
        f"stress still off by {rates[2]:.3e} at 1e-4 strain -- not a "
        f"finite-strain correction, an error")
    # Roughly first order: each 10x smaller strain should cut the gap ~10x.
    assert 3.0 < rates[0] / rates[1] < 30.0, f"unexpected rate: {rates}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
