"""
Family B step 1: hard data before switching PET/GLASS off the invented SRI
device onto the real Abaqus CPE4I (`Q4_EAS` + J2Plasticity).

Two measurements, both single-element, both at the production aspect
ratio range:

  T1 (objectivity, opus review's corrected canary): apply a fixed
      inhomogeneous nodal perturbation (real strain), THEN rigidly rotate
      the whole configuration by theta. An objective element satisfies
      f(theta) == T8(theta) @ f(0) exactly.

  A2 (bending accuracy vs aspect ratio): pure-bending nodal pattern,
      measure the element's bending energy vs the analytically-exact
      value for the same curvature. AGENTS.md 4.1 documents co-rotational
      as 1 + 0.35*AR^2 artificially stiff, EAS as AR-independent -- this
      re-measures both directly rather than trusting the note.
"""

import numpy as np
import jax.numpy as jnp

from dispsolver.material.plastic import J2Plasticity
from dispsolver.element.q4_sri_jax import compute_corotational_sri_j2_contributions_jax
from dispsolver.element.q4_eas_jax import compute_eas_j2_contributions_jax

# Elastic-only J2 (huge yield stress) so this measures the ELEMENT, not
# the plasticity -- matching how verification/element_backends.py's own
# make_solver keeps J2 elastic for element benchmarks.
E, NU, SY0, H = 4000.0, 0.3, 1e12, 0.0
_MAT = J2Plasticity(E=E, nu=NU, sigma_y0=SY0, H=H)
LAM, MU = _MAT.lam, _MAT.mu

_STATE0 = np.zeros((4, 5))
_STATE0[:, 0] = 1.0   # F_p_inv = identity
_STATE0[:, 3] = 1.0
_ALPHA0 = np.zeros(4)


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _T8(theta):
    R = _rot(theta)
    T = np.zeros((8, 8))
    for i in range(4):
        T[2 * i:2 * i + 2, 2 * i:2 * i + 2] = R
    return T


def _f_sri(coords, u):
    f, K, _, _ = compute_corotational_sri_j2_contributions_jax(
        jnp.asarray(coords), jnp.asarray(u), jnp.asarray(_STATE0),
        LAM, MU, SY0, H, 1.0)
    return np.asarray(f), np.asarray(K)


def _f_eas(coords, u):
    f, K, _, _, _ = compute_eas_j2_contributions_jax(
        jnp.asarray(coords), jnp.asarray(u), jnp.asarray(_ALPHA0),
        jnp.asarray(_STATE0), LAM, MU, SY0, H)
    return np.asarray(f), np.asarray(K)


KERNELS = [("Q4_COROTATIONAL_SRI", _f_sri), ("Q4_EAS (CPE4I)", _f_eas)]


# ---------------------------------------------------------------------
# T1: strain-then-rotate objectivity
# ---------------------------------------------------------------------
def t1_objectivity(width=1.0, height=0.15):
    """Real free-span aspect ratio (AR = 1.0/0.15 ~ 6.7 in the ex12 mesh's
    PET rows). Perturbation is inhomogeneous (NOT a homogeneous strain) so
    Gauss-point and centroid gradients differ -- the whole point, since a
    homogeneous state makes SRI's mixed sampling inert and hides the
    defect (opus review's finding)."""
    coords = np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]])
    rng = np.random.default_rng(0)
    d = rng.uniform(-0.004, 0.004, size=8) * width   # ~0.4% strain

    print(f"\n=== T1 objectivity (strain-then-rotate), element {width} x {height} "
          f"(AR={width/height:.1f}) ===")
    print(f"{'element':24s} " + "  ".join(f"{np.rad2deg(t):7.1f}deg" for t in
                                          [0.1, 0.5, 1.0, np.pi / 2]))
    for name, fn in KERNELS:
        f0, _ = fn(coords, d)
        errs = []
        for theta in [0.1, 0.5, 1.0, np.pi / 2]:
            R = _rot(theta)
            coords_rot = coords @ R.T
            # rotate BOTH the reference geometry and the deformation
            d_rot = (d.reshape(4, 2) @ R.T).reshape(8)
            f_th, _ = fn(coords_rot, d_rot)
            err = np.linalg.norm(f_th - _T8(theta) @ f0) / max(np.linalg.norm(f0), 1e-30)
            errs.append(err)
        print(f"{name:24s} " + "  ".join(f"{e:10.2e}" for e in errs))


# ---------------------------------------------------------------------
# A2: bending stiffness vs aspect ratio
# ---------------------------------------------------------------------
def a2_bending_vs_ar(ars=(1.0, 2.0, 5.0, 7.5, 10.0, 15.0)):
    """Pure-bending nodal pattern u_x = kappa*x*y, u_y = -kappa*x^2/2.
    Compare each element's stored bending energy against the plane-strain
    analytic value for the same curvature; ratio 1.0 = no artificial
    stiffness, >1 = locking."""
    print("\n=== A2 bending stiffness ratio vs aspect ratio "
          "(1.00 = exact, >1 = artificially stiff / locking) ===")
    print(f"{'AR':>6s}  {'Q4_COROTATIONAL_SRI':>22s}  {'Q4_EAS (CPE4I)':>18s}")
    kappa = 1e-4
    h = 0.15
    E_ps = E / (1.0 - NU ** 2)          # plane-strain bending modulus
    for ar in ars:
        w = ar * h
        coords = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
        cx = coords[:, 0] - w / 2.0     # centre the bending about the element
        cy = coords[:, 1] - h / 2.0
        u = np.zeros(8)
        u[0::2] = kappa * cx * cy
        u[1::2] = -0.5 * kappa * cx ** 2
        # analytic strain energy of a plane-strain beam segment in pure
        # bending: U = 0.5 * E' * I * kappa^2 * L,  I = h^3/12 (unit depth)
        U_exact = 0.5 * E_ps * (h ** 3 / 12.0) * kappa ** 2 * w
        row = [f"{ar:6.1f}"]
        for name, fn in KERNELS:
            f, K = fn(coords, u)
            U_num = 0.5 * float(u @ K @ u)
            row.append(f"{U_num / U_exact:22.3f}" if name.startswith("Q4_CORO")
                       else f"{U_num / U_exact:18.3f}")
        print("  ".join(row))


if __name__ == "__main__":
    t1_objectivity()
    a2_bending_vs_ar()
