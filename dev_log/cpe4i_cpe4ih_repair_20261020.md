# CPE4I / CPE4IH / C3D8IH repair — 2026-10-20

Owner: the integrating session (took over from agy on the user's instruction). Branch `fix/cpe4i-cpe4ih`.

## What was reported vs what was found

The inherited test `test_slender_beam_incompressible_bending_locking_free` failed on its last assertion
(`n_locked_i == 2`, "CPE4I locks in the bilinear hourglass volumetric mode"). Measured: it never held — not at
0a858e8 (where the test was added), not now. CPE4I has ONE locked (mean-dilatation) mode on a distorted quad,
exactly like CPE4IH, and the two spectra agree to 3 digits at nu = 0.4999:

    CPE4I  eig/G: 0 0 0 0.743 1.68 2.32 2.5 1.18e4
    CPE4IH eig/G: 0 0 0 0.743 1.68 2.32 2.5 1.18e4

That is mathematics, not a bug: the enhanced modes xi, eta absorb the linear part of the dilatation, leaving only
the element-mean constraint, which the constant-pressure part of CPE4IH also enforces. So the linear tangent cannot
tell the two elements apart; the real defects were elsewhere.

## Defects found and fixed

| # | Defect | Evidence | Fix |
|---|---|---|---|
| 1 | CPE4IH evaluated a nonlinear material on `B u` only (no `M alpha`) and condensed once from alpha = 0 | plastic bending of a strip, kappa = 0.04: `converged = False`, moment 1.0232 x oracle (CPE4I: converges, 0.9854) | new `compute_cpe4ih_nonlinear_numba`: (alpha, p_lin) iterated to element equilibrium, material fed `B u + M alpha`, converged state written back. Linear elastic keeps the closed form. After: converges, 0.9854 = CPE4I |
| 2 | Singular enhanced block at plastic collapse raised inside the parallel assembly | thick cylinder above p_lim: CPE4I -> `SystemError: ... returned a result with an exception set` | `solve_small` (pivoted Gauss, never raises) in CPE4I and CPE4IH; singular -> `has_error` -> the solver cuts the increment back |
| 3 | CPE4I had 4 internal dof; Abaqus has 5 for plane strain | Theory Guide 3.2.5 (4 CPS4I, 5 CPE4I/CAX4I/CPEG4I, 13 C3D8I = 9 + 4): non-hybrid elements add bilinear volumetric terms against volumetric locking, one in 2D, four in 3D | 5th mode `(detJ0/detJ) * xi*eta * [1,1,0]`; orthogonal to constants (patch test exact); CPE4IH does not take it ("hybrid elements ... do not include the locking-prevention incompatible modes") |

Effect of the 5th mode, measured: none in the incompressible linear limit (Cook's membrane nu = 0.4999: identical;
beam tip, spectrum over taper 0..0.9: identical); +0.1 % flexibility at nu = 1/3 (Cook n = 2: 19.194 -> 19.212).
It is adopted because the Abaqus element has it, not because a benchmark here needed it. No benefit in finite-strain
incompressible plasticity was demonstrated either (thick cylinder: both variants collapse near p_lim).

## Measurements worth keeping

Cook's membrane, plane strain, E = 1, tip uy, n = 2/4/8/16/32 (all converge to the same limit):

    nu=0.4999  CPE4 12.656 16.744 18.440 19.032 19.258 | CPE4I 15.995 17.600 18.552 19.013 19.231
               CPE4H 11.878 16.363 18.292 18.977 19.236 | CPE4IH 15.995 17.600 18.552 19.013 19.231

Slender cantilever (L/H = 50, 10 elements) tip deflection / exact:

    rect           CPE4 0.10-0.15  CPE4H 0.11-0.15  CPE4I 1.0000  CPE4IH 1.0000   (nu 0.3 and 0.4999)
    zig-zag trapezoid (alternate +-0.18 element-length skew, ~42 deg)
                   CPE4 0.010-0.015  CPE4H 0.012-0.014  CPE4I 0.09  CPE4IH 0.09

The trapezoid mesh is a harsh test (every Q4-type element degrades); the earlier test defined it and never
asserted on it. Thick cylinder under internal pressure, 4 x 12 mesh, p/p_lim = 0.90 / 0.99 / 1.03 / 1.10
(p_lim = 2/sqrt(3) sy ln(b/a) = 0.8004): CPE4 converges at every level (locks); CPE4IH and CPE4I stop converging
above p_lim (correct: no equilibrium exists).

## C3D8IH (same family, repaired in the same branch)

`c3d8ih_numba.py` had the same two defects: `eps = B @ u_elem` without the enhanced strain, one linear
condensation from alpha = 0, and `np.linalg.inv` in the condensation. The nonlinear path is the new
`compute_c3d8ih_nonlinear_numba` (alpha 9 + p_lin 3 iterated to equilibrium, material fed `B u + B~ alpha`);
the closed form, the distortion / anti-inversion controls, `K_eff`, initial stress and damping are unchanged.

Condensed tangent against a central finite difference of the condensed force, elastic-plastic state at 2-4 times
the yield strain, 8 random displacement states, `|Ke - Kfd| / |Kfd|` (worst case):

    element   before   after
    CPE4I     1e-10    1e-10     (was already consistent)
    CPE4IH    0.28     1e-10     (now exactly as consistent as CPE4I)
    C3D8IH    0.09-0.15  1e-3    (floor, see below)

The 3D floor is not this defect: the 3D small-strain-in-co-rotated-frame kernels map the Hencky strain to
Green-Lagrange before the material call and use dS/dE_GL as the tangent, which omits dE_GL/d(strain) -- an error
proportional to the strain (7.6e-5 at eps = 1e-4 in an elastic state, 1e-3 at 2e-3). Closing it means changing
the strain interface of every such kernel; not done.

What did NOT discriminate: the plane-strain plastic-bending oracle and the thick-cylinder collapse run give the same
answers for the unrepaired C3D8IH (converges, within 3 %; no exception), so those two tests pin behaviour only.
The finite-difference tangent test is the one that fails on the old kernel (6 % vs a 0.5 % bound).

## Not done / follow-up

- **Five additional variables.** Abaqus exposes the incompatible-mode amplitudes as 5 extra solution-dependent
  variables of CPE4I/CPE4IH (4 for CPS4I). Here alpha is re-solved on every call and never stored, so those
  variables are not in the SDV array or the output. Storing them would also give a warm start. Not implemented:
  the SDV array is sized per material, and adding element-level slots touches restart and output.
  (The element reference lists five for CPE4IH too, while the Theory Guide says hybrid elements omit the
  locking-prevention modes; the meaning of the fifth variable for CPE4IH is unclear from the text available.)
- 3D `C3D8I` (`c3d8_eas_tl_numba.py`) iterates alpha but still calls `np.linalg.solve/inv` (with a ridge); not changed.
- The 3D strain-interface floor described above (1e-3 tangent error, proportional to strain).
- **CPE4H (solver2d) converged to u = 7e10 at p = 1.10 p_lim in the cylinder probe** instead of failing. It is not
  CPE4I/CPE4IH, so it was left alone, but a "converged" answer that large should be looked at.
- 3D `C3D8I` and `C3D8IH` have 13 internal dof in Abaqus (9 + 4 volumetric); the repo's C3D8I uses 9.
