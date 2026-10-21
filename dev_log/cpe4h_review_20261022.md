# CPE4H review after 6456909 (2026-10-22)

State on master: f3bccf1 (agy) contains the fix below (it swept up the uncommitted shared tree again); this note is the record.

## What 6456909 broke
CPE4H on a non-linear-elastic law got a Hencky -> Green-Lagrange strain at the Gauss points. Four tests of
`tests/test_plastic_history_2d.py` failed (15/15 passed at d7a319c):
* J2 was converted too: residual stress -9.115 vs closed form -8.920 (2.2%).
* Pressure stiffness `K_bulk = props[1]`: D1 for `[C10, D1]`, nu for J2. Now the material's reference bulk modulus
  (1^T C 1 / 9 at zero strain): `[C10, D1]`, `[E, nu]`, `[mu, K]` give one result.
* Tangent omitted dE/d eps: 3-7% off the Jacobian at 5% strain, growing with strain.

Fix (in HEAD): conversion for hyperelastic laws only (`is_hyperelastic_material`), reference K_bulk, tangent with the
chain factor dE/d eps (central difference of the conversion; FD-exact to 1e-9 at the element).
Guards: `tests/test_cpe4h_hyperelastic_large_strain.py` (fails 9/10 on 6456909). CPE4H test files: 104 passed.

## The tangent question (measured, not settled)
Original small-strain hand-over: the 14-layer fold (`example_disp_fold/test_wht_neohooke_full.py`) fails in its first
step (25 iterations), so agy's conversion was needed.
Same f, two tangents, 1 deg per step, 90 substeps:
| tangent | reaches | M_L (N*mm) |
|---|---|---|
| exact (chain factor) | step 21 (diverges at 22) | -0.31 at 5, -0.77 at 15 deg, smooth |
| dS/dE only (modified Newton) | step 87 | -1.9 at 5, -3.7 at 15, sign flip at 35, +41 at 45 deg |
FrontISTR reference (agy's table): M_L = -0.145 at 4 deg. The modified-Newton run is 13x off at 5 deg, so its "convergence"
to 87 deg is not evidence of a good solution. With 2 deg per step the exact tangent fails at step 1; a cutback harness would
likely cover that (the script has none).

## Open
* At states both variants report as converged, the internal force on the FREE dofs is not zero (no external load there):
  exact 0.56 N (max 0.12), modified Newton 27.5 N (max 6.0), against reactions 0.012 / 0.094 N. Either the 2D solver
  accepts increments with a residual far above the tolerance (AGENTS.md 4.20 class) or a stabilization is active and
  `f_int` is not the whole balance. Not investigated; check `solver2d` acceptance and stabilization first.
* agy's reported 2D final moment (1.99 N*mm at 90 deg) is not reproduced from a clean checkout of 6456909
  (M_L oscillates in sign, diverges at 27/45 substeps).
* The exact tangent stops at ~22 deg with 1 deg/step: the element's linear-B + converted-strain scheme on 0.03 mm PSA layers
  (aspect 33, shear in the hundreds of percent) is the limit, not the solver loop.
* 14-layer parity claims in `benchmark_14layer_wht_vs_fistr_ny1_vs_ny2_20261020.md` should be re-checked against the above.
