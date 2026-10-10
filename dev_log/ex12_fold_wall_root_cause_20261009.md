# ex12 fold wall -- root-cause analysis (2026-10-09)

Question (user): why does the ex12 14-layer PET/PSA/GLASS fold not get past a few
degrees -- and why did weeks of changes not fix it?

All numbers below were measured on the tracked deck (`examples/ex12_rigid_plate_display_fold.inp`,
reproduced byte-for-byte by `gen_ex12_inp.py` + `make_teardrop_config()`) unless stated otherwise.
Saved results were analysed with scratch scripts (layer curvature / corner-Jacobian / PSA shear,
display-vs-plate signed distance, per-increment acceptance) rather than by re-reading run summaries.

## 1. Why the previous approach could not find it

Every investigation (bisecting commits, flipping one default at a time, solution-control presets)
measured one number: the angle at which the run aborted. That number is downstream of four
independent defects, so no single flip moved it and the bisect only told *when* things got worse,
never *why*. Three further facts hid the mechanisms:

- **Convergence was never actually checked** (section 3.2): every increment of every run "converged"
  in exactly 2 Newton iterations, whatever the residual. "Zero cutbacks, 1-2 iterations/step" --
  repeatedly reported as health -- is what this criterion produces by construction.
- **Some diagnostics could not see the failure**: the display/plate tracking check compares the
  display against the RBE2 *master* node's motion -- the pinned pivot -- so it always prints
  `plate=0.000mm` and ratio 1.0000 whatever happens; nothing monitored the untied strip at all
  (the tie-gap monitor only covers tie pairs -- and it was right: 0.0006 mm at 39.6 deg, 0.561 mm
  at 41.2 deg when the tie tore in the final, falsely accepted increment; an earlier draft of this
  note wrongly said it had printed 0.0000 there -- that was a different run's step). The monitor's
  "Newton Iters: 1" is printed for 2-iteration increments.
- **The model drifted away from the one that once reached 90 deg** (2026-07-25/26): PSA became
  10x softer (mu 0.168 -> 0.0156 MPa), a GLASS top layer was added, the teardrop geometry and an
  untied buffer strip were introduced -- and no ex12 gate exists (`verification.run_all` does not
  run it), so each change was judged by its commit message. `7c81068` ("optimize ex12 folding
  convergence ... 100% 최고속 완주") reaches 3.25 deg; it was never run.

## 2. The model was ill-posed: the display falls through the plate

`96f8adb` (2026-09-10) set `tie_attach_start = 12.0` in the teardrop config: the plate spans
|x| in [7.5, 40] but the tie starts at |x| = 12, leaving a 4.5 mm strip per side where the display
rests on the plate **with neither tie nor contact** (the deck has no *CONTACT at all).

Signed distance of display-bottom nodes from the rotated rigid plate face (negative = inside):

| run | onset | at abort |
|---|---|---|
| penalty tie + CPE4I (reached 41.2 deg) | -0.057 mm at 3.8 deg | **-1.345 mm** at x=+-7.5 (display is 0.63 mm thick); tied zone -0.56 mm = the tie tearing in the last (falsely accepted) increment |
| kinematic tie + Q4_VISCO_SIMO (3.25 deg) | -0.007 mm at 2.2 deg | -0.147 mm |

Full-tie deck (`tie_attach_start=None`, everything else identical) -- the discriminating experiment:

| variant | buffer-strip deck | full-tie deck |
|---|---|---|
| kinematic + CPE4I | 1.03 deg | **38.08 deg** |
| penalty + CPE4I | 35.86 deg | **39.04 deg** |
| kinematic + Q4_VISCO_SIMO | 3.25 deg | 9.23 deg |

Fix: `constraint/rigid_segment_contact2d.py` (frictionless penalty contact against the rotating
plate face, segment-bounded) wired into ex12 as `solver.plate_contact`; or tie the full plate span.

## 3. Solver defects on the ex12 path (legacy `solver/dynamic.py`)

### 3.1 Unsymmetric tangent solved as symmetric (RBE2-condensation path)

The condensation path passed `is_symmetric=True` unconditionally -> PARDISO Cholesky on the upper
triangle. Measured on every linear solve of the first 30 steps:

| PSA element | asymmetry of K | linear-solve error vs the true matrix |
|---|---|---|
| CPE4I | 1.3e-16 | <= 5.8e-13 |
| Q4_VISCO_SIMO (F-bar) | median 1.8e-5, max 1.8e-3 | median 1.4e-3, **max 0.15** |

Fixed: symmetry is now measured on `K_full` before condensation.

### 3.2 Displacement-only acceptance against the TOTAL displacement

`rel_change = |du| / |u_total|` and `rel_change < tol` alone accepts (5-way OR). |u_total| grows
with the fold, so the test weakens monotonically; mid-fold any second iteration passes. The
~38-40 deg wall in all CPE4I runs is the first increment that needed line search: accepted with
the residual RISING (kinematic: 20.6 -> 33.4 N, LS 0.25; penalty: 377 -> 188 N, LS 0.5), after
which the next increment's EAS local solve fails. Abaqus compares c_max with the *increment*
and ANDs it with the residual test.

Fixed by removing the displacement-only (and energy-only) acceptance; the Abaqus R&C std/alt,
tiny-relative-residual and absolute-correction branches remain. Validation, A/B against the
old criterion:

| check | old | new |
|---|---|---|
| `verification.run_all` (exercises this solver) | 13/14 | 13/14; every value bit-identical except the already-failing Gulati (319.4262 -> 319.4241) |
| ex03 end-to-end (the AGENTS 4.7 regression) | 90 deg, 0 cutbacks, 54 NR iterations (1 per step) | 90 deg, 0 cutbacks, 132 NR iterations, same wall time |
| legacy-solver pytest (19 files) | 200 passed / 3 failed | identical fail set |
| ex12 full-tie, kinematic + CPE4I | 38.08 deg | **49.50 deg**, every accepted increment residual-checked |
| `tests/test_no_displacement_only_convergence.py` (Cook's membrane after a rigid 1e3 mm pre-translation) | tip 7.3988 vs 7.4355 (0.5% error from a stress-free translation) | identical |

## 4. Element: PET/GLASS on an orientation-dependent element (AGENTS 4.1)

`Q4_COROTATIONAL_SRI` bending stiffness 1.2x axis-aligned -> 20-80x at 45 deg for these aspect
ratios. The onset residual at the ~38 deg wall sits in the free span (glass/PSA interface x=4.25,
PET x=+-5.75) where elements approach 45 deg.

Same setup (full tie, kinematic, CPE4I PSA, OLD convergence criterion), PET+GLASS element only:

| PET/GLASS element | reached | end state |
|---|---|---|
| Q4_COROTATIONAL_SRI | 38.08 deg | false acceptance, then EAS local solve fails |
| Q4_EAS | 26.59 deg | false acceptance (residual 3.8 -> 51.4 N at LS=1); the first tied bottom-PSA element at the plate corner crushed through zero thickness |
| Q4_COROTATIONAL_EAS | **43.54 deg** | false acceptance (61 -> 182 N), then EAS local solve fails on 25 elements |

Q4_EAS's J2 path may carry the NumPy J2 tangent defect (AGENTS 4.16), so this ranks the paths,
not the formulations. With a frame-invariant (softer, correct) PET the stack wraps the rigid
plate corner sooner -- SRI's artificial stiffness was partly masking the corner problem (section 5).

## 5. The wall after the convergence fix: the rigid plate corner (49.5 deg)

With residual-checked convergence the run reaches 49.50 deg (14 cutbacks). In the last
CONVERGED state the bottom-PSA element at the plate's inner edge (pid 1, x~-7.62, the first tied
element at the sharp 90 deg rigid corner) already has a corner Jacobian J/J0 = -0.264 (CPE4I's
Gauss-point Jacobians stay positive, so it converges); the next increment's EAS local solve fails
at every cut-back dt. No penetration anywhere, all other layers smooth (min J/J0 >= 0.67). The
stack is wrapping the corner and squeezing a single row of nu~0.49 adhesive -- a mechanism one
element row cannot represent. Candidates: more PSA rows near the corner, a u-p hybrid PSA element,
an unbonded strip with (validated) contact so the display can lift at the corner, a filleted
plate edge.

## 6. Mechanisms observed (kinematic + Q4_VISCO_SIMO, buffer deck)

At 3.25 deg every layer kinks at x~+3.6 (6-10 curvature sign flips, asymmetric => bifurcation) and
PSA elements invert there (J/J0 -0.44/-0.77/-0.55); x=3.6 is inside a uniform 0.25 mm region, so
not mesh-triggered. Downstream of 2 + 3.1.
