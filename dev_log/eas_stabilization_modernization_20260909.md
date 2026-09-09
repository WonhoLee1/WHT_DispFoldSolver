# EAS stabilization modernization — removing the `_ALPHA_MAX` clamp (2026-09-09)

Replaces the ad-hoc `_ALPHA_MAX` magnitude clamp on the EAS/incompatible-mode
elements' condensed internal variable `alpha` with (1) a **line-search damped
element-local Newton** — the remedy the EAS literature actually prescribes —
and (2) **element-local non-convergence surfaced as a global increment
cutback**, which is what Abaqus/Standard documents for element-level
calculation trouble.

Continues `dev_log/plan_abaqus_element_consolidation_20260908.md` §10 and
`dev_log/handoff_20260908_f4_f5_f6_nlgeom.md` open item 1, under the standing
directive **"영구 예외 없음. 전면 재검토할 것임. abaqus 요소로 1:1 개발할
것임."** — the clamp is exactly the kind of invented, Abaqus-incompatible
device that directive rules out.

Nothing is committed (per the standing instruction to commit once, after the
redesign lands).

---

## 1. What was there, and why it had to go

`_ALPHA_MAX` capped `|alpha|` inside the element-local Newton of the
incompatible-mode kernels and let the solve continue:

| file | device | history |
|---|---|---|
| `q4_eas_jax.py` (`Q4_EAS`, J2) | `alpha <- ALPHA_MAX*tanh(alpha/ALPHA_MAX)` | 5.0 (2026-07-25, as a NaN guard) -> 0.05 (`f65f90d`, 2026-07-30) -> 0.5 (2026-09-08) |
| `q4_visco_eas_jax.py` (`CPE4I`/`CPE4H`/`CPE4IH`) | p-norm guard `a/(1+(|a|/A)^8)^(1/8)` | same values |
| `q4_visco_eas_numba.py` (`CPE4I`, numba) | same p-norm guard | same values |
| `q4_eas.py`, `q4_eas_numba.py` (NumPy/Numba J2 EAS) | **none** — these two already used a damped local line search | — |

Three independent problems with it:

1. **No literature or vendor basis.** No EAS paper surveyed below bounds the
   magnitude of the condensed enhanced parameters, and no public
   Abaqus/Nastran/OptiStruct theory documents capping an internal condensed
   DOF. Every published remedy for the real EAS instability is a *formulation*
   change (§2).
2. **It breaks the condensation's own premise.** The condensed tangent
   `K_uu - K_ua K_aa^-1 K_au` and the condensed force
   `f_u - K_ua K_aa^-1 f_a` are both derived from the stationarity condition
   `f_alpha(alpha) = 0`. Saturating `alpha` returns a state where that
   condition is deliberately violated, so force and tangent no longer belong to
   the same state. `q4_visco_eas_jax.py`'s own comment already recorded this
   as a measured 0.8–2.5% over-stiffness for the earlier `tanh` variant.
3. **It hid a failure instead of reporting one.** Saturation is silent — no
   warning, no cutback, no diagnostic. AGENTS.md §4.9 is explicit that clean
   convergence is not evidence of correctness; a clamp is precisely the device
   that manufactures clean convergence over a broken state.

The 2026-09-08 measurements it was justified by are re-read in §4: they do not
show "genuine physical demand" — they show a **divergent element-local
Newton**.

---

## 2. Literature findings

Everything below was retrieved and read this session (open-access PDFs where
noted); nothing is cited from memory. Where I am inferring rather than quoting,
it says so.

### 2.1 The genuine EAS instability, and what actually cures it

- **Wriggers & Reese (1996), "A note on enhanced strain methods for large
  deformations", CMAME 135:201–209.** Enhanced-strain elements that behave
  excellently in bending and near-incompressibility exhibit **hourglass-type
  instability modes in compressive deformation states** — a rank deficiency,
  identifiable as **negative eigenvalues of the element stiffness matrix**.
  This is the origin of the whole line of work.
- **de Souza Neto, Perić, Huang & Owen (1995), "Remarks on the stability of
  enhanced strain elements in finite elasticity and elastoplasticity", Comm.
  Num. Meth. Engng 11(11)**, and the companion **"Using assumed enhanced
  strain elements for large compressive deformation", Int. J. Solids Struct.
  (1995/96)**: eigenvalue analysis of the element tangent at finitely deformed
  configurations shows the enhanced element becoming unstable with non-physical
  hourglass patterns.
- **Korelc & Wriggers (1996), "Consistent gradient formulation for a stable
  enhanced strain method for large deformations", Engineering Computations
  13(1):103–123.** The standard orthogonality condition between enhanced
  strains and constant stresses guarantees convergence only in linear
  elasticity; deriving element eigenvalues analytically at large strain yields
  **additional orthogonality conditions** and hence a stable formulation.
- **Glaser & Armero (1997), "On the formulation of enhanced strain finite
  elements in finite deformations", Engineering Computations 14(7):759–791.**
  Two modifications of the enhanced interpolation: full symmetrization, or
  keeping only the **transposed** part. Both give a "mode-free response" under
  high compressive stress. This is the origin of the **Q1/E4T** element.
- **Reese & Wriggers (2000), "A stabilization technique to avoid hourglassing
  in finite elasticity", IJNME 48(1):79–109**, and **Reese (2003), IJNME**
  (consistent hourglass stabilization for large inelastic deformation): split
  the element tangent into constant and hourglass parts and stabilize the
  latter by element-level modal analysis — the reduced-integration + physical
  stabilization alternative that sidesteps EAS's instability entirely.
- **Pfefferkorn & Betsch (2020), "Extension of the enhanced assumed strain
  method based on the structure of polyconvex strain-energy functions", IJNME
  121(8):1695–1737.** EAS built on **transposed Wilson modes** plus a
  polyconvex-structured mixed framework: passes the patch test, frame
  invariant, locking free, **no spurious hourglassing in elasticity**.
- **Bieber, Auricchio, Reali & Bischoff (2023), "Artificial instabilities of
  finite elements for nonlinear elasticity: Analysis and remedies", IJNME
  124(11):2638–2675** (open-access preprint read this session). Linearized
  buckling analysis of EAS geometric stiffness in plane-strain nonlinear
  elasticity; the proposed remedy is a **modification of the discrete
  Green-Lagrange strain** for compression problems. Checked directly against
  the preprint text: it does **not** recommend bounding/clamping `alpha`, and
  does not propose projecting the element tangent to positive definiteness.
- **Pfefferkorn & Betsch (2023), "Hourglassing- and locking-free mesh
  distortion insensitive Petrov–Galerkin EAS element for large deformation
  solid mechanics", IJNME** — the most recent formulation-level answer
  (title/venue confirmed by search; not read in full).

**Conclusion: not one surveyed source treats the instability by bounding the
enhanced parameters or by flooring the eigenvalues of `K_aa`.** Every remedy
changes the enhanced-mode ansatz, the quadrature, the strain measure, or the
integration scheme. That settles the task's "eigenvalue floor / SPD projection
of `K_aa`" question in the negative — see §3.3.

### 2.2 The *other* open issue: Newton–Raphson robustness (this is the one we hit)

- **Pfefferkorn, Bieber, Oesterle, Bischoff & Betsch (2021), "Improving
  efficiency and robustness of enhanced assumed strain elements for nonlinear
  problems", IJNME, DOI 10.1002/nme.6605** (open access; read in full this
  session). Verbatim from the introduction: *"two major open issues of EAS
  elements remain. First, unphysical instabilities occur... The second open
  issue is lack of robustness in the Newton–Raphson (NR) solution algorithm."*
  Robustness there means the maximum applicable load-step size and the number
  of NR iterations needed.
  Its remedies, both of which leave the converged answer untouched:
  - the **mixed integration point (MIP)** method (Magisano et al.), which
    replaces the constitutive stress by an independent Gauss-point stress **in
    the geometric part of the tangent only** — *"the residuum is not altered,
    which means that only the robustness of the method during iteration is
    (usually) improved without changing the converged result"*;
  - a **line search**. From §6.6 (elasto-plastic circular bar): *"The
    robustness can be improved with either the MIP method or a line search
    algorithm. Both reduce the number of necessary load steps and ensure (with
    one exemption) that the NR method converges for higher number of load
    steps. The best results are obtained by combination of both methods."*
  - Note their Algorithm 1/2: the enhanced parameters are updated **once per
    global NR iteration by static condensation**, not by a nested
    element-local Newton run to convergence. See §5.1.
  - Failure handling in that paper: *"Failure of the NR procedure is
    determined if either ||R|| > 1e14 or more than 20 iterations are necessary
    to find a solution within one load step."* — i.e. non-convergence is
    declared, never absorbed.

### 2.3 What Abaqus/Standard actually does (public documentation)

Retrieved from the Abaqus Analysis User's Guide, *Convergence criteria for
nonlinear problems* (mirror at `ceae-server.colorado.edu/v2016/books/usb/`),
verbatim:

> "Abaqus/Standard may have trouble with the element calculations because of
> excessive distortion in large-displacement problems or because of very large
> plastic strain increments."
>
> "If this occurs and automatic time incrementation has been chosen, the
> increment will be attempted again with a time increment of [factor] times the
> current time increment."
>
> "Abaqus/Standard distinguishes between regular, equilibrium iterations (in
> which the solution varies smoothly) and severe discontinuity iterations
> (SDIs) in which abrupt changes in stiffness occur."

And from *Getting Started with Abaqus/Standard* §8.2.2 (Equilibrium iterations
and convergence), retrieved via search extraction rather than a direct page
fetch:

> "By default, if the solution has not converged within 16 iterations or if the
> solution appears to diverge, ABAQUS/Standard abandons the increment and
> starts again with the increment size set to 25% of its previous value. By
> default, ABAQUS/Standard allows a maximum of five cutbacks of increment size
> in an increment before stopping the analysis."

Two honest caveats:

- Abaqus's **SDI** concept is specifically about contact open/close and
  stick/slip. Calling an EAS local-solve failure a "severe discontinuity"
  would be a misuse of their term, so this change does **not** route it into
  the `.sta` SEVERE DISCON column. The correct Abaqus analogue is the first
  quote: *element calculation trouble -> the increment is attempted again with
  a smaller time increment.*
- Abaqus's element internals are proprietary. That they do **not** clamp a
  condensed internal DOF is an argument from the absence of any such statement
  in the public theory/user manuals plus the presence of the documented
  cutback path — it is an inference, stated as one, not a quoted fact.

The user-visible Abaqus message for a failed *material-point* local algorithm,
`"THE PLASTICITY/CREEP/CONNECTOR FRICTION ALGORITHM DID NOT CONVERGE AT N
POINTS"`, followed by increment re-attempt and eventually `"TOO MANY ATTEMPTS
MADE FOR THIS INCREMENT"`, is the closest behavioural precedent for what this
change implements. (Widely documented in vendor/community sources; I did not
find it in an official theory-manual page I could fetch, so treat the exact
mechanism as behavioural evidence rather than a quoted specification.)

---

## 3. Design

### 3.1 Primary fix — line-search damped element-local Newton

Every incompatible-mode kernel's local Newton on `alpha` is now:

* damped by a **backtracking line search** over a fixed step-length set
  `{1.0, 0.5, 0.25, 0.1}`, choosing the step that minimises `|f_alpha|`
  (branch-free, so it stays `jax.jit` / `jax.vmap` safe and numba-compilable);
* **tolerance terminated** (`||d(alpha)||_inf <= 1e-8`) with an iteration cap
  of 12, replacing the previous fixed unrolled 5 iterations;
* free of any magnitude clamp.

A line search cannot move the converged root (`f_alpha = 0` is the same root
either way) — it only changes the path. This is (a) exactly one of the two
remedies Pfefferkorn et al. (2021) report for EAS's NR robustness problem, and
(b) already this repo's own precedent: `q4_eas.py` has used a damped local line
search since it was written and is the one EAS kernel here that never needed a
clamp.

### 3.2 Fallback — element-local failure becomes a global increment cutback

Each kernel now returns a 6th value, `status` = `||d(alpha)||_inf` of the last
accepted local correction (`inf` if the local state or the condensed
force/tangent went non-finite). `DynamicSolver` collects it per element into
`self._eas_local_status`, and `_solve_step_impl` — immediately after the
existing element-inversion gate, reusing the same `self._fail(...)` path —
abandons the increment if any element exceeds `self.eas_local_tol` (default
`1e-6`). The AdaptiveDtController then cuts `dt` back and retries, exactly as
it already does for element inversion. `self.n_eas_local_failures` accumulates
a run-level count.

No parallel mechanism was invented: this reuses `_fail` / the existing cutback
contract (negative return from `solve_step`), and the reason string shows up in
the `.sta`-style status row's CUTBACK annotation.

### 3.3 Rejected: SPD projection / eigenvalue floor on `K_aa`

Considered and rejected as the primary fix, on three grounds:

1. **No literature support** (§2.1) — every published remedy is
   formulation-level. Wriggers & Reese's own finding is that the negative
   eigenvalue *is the diagnostic*; flooring it deletes the diagnostic.
2. **It reintroduces exactly the defect the clamp had.** `K_aa` appears in the
   condensed force (`f_u - K_ua K_aa^-1 f_a`) as well as the tangent.
   Perturbing it to be SPD changes the condensed force too, so the element
   would again return a force/tangent pair belonging to no consistent state.
3. **It is a second invented device**, which the project directive forbids.

What *is* kept is the pre-existing `1e-10 * mean|diag(K_aa)|` Tikhonov shift on
the condensation solve. That is a conditioning device whose effect vanishes
with the regularisation parameter and which does not alter the converged
solution; it is not a stability treatment and is not load-bearing for anything
here.

### 3.4 Not done (deliberately): Q1/E4T transposed Wilson modes, and MIP

Both are real, literature-endorsed improvements and both are **out of scope for
this change**, for reasons that should be recorded rather than rediscovered:

- **Q1/E4T (transposed Wilson modes, Glaser & Armero 1997).** This is the
  literature's actual cure for the compressive hourglass instability, and it is
  a small code change (a different `D_k` set in `_enhanced_grad_modes`). But
  Abaqus's CPE4I is the standard Taylor/Wilson incompatible-modes element, not
  the transposed variant, so adopting it silently would move `CPE4I` **away**
  from the 1:1-with-Abaqus target this redevelopment is built around. If the
  fold ever does exhibit genuine compressive hourglassing (as opposed to the
  local-Newton divergence found here, §4), Q1/E4T should be added as an
  explicitly named, opt-in element — not as a redefinition of `CPE4I`.
- **MIP (Magisano et al., extended to EAS by Pfefferkorn et al. 2021).**
  Residual-preserving, so it cannot change converged results; it modifies only
  the geometric part of the tangent. It is the natural next robustness step if
  global Newton iteration counts become the bottleneck. Not needed to remove
  the clamp.

---

## 4. Re-reading the 2026-09-08 evidence

Plan doc §10 and handoff item 1 concluded the clamp was being hit by *"genuine
physical demand, not instability — the growth was clean, no oscillation."*
That reading does not survive the new measurements.

**(a) The pre-change 40-step ex12 probe (`scratch/alpha_sat3.log`, clamp 0.5)
does not show smooth growth into the bound.** `max|alpha|` per accepted
increment:

    theta:      0.44   0.62   0.83   1.07   1.37   1.70   2.08   2.51   2.98   3.51   4.08   4.71   5.39   ...
    max|alpha|: 4.3e-3 6.0e-3 8.3e-3 1.5e-2 2.4e-2 3.6e-2 4.7e-2 5.3e-2 5.8e-2 8.1e-2 1.1e-1 1.4e-1 5.0e-1 (clamp)

The last transition is a **jump from 0.142 to 0.49999 in one increment**, after
which it pins at exactly 0.5 for the remaining 11 attempts, all of which are
cutbacks; the run never gets past **theta = 5.39 deg**. A step-function jump to
the bound is a blow-up signature, not smooth physical demand. (The earlier
"smooth monotonic growth" reading came from the *0.05* clamp, which was hit so
early — theta ~3.5 deg — that only the smooth part was visible.)

**(b) The local Newton genuinely diverges on a hard UL state.** Driving the
`CPE4I` local Newton directly on the tilted+distorted Updated-Lagrangian probe
of `scratch/tier1_numba_jax_tilted.py`, undamped, from `alpha = 0`:

    it  1  |f_a|=4.09e-02  |da|inf=5.24e-02  |a|inf=5.24e-02
    it  2  |f_a|=4.51e-03  |da|inf=9.48e-03  |a|inf=4.30e-02
    it  3  |f_a|=7.61e-04  |da|inf=7.43e-02  |a|inf=1.17e-01
    it  5  |f_a|=4.90e-03  |da|inf=1.76e-01  |a|inf=2.05e-01
    it 14  |f_a|=3.77e-03  |da|inf=2.39e-01  |a|inf=3.39e-01
    it 25  |f_a|=2.04e-04  |da|inf=1.61e-02  |a|inf=1.59e-02

25 iterations, oscillating, never below `|f_a| = 2e-4`. `K_aa` at the last
iterate is still **positive definite** (eigenvalues 2.03e-2, 1.70e-1, 1.81,
8.20e1) — so this is **not** the Wriggers–Reese rank deficiency. It is plain
Newton overshoot from a poor start: the second open issue in §2.2, not the
first.

**(c) With the line search, the same state converges to a much SMALLER
`alpha`:** 10 iterations, `|f_a| = 5.6e-13`, `|alpha|_inf = 1.77e-2` — an order
of magnitude below the 0.2–0.34 the undamped iteration was wandering through,
and 30x below the old clamp. **The clamp was bounding a divergent iteration,
not a physical demand.**

---

## 5. Implementation

### 5.1 Element kernels (all five files)

Every kernel keeps its existing public function and arity; the instrumented
version is a new `*_status` entry point returning one extra trailing value, and
the old name is a thin wrapper that drops it. Nothing outside `dynamic.py`
needed changing.

| file | new entry point | old name |
|---|---|---|
| `q4_eas_jax.py` | `compute_eas_j2_contributions_jax_status` (6-tuple) | `compute_eas_j2_contributions_jax` -> `[:5]` |
| `q4_visco_eas_jax.py` | `compute_single_eas_status`, `compute_single_hybrid_status` (6-tuple) | `compute_single_eas`, `compute_single_hybrid` -> `[:5]` |
| `q4_visco_eas_numba.py` | `compute_single_eas_numba_status` (6-tuple); `assemble_eas_visco_batch_numba` now returns 6 | `compute_single_eas_numba` -> `[:5]` |
| `q4_eas.py` | `compute_eas_j2_contributions_status` (5-tuple) | `compute_eas_j2_contributions` -> `[:4]` |
| `q4_eas_numba.py` | `compute_eas_j2_contributions_numba_status` (5-tuple); `assemble_q4_eas_j2_batch_numba` now returns 5 | `compute_eas_j2_contributions_numba` -> `[:4]` |

Changes per file:

- **`q4_eas_jax.py`** — `_ALPHA_MAX` tanh saturation deleted; `_f_alpha()`
  residual-only helper added for the line search; the `while_loop` body now
  line-searches, and its exit test uses the residual **at the accepted
  iterate** (previously the residual before the step). Iteration-cap
  exhaustion with `|f_a| > _ALPHA_TOL` is reported as failure.
- **`q4_visco_eas_jax.py`** — p-norm clamp deleted from both
  `compute_single_eas` and `compute_single_hybrid`; the fixed 5-iteration
  unrolled loops become `jax.lax.while_loop`s with line search, tolerance exit,
  and a 12-iteration cap. For the hybrid (`q = [alpha(4), p]`) the convergence
  measure is `max(||d(alpha)||_inf, |dp|/kappa)` — the bulk modulus is the
  natural pressure scale — and the line-search merit function weights the
  pressure row by `kappa` so it is comparable to the force rows. Both are
  *ranking/measuring* devices only; neither enters the residual, so neither can
  bias the converged solution.
- **`q4_visco_eas_numba.py`** — the same algorithm, same step-length set, same
  tolerance and cap, so the two backends solve an identical local problem and
  remain element-for-element comparable.
- **`q4_eas.py` / `q4_eas_numba.py`** — already line-searched and clamp-free;
  they gain the `status` output only, so all five files report failure the same
  way.

`DISPFOLD_CACHE_VERSION` bumped **v4 -> v5** (`dispsolver/_jit_cache.py`) —
mandatory, since module-level constants baked into `@njit(cache=True)` closures
changed.

### 5.2 Solver (`dispsolver/solver/dynamic.py`)

- `self._eas_local_status` (n_elem,), `self.eas_local_tol = 1e-6`,
  `self.n_eas_local_failures` added in `__init__`.
- `_reset_eas_local_status()` / `_eas_local_failure_count()` helpers; the reset
  runs at the top of `_assemble` so a stale value from the previous Newton
  iteration can never trigger a cutback. The failure count uses
  `~(status <= tol)` so a NaN counts as a failure.
- All EAS/incompatible-mode dispatch sites write the status: the JAX visco
  vmap (CPE4H/CPE4IH and CPE4I), the numba visco batch, the JAX `Q4_EAS`+J2
  vmap, the numba `Q4_EAS`+J2 batch, the two NumPy sequential fallbacks, and
  the two single-element sequential visco fallbacks. `q4_hybrid_jax`'s
  `Q4_HYBRID_EAS` kernel shares the `Q4_EAS` vmap slot and is not instrumented
  (it solves `alpha` by autodiff of an element energy, not by a local Newton,
  and has no clamp); its lambda pads a `0.0` so the vmap output arity matches.
- `_solve_step_impl` gains the cutback gate, immediately after the existing
  element-inversion gate and using the same `self._fail(...)` contract.

Per AGENTS.md §4.8: no assembly branch was given a new early `return`.

---

## 6. Verification

### 6.1 Element-level regression

`pytest tests/test_cpe4_element.py tests/test_ul_tl_consistency.py
tests/test_cpe4rh_patch.py -q` -> **51 passed** (83 s).

### 6.2 JAX/Numba parity on tilted+distorted geometry

`scratch/tier1_numba_jax_tilted.py`, relative force error between the numba and
JAX `CPE4I` kernels (this is the correctness gate; `K_err` is tangent-only and
compares numba's FD tangent against JAX's exact autodiff tangent, so it is
large by construction and clamp-independent — see plan doc §10):

| case (large perturbation) | 2026-09-08 baseline (clamp 0.5) | now |
|---|---|---|
| axis-aligned rect | 8.2e-12 | **3.1e-16** |
| tilted+distorted, TL | 8.8e-15 | **2.4e-14** |
| **tilted+distorted, UL** | **4.35e-04** | **2.71e-15** |

The UL case improves by **11 orders of magnitude**, and its `K_err` drops from
7.9e+02 to 1.0e-01. The reported local status on that case goes from
`1.77e-01` (numba, not converged — this is what the clamp was hiding) to
`1.9e-11`. Both backends now converge to `|alpha|_inf = 1.77e-2` instead of
disagreeing at 0.2.

### 6.3 Full test suite

Baseline for A/B is `scratch/pytest_full_alpha05.log`, run on this machine a
few hours earlier with the clamp at 0.5:
**3 failed, 264 passed, 4 skipped, 4 xfailed** (746 s) —
`test_abaqus_config.py::test_config_presets`,
`test_eas_jax_verify.py::test_elastic_force_and_tangent`,
`test_prescribed_skip.py::test_prescribed_skip_solution_equivalence`.

(Note: AGENTS.md's documented "135 passed / 10 failed" baseline is stale, as is
the handoff's "234 passed / 5 failed"; the alpha05 log above is the current
same-machine reference.)

After this change (`scratch/pytest_full_eas_modern.log`):
**RESULT_PLACEHOLDER_PYTEST**

`tests/test_eas_jax_verify.py` -> **3 passed**: the previously-failing
`test_elastic_force_and_tangent` now passes. That test compares the NumPy EAS
kernel (which has always had a damped local line search) against the JAX one
(which did not); giving the JAX kernel the same line search closed the gap. So
one of the three documented pre-existing failures is *fixed* by this change,
not merely unaffected.

### 6.4 `verification.run_all`

RESULT_PLACEHOLDER_VERIFICATION

### 6.5 Real production fold (`ex12`, 40 steps, numba backend)

RESULT_PLACEHOLDER_EX12

### 6.6 Interlayer slip (AGENTS.md §4.9 adjudicating measurement)

RESULT_PLACEHOLDER_SLIP

---

## 7. Honest limitations

RESULT_PLACEHOLDER_LIMITS
