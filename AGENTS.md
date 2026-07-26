# WHT_DispFoldSolver — Project Rule File (for LLM agents)

This file is the canonical, cross-tool reference for any LLM coding agent
(Claude Code, Gemini CLI, OpenCode, etc.) working in this repository. Read
this before making changes to `dispsolver/` or `examples/`. It describes the
**target physical problem**, the **current implementation state**, the
**solver's theoretical basis**, and the **specific numerical obstacles that
have been solved** so far, so agents don't re-derive or re-break them.

Other tool-specific rule files (`CLAUDE.md`, `GEMINI.md`) point back to this
file — keep this one authoritative and update the others only if a tool
needs tool-specific instructions that don't belong here.

---

## 1. Target Problem

Simulate a **foldable display panel** transitioning from fully flat to fully
folded (target: 90° per hinge, i.e. the two halves come together face-to-
face), driven purely by rotation of two rigid support plates about a hinge
axis — not by direct load or direct displacement of the display itself.

```
 Flat (t=0)                          Folded (t=1, 90° per side)
 ─────────────────                    ╲               ╱
 [ Display Panel  ]                    ╲             ╱
 ══════════[hinge]══════════            ╲           ╱
 [Plate L]  ↕tie  [Plate R]              [PL]       [PR]
                                           ╲    ↕tie   ╱
                                            ╲ (curved) ╱
                                              hinge
```

- The display is a **deformable multi-layer sheet**; the plates are **rigid
  bodies** that only translate/rotate as a whole.
- The display's bottom face is **tied** (not welded/meshed-shared) to the
  plate top faces, so it can slip/expand in-plane during bending instead of
  picking up artificial stress concentration at a shared-node boundary.
- Only the **hinge rotation** is prescribed as a boundary condition; the
  display's actual bent shape (the "U" between the two plates) is a
  **result** of the solve, not prescribed directly.

### 1.0 What "U-shape" / "fully folded" means — success criterion

`ex11`/`ex12` drive `THETA_MAX = 90°` **per plate** (180° combined) — the
two plates rotate until they end up roughly **parallel and facing each
other**, like a book closing all the way, not just bending to a right
angle. (`ex03_corotational_v4.py` only drives 45°/side = 90° combined —
a right-angle bend, not a full close — so don't use it as the "did the
fold work" reference for the U-shape target.)

At full closure, the display's free hinge span (`x ∈ [-10, 10]`, between
the plate inner edges) must **not** form a sharp crease/V. Real foldable
displays have a minimum bend radius (material/stack constraint against
cracking); the correct final shape is a smooth, rounded **U (teardrop)
loop** connecting the two now-parallel plates, with the plates separated
by roughly the loop's diameter rather than touching. The hinge pivots
being set *inside* the plates' own footprint (`x = ∓3`, while the plates
only span `x ≤ -10` / `x ≥ 10`, see §1.2) is specifically what lets the
plate lift and retreat from the display surface as it rotates, instead of
pinching/crushing the bending display against a flush plate edge — this
geometry choice exists *because of* the U-shape/bend-radius requirement,
not despite it.

**Concretely, when judging whether a folding run "worked" at large
rotation, check for:**
1. No element inversion / crease-sharp kink in the free hinge zone
   (`solver._check_mesh_quality`, `n_inverted == 0`).
2. A smooth, continuous curvature profile in `x ∈ [-10, 10]` (rounded, not
   angular) — plot `uy` vs `x` for the display's bottom or top surface row.
3. The two plates ending up close to parallel and facing each other, not
   just individually rotated in isolation from the display (see §4.8/4.9
   — verify the display is actually *coupled* to that plate motion, not
   just that the plates themselves reached the target angle).

**Achieved (2026-07-25, end of session)**: `examples/ex12_abaqus_inp_plate_fold.py`
reaches **full 90°/side (180° combined), t=1.0, 101 steps, zero cutbacks,
1 Newton iteration per step, ~85s wall time** — all three criteria above
satisfied (see §4.12 for the fix that got it there). This is the current
reference result for "does the U-shape fold work end to end." Use this
script + `examples/gen_ex12_inp.py` (graded mesh) as the starting point for
further work, not the earlier `ex12_rigid_plate_display_fold_corotational.py`
(still stalls around 7.9°/side per §1.2, unfixed).

### 1.1 Hinge structure & location

- Two hinge pivots, one per plate, both at mid-thickness of the display
  (`y = 0` in the 2D plane-strain models used so far).
- Left pivot at `x = -3.0 mm`, right pivot at `x = +3.0 mm` (i.e. a 6 mm
  free hinge gap between the plate edges) — see
  `examples/ex03_corotational_v4.py` (`LEFT_PIVOT`, `RIGHT_PIVOT`) and
  `dispsolver/mesh/plate_builder.py` defaults.
- Rotation is driven kinematically: `theta_L(t) = -THETA_MAX * amp(t)`,
  `theta_R(t) = +THETA_MAX * amp(t)`, with `amp(t)` a C²-smooth ramp
  (smootherstep, `10t³-15t⁴+6t⁵`, or the equivalent `s²(3-2s)` cubic used in
  `ex11`) so accelerations start and end at zero — avoids an impulsive
  kick to the Newton solver at `t=0`.
- `THETA_MAX = 45°` per plate (→ 90° total fold) in `ex03_corotational_v4.py`;
  `THETA_MAX = 90°` per plate (→ 180° total, full face-to-face fold) in
  `ex11_rigid_plate_display_fold.py`. **These are two different target
  angles in the two example files — check which one a task actually needs.**

### 1.2 Plate structure & location

- Two rigid plates, one on each side of the hinge, spanning from the hinge
  gap edge out to the display edge.
- `ex03_corotational_v4.py`: plates are **not modeled as separate parts** —
  rotation BCs are applied directly to display nodes with `|x| > 10 mm`
  (i.e. the "plate region" is just a rigidly-driven subset of the display
  mesh itself, simplification used to validate the element formulation in
  isolation).
- `ex11_rigid_plate_display_fold.py` / `ex12_rigid_plate_display_fold_corotational.py`:
  plates **are** separate rigid `Q4` parts built by
  `create_folding_plate_parts()` — left `x ∈ [-40,-10]`, right
  `x ∈ [10,40]`, `y ∈ [-0.5, 0.0]` (i.e. sit directly under the display).
  `ex11` (known-bad, see §4.10) enforces plate rigidity via
  `RBE2HingeElement` (penalty + Augmented Lagrangian — imperfect, already
  shown not to work). `ex12` (correct, see §4.10) enforces it via
  `RigidBodyPart.get_slave_displacements()` applied as an exact Dirichlet
  BC on every plate node every step — no penalty, no gap, by construction.
- **Status as of 2026-07-25 (end of session), after the §4.8/§4.10/§4.11
  fixes**: `examples/ex12_rigid_plate_display_fold_corotational.py`
  (display: `Q4_COROTATIONAL` + `J2Plasticity`; plate: exact
  `RigidBodyPart` Dirichlet BC, **not** `RBE2HingeElement` — see §4.10)
  reaches **~7.9°/side with zero cutbacks and a genuinely tight tie**
  (`max_gap ≈ 1e-5 mm` at `max|u| ≈ 5 mm`, i.e. ratio ~1e-5 — the tie is
  doing its job). Progress then stalls in wall-clock terms: Newton
  iteration count plateaus around 9-10 (vs. `target_iters=8`), so
  `AdaptiveDtController` keeps shrinking `dt` every step
  (`5/(iters+1) < 1` whenever `iters ≥ 5`) with no step small enough to
  bring iters back down to reset the trend — `dt` decays toward
  `dt_min=1e-5` and simulated-time progress crawls to a near-halt well
  before `t=1.0` (90°/side). **This is a different, milder problem than
  the RBE2 approach's hard divergence at ~3.7°/side (§4.10)** — no
  physical/tie failure, just a dt-controller/iteration-budget mismatch —
  but full 180°-combined closure (the true U-shape target, §1.0) was
  **not** achieved in this session. Earlier same-session claims of "55
  steps to 101° combined, zero cutbacks" for `ex12` were made *before*
  the §4.8 tie-assembly-skip bug and §4.10 RBE2-regression were found and
  fixed — those numbers are invalid (measured on a model where the tie
  had no effect at all) and should not be cited. `ex12` (this corrected
  version) is still the right reference starting point; getting past the
  ~8° dt-decay stall (try: higher `target_iters`/`max_iter`, or address
  the §4.11 plate-assembly-waste performance issue first so more Newton
  iterations per step is cheaper) is the next concrete task.

### 1.3 Rotation condition

- Quasi-static (`mode="quasistatic"` — no inertial/dynamic terms), driven by
  a single scalar rotation angle per side, ramped smoothly over a
  normalized time `t ∈ [0, 1]`.
- Both examples use an `AdaptiveDtController` (`dispsolver/solver/dt_controller.py`)
  to grow `dt` when Newton converges in few iterations and shrink it
  (severity-scaled by `conv_rate`) on cutback — not a fixed step count.

### 1.4 Display layup, size, thickness — ⚠️ NOT YET A REAL MULTI-LAYER STACKUP

**Current code models the display as a single homogeneous material**, not
the real polyimide/OCA/cover-layer stack of an actual foldable display.
Document this honestly rather than presenting either example's numbers as
a validated physical layup:

- `ex03_corotational_v4.py`: length 80 mm (`x ∈ [-40,40]`), total thickness
  **0.35 mm**, meshed as 7 y-direction "layers" (`pid=0..6`) — but all 7
  `pid` groups share **the same single `J2Plasticity` material instance**
  (`E=4000 MPa, ν=0.3, σ_y0=80 MPa, H=400 MPa`). The 7-layer split is a
  **mesh-refinement device** for bending curvature resolution, not a
  material stackup.
- `ex11_rigid_plate_display_fold.py`: length 80 mm, thickness **0.5 mm**,
  single `NeoHookean` material (`E=50 MPa, ν=0.45`).
- **TODO / not implemented**: true multi-material layup (distinct
  Young's modulus / thickness per physical layer — e.g. PI substrate, OCA,
  cover window). If a task requires this, it must be added; don't assume
  it exists.

---

## 2. Solver — theoretical basis & references

`dispsolver/solver/dynamic.py` (`DynamicSolver`) is a 2D plane-strain
implicit Newton-Raphson FE solver with several interchangeable element/
assembly backends, JAX-accelerated where noted.

| Component | Theory | Reference |
|---|---|---|
| EAS Q4 element (`q4_eas.py`, `q4_eas_jax.py`) | Enhanced Assumed Strain, 4-mode, Total/Updated Lagrangian | Simo & Rifai (1990), *A class of mixed assumed strain methods*; Simo & Armero (1992), geometrically non-linear enhanced strain; Simo & Hughes (1998), *Computational Inelasticity*, Springer |
| Co-rotational Q4 (`q4_corotational_jax.py`) | Element rigid-rotation extracted from deformed edge vectors, local frame kept near-small-strain even at large global rotation | standard corotational FE approach (see e.g. Wriggers (2008), *Nonlinear Finite Element Methods*, Springer, ch. 4 for the general formulation this follows) |
| J2 plasticity (`plastic_jax.py`) | Finite-strain multiplicative plasticity, logarithmic (Hencky) strain return mapping, smooth (tanh-blended) yield surface for autodiff-friendly tangents | standard multiplicative-plasticity return-map formulation (cf. Simo & Hughes 1998) |
| RBE2 hinge (`rbe2_condensed.py`) | Exact kinematic master-slave condensation `u_s = u_m + (R(θ)-I)(X_s-X_m)`, eliminates slave DOFs and Lagrange multipliers entirely → pure SPD system (condition number 10¹⁶ → ~10³) | in-house; see file docstring |
| Surface tie (`surface_tie.py`) | Penalty-based node-to-segment tie, nearest-segment projection, standard 3-node (slave + 2 master) penalty force/stiffness | standard penalty contact/tie formulation, e.g. Wriggers (2006), *Computational Contact Mechanics*, 2nd ed., Springer |
| Viscous stabilization (`stabilization.py`) | `F_stab = c·M_diag·v`, damping auto-scaled so `E_stab/E_strain < 0.5%` | mirrors Abaqus/Standard's automatic stabilization (`*STABILIZE`) |
| Linear solver | PARDISO (`pypardiso`) with factorization reuse across iterative-refinement solves; adaptive ridge regularization | — |
| Time stepping | `AdaptiveDtController`: iteration-count + convergence-rate based adaptive `dt` | in-house, `dt_controller.py` |

### Convergence criterion (current, as of this session)

`solve_step()` uses a 5-way **OR** combination (any one satisfied ⇒
converged, `n_iter > 0` required): relative displacement ratio < `tol`,
energy error < `1e-15`, KKT residual ratio < `rtol`, Abaqus-style
`R_max ≤ 0.005·q_avg AND c_max ≤ 0.01·du_max`, or absolute correction
`du_norm < atol`. **A 3-Tier AND-based replacement was attempted and
reverted** (see §4.4) — don't reintroduce it without re-validating against
`examples/ex03_corotational_v4.py` end-to-end, not just unit tests.

---

## 3. How to run / verify

```bash
pytest tests/test_convergence_fixes.py tests/test_rigid_plate_tie.py -q
python -u examples/ex03_corotational_v4.py     # simplified hinge-only, 90 deg, ~50 steps, ~a few min
python -u examples/ex11_rigid_plate_display_fold.py   # true plate+tie architecture, not yet re-validated with corotational element
```

- **Always run with `python -u`** (unbuffered) or add `flush=True` to any
  debug prints when redirecting to a file — block-buffered stdout makes a
  perfectly healthy long-running solve look hung for minutes (see §4.6).
- A "no cutbacks, 1-2 Newton iterations per step" result is **not
  inherently suspicious** here: in the driven-BC examples, the majority of
  DOFs are directly prescribed (e.g. 1344/2208 in `ex03_corotational_v4.py`)
  and only the free hinge-zone DOFs need Newton iteration, so fast
  convergence is expected. If you need to sanity-check a "too easy"
  result, turn on `solver.verbose=True` for a few representative steps and
  confirm `Disp.Ratio`/`Energy.Err` are actually decreasing across
  iterations (don't just trust the accept/reject decision) and check
  `solver._check_mesh_quality(solver.u)` for `n_inverted`/`n_warped`.

### 3.1 TODO (not yet implemented, 2026-07-26): before/after PNG capture for every folding example

Proposed by the user: every folding example script (`ex03_*`, `ex11_*`,
`ex12_*`, and any future one) should save **two** PNG snapshots of the
deformed shape — one at `t=0` (flat, before) and one at the final
converged state (after) — not just the single final-state PNG some
scripts currently save (e.g. `ex12_abaqus_inp_plate_fold.py` only writes
`ex12_final_folding_shape.png`, no before-shot).

Preferred approach when this gets implemented: add one shared helper
(e.g. `plot_fold_before_after(u_initial, u_final, mesh, ...)`) rather
than duplicating matplotlib code per script — check
`dispsolver/postprocess/viewer.py` first for reusable plotting code
before writing new (it currently has no functions defined, so a new
helper likely belongs there or in a new `postprocess/` module, not
copy-pasted into each example).

Open question not yet decided: apply to all `examples/ex*.py` folding
scripts at once, or just the primary reference ones (`ex03_corotational_v4.py`,
`ex11_rigid_plate_display_fold.py`, `ex12_abaqus_inp_plate_fold.py`,
`ex12_rigid_plate_display_fold_corotational.py`) first. Ask the user
before doing a repo-wide sweep — scope was not settled when this was
recorded.

---

## 4. Numerical obstacles solved this session (2026-07-25) — don't re-break these

### 4.1 The "38° wall"
Standard Q4 elements + direct load/displacement drive would stall or
invert elements (`det(F) ≤ 0`) around ~38° of fold. Fixed by combining
co-rotational kinematics (isolates element rigid rotation from strain
computation) with RBE2 kinematic condensation (removes the saddle-point
conditioning problem) and (optionally) viscous stabilization.

### 4.2 Co-rotational Q4 element was silently dead code
`mesh.add_element(..., "Q4_COROTATIONAL", ...)` **does nothing** —
`DynamicSolver` never reads the mesh's per-element type string. Element
type is controlled **only** by the `element_type=` constructor argument
(a string, or a `{pid: str}` dict routed through `element_type_by_pid`).
To actually use the corotational element you must pass
`element_type={pid: "Q4_COROTATIONAL", ...}` (dict form, which also
requires `material=` to be a matching `{pid: ...}` dict) — see
`dynamic.py:_pid_element_type()`.

### 4.3 `eigh` reverse-mode NaN gradient
`pk2_voigt_jax` (`material/plastic_jax.py`) embeds the 2D deformation
gradient into 3×3 and calls `jnp.linalg.eigh` on the left Cauchy-Green
tensor. Under `jax.jacobian`/`jacrev`, this **NaNs out whenever two
eigenvalues coincide** — which happens exactly at "no out-of-plane strain"
states (the always-unit 3rd eigenvalue collides with an in-plane one),
i.e. constantly in a plane-strain 2D model. `tangent_voigt_jax` already
mitigates this with `1e-10` symmetry-breaking noise before its own
`jacfwd` pass; any **new** code that differentiates through `pk2_voigt_jax`
directly (e.g. a hand-rolled `jax.jacobian` over an element routine) needs
the same noise injection or will silently zero out the whole element via
the NaN-guard (which looks like "converged with zero force", not a crash).

### 4.4 Full-autodiff element tangent ⇒ unusable JIT compile time
An earlier version of `compute_corotational_j2_contributions_jax` computed
the tangent via `K = jax.jacobian(force_fn)` where `force_fn` looped 4 GPs
of `eigh`-based plasticity — `jax.jit(jax.vmap(...))` of this took **480+
seconds and never finished**. Fixed by switching to an **analytic
material-only tangent**: `K_local = Σ_gp BLᵀ·C_v·BL·w` (using the existing
cheap `tangent_voigt_jax`, one `jacfwd` per GP, not a full element
autodiff), then `K_global = T8 @ K_local @ T8ᵀ`. This is a **modified
Newton** (drops the `dT8/du` rotational geometric-stiffness term) — costs
a couple extra Newton iterations near large rotation increments (observed:
NR iters briefly 2 instead of 1 around the 38° region), not full quadratic
convergence, but compiles in ~3s and the line-search/cutback machinery
already covers the resulting non-quadratic convergence.
**Do not reintroduce full-element `jax.jacobian` autodiff tangents for
plasticity-coupled elements without budgeting for this compile-time
cliff.**

### 4.5 Per-pid JIT recompilation for identical materials
Building a fresh Python closure + `jax.jit(jax.vmap(...))` inside a
`for pid, mat_adapter in self.materials.items(): ...` loop forces one
compile **per pid** even when several `pid`s share the literal same
material object — JAX's cache doesn't know two different Python closures
are semantically identical. Fixed by caching the jitted vmap function by
`(lam, mu, sigma_y0, H)` key (`_coro_vmap_cache` in `dynamic.py`) so pids
with identical material params share one compiled function. Recompilation
is still unavoidable across pids with **different element counts** (JAX
recompiles per input shape) — this is expected and not a bug.

### 4.6 Buffered stdout looked like a hang
After removing `flush=True` debug prints, a genuinely-healthy run
redirected to a log file (`python examples/foo.py > out.log 2>&1`)
produced **no visible output for 400+ seconds** even though it was
actually progressing normally — Python fully block-buffers stdout when
it's not a TTY. Always use `python -u` (or explicit `flush=True`) when
piping a long-running solve to a file for live monitoring; don't conclude
"hung" from silence alone — check with `ps`/process-alive and re-run with
`-u` before assuming a real bug.

### 4.7 3-Tier convergence rewrite (attempted, reverted)
Replacing the 5-way OR convergence criterion (§2) with a strict
`(force_converged AND correction_converged) OR abs_converged` 3-tier check
looked like a reasonable "Abaqus-style" cleanup on paper, but caused
`ex03_corotational_v4.py` to cutback-cascade down to `dt≈9e-7` near `t=0`
and stall — the stricter AND condition is much harder to satisfy than the
original OR combination for this problem's near-zero-deformation start.
**Reverted.** The one genuine bug found and kept fixed: `max_R_val` /
`max_du_val` (needed by the Abaqus-style branch) were previously computed
**only inside `if verbose:`**, so that branch was silently dead whenever
`verbose=False` — now computed unconditionally.

### 4.8 [CRITICAL, silent] `penalty_constraints` (surface ties) and `rbe2_dof_map` never applied for batch assembly paths — display was completely frozen

`DynamicSolver._assemble()` has an if/elif chain that dispatches to one of
several assembly strategies (`use_jax_vmap`, `use_j2_batch`,
`use_multi_material_batch`, `use_jax_grouped_vmap`, sequential fallback).
The tie-penalty application (`self.penalty_constraints`, i.e.
`SurfaceTieConstraint`) and RBE2 kinematic-condensation DOF elimination
(`self.rbe2_dof_map`) live in **one shared block at the very end of
`_assemble()`**, after the if/elif chain. But the `use_j2_batch` and
`use_multi_material_batch` branches used `return self._assemble_j2_batch(...)`
/ `return self._assemble_multi_material_batch(...)` **directly** — an early
return that skips the shared tail entirely. Any model using a multi-pid /
multi-material setup (i.e. `ex11_rigid_plate_display_fold.py`,
`ex12_rigid_plate_display_fold_corotational.py`, or any future model
combining a display + rigid plates) got **zero tie force and zero tie
stiffness, silently**, for the entire run.

**Symptom**: Newton converged perfectly (small residuals, no cutbacks,
"Increment Converged" every step) and the *rigid plate* rotated exactly as
prescribed — but the *display*'s displacement was **exactly `0.0` at every
node, every step**, because nothing in the assembled system ever coupled
it to the moving plate. This is easy to miss from convergence output alone
(there is no error — non-convergence — to see); it only becomes visible by
plotting the deformed shape (a visible gap opens between plate and
display) or, better, by comparing displacement magnitudes between the two
regions programmatically (see `diagnostics.py` below).

**Fix**: changed the two branches to assign `f_int, K_T, state_new = self._assemble_j2_batch(u, dt)` /
`self._assemble_multi_material_batch(u, dt)` instead of returning directly,
so execution falls through to the shared tie/RBE2 tail. Verified: display
displacement went from `sum(|u|)=0.0` to tracking the plate's motion
(comparable max displacement), full test suite still 142 passed (3
pre-existing unrelated `test_rbe2.py` INP-parser failures, not touched by
this fix).

**If you add a new assembly branch to `_assemble()` in the future, do NOT
`return` from inside the if/elif chain** — assign to `f_int, K_T,
state_new` and let control fall through to the tail, or you will silently
reintroduce this bug for that branch.

### 4.9 Convergence success is not proof of physical correctness — use `diagnostics.py`

The bug in §4.8 converged cleanly for dozens of steps while being
completely wrong. **A model with a driving region (rigid plate, prescribed
motion) coupled to a driven region (display, tied/constrained) needs an
explicit runtime check that the driven region is actually responding**, not
just "did Newton converge." `dispsolver/solver/diagnostics.py` provides:

- `check_region_tracking(u, driving_dofs, driven_dofs, ...)` — flags when
  a driven region has moved by less than a threshold fraction of the
  driving region's motion.
- `check_tie_gap(u, coords, tie_constraint, nid_to_idx)` — computes the
  actual penalty-spring gap for a `SurfaceTieConstraint`'s pairs; a gap
  comparable to the imposed motion itself means the tie isn't constraining
  anything.
- `check_hinge_rotation(...)` — verifies a master/slave node pair's
  current relative angle actually matches the prescribed hinge theta.
- `sanity_report(...)` — combines the above, meant to be called once per
  step in a time-stepping loop; prints `[WARN]`/aborts the run the moment
  something looks physically wrong rather than after a multi-minute run.

`examples/ex12_rigid_plate_display_fold_corotational.py` calls
`sanity_report(...)` every step and `break`s on failure — use this as the
template for any new plate+tie+display example. **When adding a new
coupled subsystem (a new tie, a new rigid part), add its
driving/driven DOF groups to the sanity check rather than trusting
convergence output alone.**

### 4.10 [CRITICAL] `RBE2HingeElement` was already tried and abandoned for driving the plate — don't reintroduce it

**Project history** (from the user, not discoverable from the code alone):
`RBE2HingeElement` (`dispsolver/element/rbe2.py`, a penalty +
Augmented-Lagrangian rigid link, see its own docstring) was the **first**
mechanism tried for rotating the display and it **did not work**. That
failure is *why* `RigidBodyPart` (`dispsolver/part/rigid_body.py`, exact
kinematic rigid-body motion: `u_s = u_m + (R(theta)-I)@d0`, applied as a
plain Dirichlet BC — no penalty, no solved DOF, no gap by construction)
and `SurfaceTieConstraint` were introduced in the first place.

`ex11_rigid_plate_display_fold.py` and the pre-2026-07-25-late version of
`ex12_rigid_plate_display_fold_corotational.py` both used
`rbe2_elements=[rbe2_left, rbe2_right]` (`RBE2HingeElement`) to drive the
plates — a **regression back to the already-abandoned approach**, likely
introduced in an earlier session that didn't know the history above.

**Symptom this caused**: while diagnosing §4.8/4.9-era tie behavior, the
tie's gap-to-displacement ratio (`check_tie_gap`) was found to saturate
around ~40% **completely independent of `penalty_stiffness`** (tested
1e2 through 1e10, 8 orders of magnitude, near-identical trajectories, and
Newton iteration count stayed flat at 3 regardless of `k` — ruling out
ill-conditioning as the cause). Root cause: `RBE2HingeElement`'s own
finite penalty stiffness on the plate's rigid-body constraint meant the
plate itself never became genuinely rigid, so the tie was chasing a
moving, non-exactly-rigid target — no amount of tie stiffness could fix
that, because the compliance wasn't in the tie.

**Fix**: rewrote `ex12_rigid_plate_display_fold_corotational.py` to
remove `RBE2HingeElement`/`rbe2_elements` entirely. Instead, every plate
node's displacement is computed directly each step via
`RigidBodyPart.get_slave_displacements(u_master, theta)` and applied as a
Dirichlet BC (`solver.set_prescribed_dofs(...)` once at setup,
`solver._bc_base_vals = ...` updated per step — the same pattern
`ex03_corotational_v4.py` already used, just applied to a separate plate
part's node set instead of directly to display nodes). With this fix and
`penalty_stiffness=1e5` (unchanged, it was never the problem), tie gap
dropped to `~1e-5 mm` at `max|u| ~ 1mm` (ratio ~1e-5) — the tie is now
genuinely tight. **`ex11_rigid_plate_display_fold.py` still uses the old
`RBE2HingeElement` approach and should be treated as a known-bad
reference, not a baseline to compare against** — its NeoHookean-vs-
corotational comparison numbers from earlier in this session are invalid
for the same reason.

**If you are asked to drive a rigid part's motion in this codebase: use
`RigidBodyPart.get_slave_displacements()` as a Dirichlet BC. Do not use
`RBE2HingeElement`/`rbe2_elements=` for this — it has already been tried
and does not work for the display-folding problem.**

### 4.11 [PERFORMANCE, CRITICAL] `PyPardisoSolver()` was reconstructed every Newton iteration — 85% of wall time wasted on a filesystem glob scan

`_solve_linear_system()` (`dynamic.py`) called `from pypardiso import
PyPardisoSolver; p_solver = PyPardisoSolver()` **fresh on every single
call** (i.e. every Newton iteration, every step). `PyPardisoSolver.__init__`
does an expensive filesystem `glob.glob()` search (looking for the MKL
runtime), completely independent of matrix factorization. Profiling a
single `solve_step()` on the (small, 1186-DOF) `ex12` model showed
**8.47s of a 9.99s call (85%)** spent inside this `__init__`'s glob/scandir
machinery (`nt.scandir` called ~22,754 times). This is separate from
(and much larger than) the earlier B2 "PARDISO factorization reuse" fix
— that one only reused `factorize()`/`solve()` *within* a single
`_solve_linear_system()` call; the `PyPardisoSolver` Python object itself
was still being thrown away and rebuilt every call.

**Fix**: added a process-wide cached singleton (`_get_pardiso_solver()`
in `dynamic.py`) so `PyPardisoSolver()` is constructed exactly once per
process; every `_solve_linear_system()` call reuses it (still calling
`factorize()` fresh each time, which is correct/necessary since `Js`
changes every Newton iteration — only the expensive `__init__` is
avoided). **Effect: full `pytest tests/` runtime dropped from 410.85s to
151.90s (63% reduction)**, same 142 passed / 3 pre-existing-unrelated
failed. Do not reintroduce a per-call `PyPardisoSolver()` construction.

**Known remaining performance issue (not yet fixed, documented for
whoever picks this up next)**: with the plate now fully Dirichlet-BC
driven (§4.10), `ex12`'s plate elements (pid=2, plain `Q4` + `NeoHookean`,
480 elements) still get assembled through the slow sequential
`_element_contributions()` Python loop on *every* Newton iteration, even
though every plate DOF is prescribed and that assembly work cannot
affect the free-DOF solve at all. Profiling after the PARDISO fix showed
this costing **4.18s of 7.18s (58%)** of a single (warm) Newton
iteration on `ex12`. The clean fix would be for `DynamicSolver` to skip
element assembly for any element whose every DOF is currently
prescribed — not implemented in this session because it touches the
shared multi-pid assembly path (`_assemble_multi_material_batch`) used
by every multi-material model, and needs its own careful
verification pass.

### 4.12 [SOLVED] Tip-element inversion at ~47.9°/side — graded mesh, not tie/RBE2 stiffness

A different display-folding path, `examples/ex12_abaqus_inp_plate_fold.py`
(reads mesh/materials/constraints from an Abaqus `.inp` deck, generated by
`examples/gen_ex12_inp.py`, using **exact RBE2 kinematic condensation**
for the plate — `dispsolver/constraint/rbe2_condensed.py`, zero
compliance — plus a `theta_penalty_k`/`theta_targets`-driven rotation, and
a plain penalty `SurfaceTieConstraint` coupling the display) reached
47.9°/side with clean 1-3-iteration convergence, then hit genuine element
inversion (`det(F) ≈ -0.10`) in the display's two outermost free-edge
elements (x∈[-40,-39] and x∈[39,40]).

**Root cause**: with the plate *exactly* rigid (condensation, zero
compliance) and the display only elastically/penalty-tied to it, the
stiffness mismatch is absorbed as shear strain. Interior elements share
that strain with neighbors on both sides; the two outermost tip columns
have no outboard neighbor, so it all concentrates into one element each —
classic edge-singularity behavior. **Counterintuitive but important**:
stiffening the tie (or adding Augmented Lagrangian to make it exactly
rigid too) makes this *worse*, not better — it forces the display bottom
row even more tightly onto the rigid plate line, concentrating *more*
shear into the tip element. (Don't confuse this with §4.10's "tie gap
independent of stiffness" finding — that was under the abandoned
penalty+AL `RBE2HingeElement`, where the plate itself still had
compliance; here the plate is exactly rigid, so tie stiffness matters,
in the opposite direction from what you'd guess.)

**Fix**: mesh discretization, not solver/constraint code. `gen_ex12_inp.py`
now has `_graded_display_x()`, clustering the display's x-columns near
the four transition locations — the free tips (`x=±40`) and the
plate/hinge boundary (`x=±10`) — down to 0.25mm spacing, with 0.5mm
through the free hinge span and 1mm under the rigid plate body (up from a
uniform 80-column 1mm grid to ~120 graded columns). Spreading the
shear/rotation gradient across more, narrower elements keeps
`det(F) > 0` all the way to full closure. **No solver or constraint code
was changed for this fix.** Also fixed in passing: the `dt <= dt_ctrl.dt_min`
guard in `ex12_abaqus_inp_plate_fold.py` (`np.clip` in
`AdaptiveDtController.update()` means `dt` never goes *strictly* below
`dt_min`, so a `<` guard never fires and a residual failure becomes an
infinite cutback loop instead of a clean abort).

**Result**: full 90°/side (180° combined), 101 steps, **zero cutbacks**,
1 Newton iteration per step throughout, ~85s wall time, `n_inverted == 0`
end to end — see §1.0's "Achieved" note.

**If a future mesh/example hits inversion at a rigid-plate/compliant-tie
or rigid-plate/compliant-display boundary, try mesh grading toward that
boundary before reaching for tie-stiffness changes** — this session's
evidence is that stiffness changes there are counterproductive.

---

## 5. Where things live

```
dispsolver/
  element/
    q4_eas.py, q4_eas_jax.py       # EAS Q4, Simo-Rifai, NumPy + JAX
    q4_corotational_jax.py         # Co-rotational Q4 (+ J2 plasticity variant)
    rbe2_jax.py, rbe2.py           # RBE2 hinge element(s)
  material/
    plastic.py, plastic_jax.py     # J2 plasticity (finite-strain, multiplicative)
    viscoelastic.py                # Prony + WLF
  constraint/
    rbe2_condensed.py              # Kinematic condensation manager (SPD reduction)
    surface_tie.py                 # Penalty surface-to-surface tie
    spring_hinge.py
  part/
    rigid_body.py                  # RigidBodyPart (master RP + slaves, dual-mode)
  mesh/
    plate_builder.py               # create_folding_plate_parts()
  solver/
    dynamic.py                     # DynamicSolver — main Newton-Raphson driver
    dt_controller.py                # AdaptiveDtController
    stabilization.py                # AbaqusViscousStabilization
    diagnostics.py                  # sanity_report() etc. — physical (not just convergence) runtime checks, see §4.9
examples/
  ex03_corotational_v4.py          # simplified hinge-only-BC validation
  ex11_rigid_plate_display_fold.py # KNOWN-BAD reference only -- uses RBE2HingeElement to drive the plate, already shown not to work (see §4.10). Do not use as a baseline.
  ex12_rigid_plate_display_fold_corotational.py  # Python-built model, RigidBodyPart exact Dirichlet BC + surface tie; includes sanity_report()/fold_success_verdict() wiring, but still stalls ~7.9deg/side (unfixed, see 1.2/4.11)
  ex12_abaqus_inp_plate_fold.py    # PRIMARY REFERENCE -- reads .inp deck, exact RBE2 condensation + theta_penalty drive; reaches full 90deg/side / 180deg combined, zero cutbacks (see 1.0/4.12)
  gen_ex12_inp.py                  # generates the .inp deck ex12_abaqus_inp_plate_fold.py reads; _graded_display_x() is the tip-inversion fix (4.12), edit here not in the solver for mesh-resolution issues
dev_log/                            # dated work logs — check for the most recent status before assuming something is/isn't done
```
