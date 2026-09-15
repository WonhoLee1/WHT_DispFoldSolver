# Penalty-regularized Coulomb friction -- Stage 1 implementation (2026-09-16)

Implements `dev_log/contact_friction_precise_design_20260916.md` (Phase 3
of the contact roadmap, priority item #3 of `dev_log/contact_development_
report_20260915.md`, picked up per "계속해서 진행" after items 1 (surface-
to-surface) and 2 (nonlinear/soft laws for deformable contact) were
completed). Scope, per the design's own recommendation: friction added to
`SurfaceContactConstraint3D` (rigid plane) and
`DeformableSurfaceContactConstraint3D` (5-node stencil).
`SurfaceToSurfaceContactConstraint3D` (dual-mortar) stays frictionless, as
that class's own docstring already states.

## What was built

Exact port of the design doc's mechanism:

- `friction_coefficient: float = 0.0` (0 = frictionless, exact byte-
  identical fallback), `tangential_stiffness_ratio: float = 1.0`
  constructor params on both classes.
- `k_t = tangential_stiffness_ratio * k_contact` -- reuses the already-
  auto-derived normal stiffness, no separate magic number.
- `_friction_force_and_tangent()`: the Coulomb stick/slip return map
  (`|s|<=s_crit`: `f_t=k_t*s`, `Ktt=k_t*I`; `|s|>s_crit`: `f_t=mu*p_i*ŝ`,
  `Ktt=(mu*p_i/|s|)*(I-ŝ⊗ŝ)`, `s_crit=mu*p_i/k_t`) -- a pure function, no
  side effects, shared identically by `assemble()`, `get_friction_state()`,
  and `commit_friction_state()`.
- Friction anchor: a fixed 3D point for the rigid-plane class (built once,
  since the plane never rotates); a CONVECTED `(xi_anchor, eta_anchor)`
  material coordinate on the master face for the deformable class (re-
  evaluated against the current deformed geometry every call, exactly
  like the existing `xi_m, eta_m` projection already is) -- avoids the
  AGENTS.md sec4.14-class defect where a fixed global anchor would misread
  rigid rotation of the master face as spurious slip.
- `get_friction_state(u)`: pure query, mirrors `get_active_set()`'s
  convention -- returns the `frozenset` of currently-slipping node ids.
- `commit_friction_state(u)`: mutates the anchor via a radial return
  (shrinks the stored elastic slip to exactly `s_crit`), called ONLY at
  `solve_step()`'s three existing `update_state=True` sites -- NOT at
  PDASS's/the chattering gate's per-iteration commit points, because the
  anchor is physical slip history (like a plastic internal variable), and
  advancing it on a rejected Newton trial would permanently corrupt that
  history.
- `dynamic3d.py`: `_contact_full_state()` (unions `_contact_active_set()`
  with every constraint's `get_friction_state()`) replaces
  `_contact_active_set()` at the two call sites feeding SDI's `is_sdi`
  comparison; `_commit_contact_friction_state()` wired at the three
  `update_state=True` sites. Both are additive no-ops for every
  constraint/model that doesn't use friction.

## A real bug found and fixed by direct verification, not by re-reading the design

**The design doc's own sec7 pseudocode had the friction force sign
backwards, and this implementation's first cut copied it verbatim.**
Both said `f_contact[...] += -f_t_3d` (matching the ONE-SIDED normal-
contact convention, `f_int = -F_physical`). This is wrong: a STICK/SLIP
tangential spring is TWO-SIDED (works symmetrically for +s and -s), the
same structural category as `surface_tie3d.py`'s own bonded-tie spring,
not normal contact's one-sided complementarity. The tie's own docstring
history already derives the correct rule for a two-sided spring: its
physical restoring force is `F_physical=-k*violation`, so
`f_int=-F_physical=+k*violation` -- the negation cancels, and the tie's
code uses `+k*gap_vec` directly with no explicit negation. The identical
rule applies to friction (`s` plays `gap_vec`'s role): `f_int` needed
`+f_t_3d`, not `-f_t_3d`.

**How it was caught**: NOT by re-deriving the sign on paper again (that's
exactly how the design doc got it wrong the first time -- an abstract
sign argument, without checking it against the class's own established
convention for a TWO-sided spring). Caught by writing a real two-cube
sliding-block test (`tests/test_contact_friction.py::
test_full_solve_sliding_block_stick_then_slip`) that failed to converge
on the very first tangential increment, then root-causing it with a
direct finite-difference check:

```
f_int_x at s=0.005 (STICK):  -0.5      (coded, wrong-signed force)
finite-difference d(f_int_x)/d(u_x):  -100.0
coded stiffness K[0,0]:  +100.0        (Ktt = +k_t*T, coded correctly)
```

The coded tangent (+100) and the force's own finite-difference derivative
(-100) disagreed in sign -- Newton's linear solve was therefore pointing
in the wrong direction for the tangential DOF, which is exactly what
broke convergence on a real solve (a purely "does the physics look
right" review would not have caught this without the finite-difference
check; the force alone looked plausible on paper).

**Fix**: `SurfaceContactConstraint3D`: `f_node = f_node + f_t_3d` (was
`-`). `DeformableSurfaceContactConstraint3D`: `f_vec = f_mag*n - f_t_3d`
(was `+`) -- the sign flips the OTHER way here because this class
combines normal and tangential into one vector before applying the
SAME `-w_a*(...)` distribution pattern to both (the shared weight
already carries the one-sided/two-sided distinction correctly once the
vector's own sign is right). Re-verified by finite difference on BOTH
classes after the fix (`fd == coded K` to 1e-6 in both cases, including
a cross term: the deformable class's master-vertex force derivative
`d(f_int_master)/d(u_slave) = -25.0` matched `K[3,0] = -25.0` exactly).

**This is precisely the class of defect this project's own AGENTS.md
sec4.16-4.18 methodology exists to catch** ("a contract can be masked by
another contract's failure" / "verify by refinement, not by assumption")
-- a plausible-looking design doc formula, copied into code, that looked
right until a real nonlinear solve was actually attempted and finite-
differenced.

## Verification (`tests/test_contact_friction.py`, 10 tests, all passing)

1. `test_stick_slip_switch_matches_closed_form` / `test_stick_slip_force_
   continuous_at_switch`: the STICK/SLIP formula against the exact closed
   form, and the algebraic force-continuity claim at the switch.
2. `test_friction_disabled_is_exact_zero_fallback` /
   `test_friction_requires_active_normal_contact`: the off-by-default and
   "no normal contact -> no friction" guards.
3. `test_anchor_unaffected_by_trial_evaluation` / `test_commit_friction_
   state_radial_return`: the design's single most important decision
   (sec5) -- the anchor never moves except via an explicit
   `commit_friction_state()` call, and that call performs exactly the
   radial-return arithmetic (remaining elastic slip == `s_crit`, not 0
   and not the full trial slip).
4. `test_frame_consistency_pure_rotation_no_spurious_slip` /
   `test_frame_consistency_real_slip_still_detected`: the convected-
   anchor fix -- a pure rigid rotation of the master face (zero real
   slip) measures `s_mag<1e-8` at every rotation angle up to 180°, while
   a genuine relative displacement is still correctly detected.
5. `test_full_solve_friction_disabled_matches_frictionless_baseline` /
   `test_full_solve_sliding_block_stick_then_slip`: a real
   `DynamicSolver3D.solve_step()` two-cube fixture -- friction off is
   unaffected; friction on reproduces the qualitative stick-then-slip
   signature (reaction force tracks `k_t*s_applied` below `s_crit`,
   plateaus at `mu*p_i` above it) end to end, through the real
   `_contact_full_state()`/SDI/three-commit-site wiring, not just direct
   calls.

Full contact regression family (chattering stabilization, PDASS/AL,
Phase 1, surface-to-surface, deformable soft/nonlinear, this file):
**41 passed**, zero regressions. A quick non-contact 3D element smoke
check (`test_3d_c3d8r.py`, `test_3d_c3d4_anp.py`, `test_3d_c3d10m.py`,
`test_3d_c3d6.py`): 17 passed, 1 failed -- the failure
(`test_c3d4_anp_tangent_consistency`) is a pre-existing, unrelated
failure already confirmed via `git stash` earlier this session.

## What is still open (design doc sec9.2, unchanged)

- `SurfaceToSurfaceContactConstraint3D` friction -- explicitly deferred,
  a materially harder mortar-friction design problem.
- A dedicated stick/slip chattering gate (analogous to
  `chatter_stabilization`) -- not built; the design's own staging
  discipline says build only if measurement on a real fixture shows the
  plain return-map chatters (not attempted in this pass; the sliding-
  block test above did not chatter, but that fixture wasn't designed to
  stress-test this specifically).
- Un-freezing the normal<->tangential coupling term
  (`d(f_t)/d(p_i)*d(p_i)/d(u_normal)`) -- accepted modified-Newton
  simplification, matching precedent elsewhere in this codebase.
- Anisotropic friction, rate/pressure/temperature-dependent `mu`, a
  `tau_max` shear-stress limit -- real Abaqus features, not requested,
  not built.
- No CAE-level (`ContactPair`/`interaction.py`) exposure for friction yet
  -- both classes are usable directly (as this test file does); wiring
  `friction_coefficient`/`tangential_stiffness_ratio` through
  `build_runtime_constraint()` is a separate, small follow-on task,
  mirroring exactly how `chatter_stabilization`/`hysteresis_band` were
  exposed there.
