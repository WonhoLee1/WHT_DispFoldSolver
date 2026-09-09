# Handoff addendum (2026-09-08) — SRI sign bug, node-id overflow, CPE4I/CPE4H Numba

Continuation of `dev_log/handoff_element_ul_numba_20260907.md`. Two real,
independent bugs found and fixed; two new Numba kernels completed and wired
behind an opt-in flag pending a clean full-fold confirmation run.

## 1. Fixed: transposed rotation matrix in the new SRI Numba kernel

`dispsolver/element/q4_corotational_sri_j2_numba.py::_element_rotation` built
`R = [[c, s], [-s, c]]`. With the row-vector convention used downstream
(`coords_curr @ R`), that is `R^T`, i.e. `R(+theta)` applied a second time
instead of the pull-back `R(-theta)`. A pure rigid rotation therefore produced
spurious local displacement growing linearly in theta (`max|u_l| ~ 2x` the
element size at 90 deg, instead of exactly 0), which fed nonsense strain into
the J2 return map. This is what produced `Max Res.Force = 4.19e+25` on the
very first real (rotated) Newton iteration — invisible in the original
single-element validation because that test used small strains without an
isolated pure-rotation case.

Found by an `oh-my-claudecode`/fable deep-dive review, confirmed by direct
canary test (rigid rotation at several angles -> `|f|` must be ~0), fixed by
swapping to `R = [[c, -s], [s, c]]` (matching the already-correct sibling
kernels `q4_sri_numba.py` / `q4_sri_hybrid_numba.py`, which this new kernel's
`R` had accidentally transposed relative to). Re-validated: canary now ~1e-13
to 1e-18 at every angle; accuracy vs the JAX reference unchanged (force error
~1e-13 on stretch/shear/bend, including combined rotation+stretch).

## 2. Found and guarded: display mesh node-id overflow into the reserved plate range

**Root cause of the mesh-level Numba blow-up that was chased for hours after
fix #1** (a *second*, unrelated defect on the same production `.inp`):
`base_node_id=10000` in `create_folding_plate_parts` (called from
`examples/gen_ex12_inp.py`) is a hardcoded convention several files rely on to
tell display nodes (`nid < 10000`) apart from plate nodes (`nid >= 10000`):
`dispsolver/export/plotter.py`, `dispsolver/solver/dynamic.py:~2307`
(`max_displacement_corr` plate-DOF exclusion), `dispsolver/postprocess/interlayer.py`,
`examples/check_interlayer_shear.py`, `examples/ex12_abaqus_inp_plate_fold.py`'s
own diagnostics added this session.

This session's mesh refinement for `CPE4H` accuracy (PSA layer at 2 element
rows instead of 1, free-span `dx` from 0.25 to 0.05 across the whole untied
span) pushed the **display mesh's own node numbering past 10000** (max display
node id 17130 in the refined config). Past that point, display elements
started silently reusing plate node ids: e.g. element 9600's connectivity
`[9618, 9619, 10220, 10219]` cited real nodes 10219/10220, which the parser
correctly resolves to the LEFT PLATE's actual grid nodes near `x = -40` (the
plate builder's own numbering, unaware the display had already claimed part
of that range) instead of adjacent display nodes near `x = 40`. The resulting
element spans up to 44 mm (should be ~0.25 mm) -- **721 such elements**, all
in pid 7/8 (the two families whose element type was being tested).

Symptom that made this look Numba-specific: the JAX kernel's existing
`elem_nan = isnan(f)|isnan(K); f = where(elem_nan, 0, f)` guard silently
zeroed these 721 garbage elements' contribution (so JAX's residual stayed
numerically "sane" -- 893 at the same state where Numba, which has no such
guard, reported 7.45e+26). **This means the 721 malformed elements have been
contributing exactly zero stiffness/force under JAX all along** whenever this
overflow condition is hit -- not a Numba defect, a pre-existing generator gap
now actually exercised because someone (this session) finally used a mesh
fine enough to trigger it. Confirmed present regardless of backend by
constructing the same degenerate element and evaluating it directly.

**Fix applied**: not a numbering change (raising `base_node_id` would desync
the ~6 files above that hardcode 10000 as the display/plate boundary, trading
one silent bug for another under time pressure). Instead, `gen_ex12_inp.py`
now asserts `max(display_node_id) < 10000` right after building the display
grid, with a message pointing at this file. **This mesh (PSA x2 rows + fine
free-span dx) currently fails that assert and cannot be generated as-is** --
coarsen it (fewer PSA rows, or a larger free-span dx) before reusing it, or
raise `base_node_id` in gen_ex12_inp.py AND every file listed above together.
Production config was rolled back to PSA x1 row / `dx=0.25` (6076 elements,
`max nid = 10611`, confirmed zero degenerate elements) with `CPE4I` (accurate
at that coarser AR, see below) instead of `CPE4H`.

**Action item for whoever revisits `CPE4H` in production**: this assert is
the reason it is not the current default. Either budget the node-id headroom
properly (touch all ~6 files) or find a way to get `CPE4H` accurate without
2 rows through the PSA layer before re-enabling it.

## 3. New: CPE4I Numba kernel (`dispsolver/element/q4_visco_eas_numba.py`)

Ported `q4_visco_eas_jax.compute_single_eas` (incompatible modes) to Numba.
No analytic material tangent exists for the Numba viscoelastic path (documented
in `q4_visco_hybrid_simo_numba`: a material-only tangent was ~50% off for this
near-incompressible material), so this uses a **nested finite difference**:
an inner fixed 5-iteration Newton on the 4 enhanced parameters (its own 4x4 FD
Jacobian), producing a condensed force at each of the 8 outer perturbations of
`u`; the outer 8x8 FD tangent of those already contains the
`-K_ua K_aa^-1 K_au` term because alpha is re-equilibrated at every
perturbation. Alpha is bounded with the same p-norm guard used in the JAX
kernel after review found the original `tanh` biases the solution.

Updated-Lagrangian support (`F_n` per GP) added to match the JAX kernel, which
also gained it this session -- `compute_single_eas`/`compute_single_hybrid`
now both return `F_n_new` as their 5th value.

Validated against JAX: rigid-rotation canary (this element is Total-Lagrangian
with the enhancement carrying rotation through `F`, not a co-rotational frame,
so this specific bug class is structurally impossible here, but the canary is
kept as a standing regression guard) gives `|f| ~ 1e-18` to `1e-20` at every
angle including with UL carry-through; force error vs JAX is 3e-14 (shear) to
2e-5 (bend, rotation+bend). Wired into `dynamic.py`'s batch dispatch behind
`enable_new_numba_elements` (same flag as `CPE4H`/`SRI`), alongside a Numba
`assemble_eas_visco_batch_numba` batch assembler.

## 3b. Fixed: shared viscoelastic material kernel crashed on near-singular C (real production crash)

After fixes #1/#2, the corrected coarse-mesh production run reached step 6
(4.25% fold) before a `ZeroDivisionError` inside `assemble_eas_visco_batch_numba`
crashed the whole process (surfaced by Numba's `prange` exception plumbing as
`SystemError: ... returned a result with an exception set`, one layer removed
from the real error). Root cause: `_simo_pk2_numba` (in
`q4_visco_hybrid_simo_numba.py`, shared by ALL THREE Numba viscoelastic
kernels -- `Q4_VISCO_SIMO`/`Q4_UP`, `CPE4H`, and now `CPE4I`) computed
`Cinv = np.linalg.inv(C)` with no regularisation, while its own JAX reference
(`q4_visco_simo_fs_jax._simo_pk2`) uses `jnp.linalg.inv(C + 1e-15*eye(3))`.
A line-search TRIAL state (rejected afterwards by the inversion check, but
evaluated first to know whether to reject it) had a locally near-singular
`C`, which JAX handles gracefully (huge-but-finite inverse) and the Numba
port did not (hard exception, killing the whole run instead of letting the
line search backtrack).

Fixed with the same `+1e-15*eye(3)` JAX uses. Additionally added the
NaN/Inf-zero output guard every JAX kernel already has (`bad = any(!isfinite(f))
or any(!isfinite(K)); if bad: f,K = 0`) to `q4_visco_eas_numba.py`,
`q4_visco_hybrid_up_numba.py`, and `q4_corotational_sri_j2_numba.py` as a
second, independent layer of defense (catches value-based garbage even where
no exception is raised). Verified: canary/accuracy unchanged after the fix,
and a deliberately near-collapsed element (`u` compressing element width to
~0) now returns a finite (large but non-crashing) force instead of raising.

**Lesson for anyone porting another JAX kernel to Numba in this codebase**:
diff every `jnp.linalg.inv`/`solve` call against its Numba port line by line
for a dropped regularisation term -- this is now the second time in one
session a hand-transcription silently dropped a stability term that only
matters away from the identity/near-identity states the initial validation
tests exercise (the first was the SRI rotation-matrix transpose, #1 above).
The single-element validation tests that passed before both fixes never
included a genuinely near-degenerate state, which is exactly the regime a
real Newton line search visits routinely and safely (by design) in every
other kernel.

## 3c. NOT fully fixed: a second, unidentified division-by-zero still crashes CPE4I Numba

After the Cinv fix (#3b), the SAME production scenario (step 6, 4.25% fold,
the line-search-collapse trial state) crashed again with the identical
`ZeroDivisionError` -> `SystemError` signature inside
`assemble_eas_visco_batch_numba`. The Prony recurrence's small-`ratio` guard
(`if ratio < 1e-12: ...`) IS present in the Numba port (checked directly,
matches JAX) so that is ruled out. Not yet isolated further -- this needs a
**direct single-element repro** (capture the exact `coords`/`u_elem`/`F_n`/
`state` at the crash point via a monkeypatch or a saved checkpoint, the way
`scratch/probe_cpe4h_tip.py` did for bug #3b) rather than re-running the full
20-40-step production loop per iteration (each JAX-warmup + Numba-compile
cycle costs real minutes).

**`enable_new_numba_elements` remains `False` by default at all three gate
sites in `dynamic.py` (SRI, CPE4H, CPE4I) -- confirmed.** This means:
- The SRI sign-bug fix (#1) and the Cinv regularisation fix (#3b) are real,
  verified-correct, and worth keeping, but **do not affect any run that
  doesn't explicitly opt in** (`solver.enable_new_numba_elements = True`, as
  `scratch/run_full_numba.py` does). Every JAX-path run this whole session
  (09-07 and 09-08) was unaffected by either bug or either fix.
- CPE4I-via-Numba is **not safe to enable** until the remaining
  division-by-zero is found. CPE4H-via-Numba was never re-tested after the
  mesh got rolled back to the config that doesn't use it (production PSA is
  `CPE4I` again, see #2) -- treat it as equally unverified against a real
  line-search-collapse trial until someone re-runs it under the same stress.
- The JAX-based CPE4I/CPE4H/SRI kernels (the actual default path) have no
  known correctness issues from today's session; only their Numba ports do.

## 4. Current state / next step

- `enable_new_numba_elements` default `False` still applies -- both new
  kernels (CPE4H, now also CPE4I) and the fixed SRI kernel are opt-in.
- Production config: PET/GLASS `Q4_COROTATIONAL_SRI` (Numba, fixed), PSA
  `CPE4I` (Numba, new, validated), coarse mesh (6076 elements, no degenerate
  elements). A full run with `enable_new_numba_elements=True` was launched
  (`scratch/full_numba_prod2.log`, driven by `scratch/run_full_numba.py`) to
  confirm a sane Newton residual end-to-end on the corrected mesh -- check
  that log first thing next session if it is not already resolved.
- Still open from the 09-07 handoff and not touched today: `Q4_EAS` (plain,
  non-corotational) batch UL gate -- the single-element Numba kernel accepts
  `F_n` but does not return `F_n_new`, and the batch assembler does not thread
  `F_n` through at all. Not production-blocking (current PET config is SRI,
  not plain EAS) but left as a real gap for "all elements support UL" if
  anyone picks it up.
- Corotational-EAS Numba (`Q4_COROTATIONAL_EAS`) was not ported -- that
  element already failed to *converge* at the production aspect ratio before
  any Numba work (09-07 finding), so a faster (Numba) version of a formulation
  already known not to work there is low priority.
