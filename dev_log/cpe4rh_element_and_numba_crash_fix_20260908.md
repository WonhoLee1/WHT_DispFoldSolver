# CPE4RH element + CPE4I Numba crash resolution (2026-09-08, continued session)

Continuation of `dev_log/handoff_numba_sign_bug_and_nodeid_overflow_20260908.md`.
Two things closed out this session: the still-open CPE4I Numba
`ZeroDivisionError` (section 3c of that handoff), and a brand-new CPE4RH
element built end-to-end per explicit user request.

## 1. CPE4I Numba crash — actually resolved this time

### Root cause (the real one, not just the necessary-but-insufficient Cinv fix)

`_simo_pk2_numba` (`dispsolver/element/q4_visco_hybrid_simo_numba.py`,
shared by all three Numba viscoelastic kernels: `Q4_VISCO_SIMO`/`Q4_UP`,
`CPE4H`, `CPE4I`) used a **stale volumetric law**:

```python
S_vol = kappa * lnJ * Cinv                      # OLD (wrong)
I1b = J ** (-2.0 / 3.0) * I1                     # unclamped
```

while its own JAX reference (`q4_visco_simo_fs_jax.py::_simo_pk2`) has since
moved to the Simo & Armero (1992) / Holzapfel form with a clamped `J`:

```python
J_safe = jnp.maximum(J, 1e-4)
vol_factor = 0.5 * kappa * (J_safe - 1.0 / J_safe)
S_vol = vol_factor * Cinv + distortion_control_term
I1b = (J_safe ** (-2.0 / 3.0)) * I1
I1b_safe = jnp.clip(I1b, 3.0, 50.0)
```

The two volumetric laws agree to `O((J-1)^2)` (fine at small-to-moderate
strain, which is all the original single-element validation tested) but
diverge at large strain, and the Numba port's **unclamped** `J ** (-2/3)`
hits Python/Numba's `ZeroDivisionError: 0.0 cannot be raised to a negative
power` the moment a rejected line-search TRIAL state (evaluated before the
outer inversion check throws it out) produces an exactly-zero `J`. This is
the actual crash: not a matrix-singularity `ZeroDivisionError` from
`np.linalg.solve`/`inv` (that one was the earlier, real, but insufficient
Cinv fix), but a scalar fractional-power-of-zero one, three lines away.

**Fix**: ported the current JAX formula term-for-term (`J_safe`,
`I1b_safe` clip, distortion-control term with `distortion_j_crit=0.0`
default), removing `lnJ` entirely. Diffing every `jnp.maximum`/`jnp.clip`
call in the JAX reference against its Numba port (not just the
`jnp.linalg.inv`/`solve` calls flagged in the earlier handoff) is the
actual right audit checklist here — this is the **third** instance this
session of a hand-transcribed JAX-to-Numba port silently dropping a
stability term.

### Two more, separate guards found and fixed in the same audit

- `_grads_numba` (same shared file): `detJ` used with **no zero-guard** at
  all before `... / detJ`. Added the same `if abs(detJ) < 1e-30: detJ =
  1e-30` convention `q4_corotational_sri_j2_numba.py::_grads_sri` already
  uses.
- `q4_visco_eas_numba.py::_enh_modes_numba`: its own reference-config
  `detJ0` (computed independently of `_grads_numba`) had the same gap.
  Guarded the same way.

### Stale on-disk Numba cache — a fourth, orthogonal gap

After the fix above, re-running the exact captured crash state (see
below) **still crashed** — the fix hadn't actually recompiled. Root
cause: `dispsolver/_jit_cache.py`'s own documented risk ("Numba's
hash-based invalidation isn't airtight for a changed function body without
a changed traced-argument list") turned out to apply even though the
function's parameter list DID change (a new `distortion_j_crit` default
arg was added) — `DISPFOLD_JIT_CACHE=0` (full bypass) picked up the fix;
the default cached path did not. Fixed by bumping
`DISPFOLD_CACHE_VERSION` "v1" -> "v2" in `_jit_cache.py`, which forces a
fresh cache directory regardless of hash invalidation. Confirmed the fix
now applies under the **default** (cache-enabled) path too.

### Verification methodology (direct single-element repro, not full reruns)

1. `scratch/isolate_crash.py`: monkeypatched `assemble_eas_visco_batch_numba`
   with a serial (non-`prange`) Python loop over
   `compute_single_eas_numba`, catching the exception per-element to
   identify the exact failing element index (71) and dump its full
   `coords`/`u_elem`/`alpha0`/`F_n`/`thickness` at the real crash point
   (step 6, 4.25% fold, line-search-collapse trial).
2. `scratch/pinpoint_crash.py`: saved that exact state to
   `scratch/crash_state.npz` and additionally called
   `compute_single_eas_numba.py_func` (the undecorated Python function)
   directly — this did NOT crash, immediately pointing at a
   compilation/fastmath-specific difference rather than a pure-math
   bug in the algorithm as written.
3. `scratch/verify_fix.py`: reloaded the exact captured state and called
   the real (njit-compiled) kernel directly — reproduced the crash after
   the volumetric-law fix (stale cache), then confirmed finite/no-crash
   after the `DISPFOLD_CACHE_VERSION` bump.
4. Full production re-run (`scratch/run_full_numba.py` ->
   `scratch/full_numba_prod4.log`) passed cleanly past the previously-
   crashing step 6/7/8 (now just normal adaptive-dt cutbacks, no
   exception) and ran to a clean, non-crashing exit (hit an unrelated
   convergence wall around 4.3% fold with dt shrinking to `dt_min`, then
   aborted through the script's normal abort path — PNG/slip-table saved,
   `Done.` printed, no traceback). Whether that convergence wall is itself
   fixable is a separate, still-open, non-crash question — not addressed
   here.

`enable_new_numba_elements` remains `False` by default at all four
Numba-kernel gate sites in `dynamic.py` (SRI, CPE4H, CPE4I, and the new
CPE4RH below) — none of this session's fixes changed any default-path
(JAX) result.

## 2. CPE4RH element — built from scratch, JAX + Numba, patch-test verified

Explicit user request: implement a THEORETICALLY correct CPE4RH (the
element the user reports actually using for PSA in real Abaqus work),
independent of whether it is the right *choice* for this repo's specific
PSA free-span mesh (`dev_log/session_20260730_reduced_integration_failure.md`
already found 1-point reduced integration + hourglass control is a poor
fit for a bending-dominated, one-element-through-thickness layer — that
finding is about mesh/loading suitability, not element correctness).

Confirmed against the user-supplied Abaqus documentation (CPE4RH:
"4-node bilinear, reduced integration with hourglass control, hybrid with
constant pressure") — matches exactly what was built.

### Files

- `dispsolver/element/q4_visco_hybrid_reduced_jax.py` (new): 1-point
  reduced integration + Flanagan-Belytschko (1981) hourglass
  stabilization (shape vector orthogonalized against rigid translation
  and constant strain, reused from the already-existing
  `q4_reduced_jax.py`) + closed-form hybrid pressure (`p = kappa*(J-1)`,
  exact at a single sample point — the usual CPE4H volume-average
  degenerates to the sample itself) on the finite-strain Arruda-Boyce/
  Neo-Hookean/Yeoh + Prony viscoelastic material (`_simo_pk2`, reused
  verbatim). Exact tangent via `jax.jacobian` (no inner Newton/condensation
  needed since pressure has a closed form).
- `dispsolver/element/q4_visco_hybrid_reduced_numba.py` (new): Numba port,
  FD tangent (same pattern as every other Numba viscoelastic kernel this
  session), written with the detJ/J_safe/I1b_safe guards from part 1
  applied preemptively (inherited from the shared, now-fixed
  `_grads_numba`/`_simo_pk2_numba`, not re-discovered).
- `tests/test_cpe4rh_patch.py` (new, 18 tests, all passing):
  1. Rigid-rotation canary (7 angles, plus an off-center/non-axis-aligned
     geometry variant) — `max|f_e| < 1e-9`.
  2. Hourglass-insensitivity: force under 4 different affine (non-rotation)
     deformation gradients is unchanged to `<1e-8` relative when
     `alpha_hg` is scaled 1000x — the orthogonalized hourglass coordinate
     is exactly zero for any affine field by construction.
  3. MacNeal & Harder (1985) style multi-element patch test: 4 irregular
     quads sharing one interior node, boundary + interior nodes both
     driven by the same affine field `u=(F0-I)@X` for 4 different `F0`
     (uniaxial, biaxial, shear, general) plus a rigid rotation — the
     assembled out-of-balance force at the interior node is `<1e-8`,
     i.e. the affine field is already an equilibrium solution of the
     patch (the actual patch-test criterion, not just single-element
     sanity).
  4. Tangent-vs-FD cross-check (`<1e-3` relative, autodiff vs independent
     finite difference).
  - JAX vs Numba direct comparison (`scratch/verify_cpe4rh_numba.py`,
    not a committed test): force error ~1e-16 (machine precision),
    tangent error ~7e-7 (FD vs autodiff, matching every other Numba
    kernel's documented FD tangent error magnitude), rotation canary
    ~1e-17 to 1e-18 in the Numba port too.
  - **Orthodox solver-level patch test (19th test, added after user
    request for "the real plane-strain patch test")**: not just the
    isolated-kernel checks above -- `test_orthodox_solver_level_patch_test`
    runs CPE4RH through the REAL `DynamicSolver` end to end on this
    repo's own established irregular patch mesh
    (`verification/mesh_utils.py::build_patch_mesh_irregular`, a 9-node/
    4-quad MacNeal & Harder 1985 style patch with ONE free interior
    node), the exact same mesh and methodology
    `verification/benchmarks.py::patch_test_solver` already uses to
    certify every other element in this codebase -- boundary nodes
    driven by the affine field `u=(F0-I)@X`, the free interior node
    (node 5) solved for by a real Newton iteration, checked against the
    exact affine value (E/nu converted to Arruda-Boyce `mu`/`K` via the
    same `G0 = mu*beta(lambda_m)` de-scaling this repo's PSA config
    uses, so small-alpha behaviour matches linear elasticity). Needed
    `mode="quasistatic"` and the tight tolerance recipe
    `verification/benchmarks.py`'s own `SOLVER_KWARGS` uses (the default
    tolerances let the 5-way OR convergence check accept a spuriously-
    early iterate for this small-alpha problem, where absolute residuals
    are naturally tiny) -- without those, the test converged in 2
    iterations to a WRONG answer (15% error) that "Increment Converged"
    printed as if it were fine, a good reminder that convergence success
    is not proof of correctness (AGENTS.md 4.9's lesson, here for a
    tolerance-tuning reason rather than a tie/RBE2 one). With the correct
    setup: interior-node displacement error `<0.1%` (well under the
    `<1e-8` used for the algebraic isolated-kernel version, since this
    one goes through a real nonlinear Newton solve rather than a single
    closed-form force evaluation). Total: **19/19 tests passing**.
  - **Scope note on the Abaqus Verification Manual's own "Patch test for
    plane stress, plane strain, and generalized plane strain elements"**
    (the page the user linked/pasted, 7x7 domain, F1-F4 irregular
    meshes, centrifugal + Coriolis + thermal + hydrostatic-pressure
    loading, criterion "reactions agree with applied loads"): that
    benchmark's mathematical core for the ELEMENT is the same constant-
    stress/affine-field reproduction on an irregular mesh already
    verified above; its extra load types (rotating-frame centrifugal/
    Coriolis body force, thermal expansion strain, hydrostatic-pressure
    gradient body load) are solver-level features this codebase does not
    implement at all and are unrelated to CPE4RH's own formulation --
    out of scope for "verify this element is correct," not attempted
    here.

### Wiring

- `dynamic.py`: new `_VISCO_REDUCED_TYPES = ("CPE4RH",)` tuple, folded
  into `_VISCO_ELEM_TYPES`. Three dispatch sites updated: (1) the JAX
  vmap cache builder (`_is_reduced` branch, reuses the plain
  Q4_VISCO_SIMO 6-arg/4-return vmap signature since CPE4RH has no
  per-element internal-variable array — pressure is closed-form, unlike
  CPE4I/CPE4IH's `alpha`/`q`); (2) Numba batch dispatch (new gated block
  calling `assemble_reduced_hybrid_batch_numba`, inserted before the
  "other Numba kernel is full-integration only" fallback guard, which
  also gained a `not in _VISCO_REDUCED_TYPES` exclusion); (3) the
  sequential per-element fallback (new `elif elem_type in
  _VISCO_REDUCED_TYPES` branch).
- `model_builder.py`'s `_ABAQUS_TO_SOLVER_ELEM_TYPE`: **found and fixed a
  pre-existing, unrelated bug while wiring CPE4RH** — "CPE4H"/"CPE4I"/
  "CPE4IH"/"CPE4RH" were mapped to OLD, wrong, legacy linear-elastic-only
  element names (`Q4_HYBRID`, `Q4_EAS`, `Q4_HYBRID_EAS`, `Q4_HYBRID_RH`)
  instead of their own literal names, which is what `dynamic.py`'s
  dispatch tuples actually key on. Any Abaqus `.inp` loaded through
  `AbaqusModelBuilder` (not the ex12/gen_ex12_inp.py pipeline, which
  builds `element_type_by_pid` directly in Python and bypasses this
  table) with these element names would have silently gotten the wrong
  physics. Fixed to map each name to itself.
- End-to-end smoke test (`scratch/smoke_cpe4rh_solver.py`): single-element
  model built through the real `Mesh`/`ViscoelasticMaterial`/
  `DynamicSolver` API (not a bare kernel call), `element_type="CPE4RH"`,
  5% prescribed stretch — converged in 1-2 Newton iterations, finite
  displacement and state.

## 3. Incidental fix: verification/element_backends.py

A/B verification (`git stash`/`stash pop` around a `verification.run_all`
baseline check) surfaced `jax_q4_simo_fs_K`/`_f_int` unpacking 3 values
from `compute_single` (`q4_visco_simo_fs_jax.py`), which now returns 4
(`f_int, K, state_new, F_n_new`) since an **earlier-session** Updated-
Lagrangian upgrade added `F_n_new` as the 4th return value. Unrelated to
today's CPE4RH/CPE4I work, but a real, confirmed regression from that
earlier change, caught only because this session's full verification
pass surfaced it. Fixed (unpack 4, discard `F_n_new`).

## 4. Final verification state

`python -m verification.run_all`: **12/14 passed**. The 2 failures are
both confirmed pre-existing and unrelated to anything in this document:

- **2-Point Bending (Gulati Elastica)**: element inversion in a plain
  `Q4_COROTATIONAL` bending solve (pid 1, no viscoelastic material, no
  CPE4RH) — an earlier-session issue (`dev_log/plan_two_point_bending_and_ex13_verification_20260905.md`),
  untouched by today's work.
- **Convergence Cantilever Mesh (Q4 B-bar)**: plain Q4 B-bar mesh-
  refinement order-fit, confirmed failing identically (different %,
  same failure) in a `git stash`-based baseline run of the code as it
  stood before ANY of today's changes.

`enable_new_numba_elements` still defaults `False` everywhere — none of
today's fixes or additions change the default (JAX) production path's
behaviour.
