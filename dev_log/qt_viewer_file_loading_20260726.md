# Qt viewer: open a saved result file, with full stress support — 2026-07-26

Closes `witty-juggling-moore.md`'s Qt-viewer plan.

## What changed

1. **`result_io.py`**: `ResultWriter` gained `material_objects=` (live
   `{pid: MaterialModel}`, distinct from the existing JSON-safe
   `materials` param summary) and `add_step(..., state=)`. Both are
   **pickle-only** -- the `.h5` backend keeps its plain-array/JSON
   contract and simply doesn't carry them (`Result.has_state` /
   `bool(result.material_objects)` tell you which parts of a given
   result will work). `Result` gained `.state(step)` and
   `.material_objects`.
2. **`ex12_abaqus_inp_plate_fold.py`**: `run_folding_from_result` now
   passes `material_objects=result.materials` to `ResultWriter` and
   `state=solver.state.copy()` to every `add_step` -- free (the array
   already exists), no new per-step computation.
3. **`dispsolver/postprocess/live_view.py`** (new): `ResultSolverAdapter`
   -- a solver-shaped read-only view of one step of a `Result`, exposing
   exactly what `viewer.py` reads (`n_elem`, `conn`, `coords`, `u`,
   `state`, `material`, `materials`, `material_params`,
   `mesh.elements[eid].pid`, `elem_ids`, `time`, `n_dofs`). No
   recomputation, pure array slicing.
4. **`viewer.py`**: added `launch_from_result(path_or_result, step=-1)`
   and a step slider (only shown when backed by a multi-step `Result`;
   a live-solver session is unaffected). Also fixed a **pre-existing
   bug**: `compute_field`'s stress branch used `solver.material` -- a
   single global material (the first pid's) -- for every element
   regardless of its actual pid. Invisible for a single-material model;
   wrong for ours (PET/PSA/STEEL). Now looks up the element's own
   material via `solver.materials[pid]`.

## Two more pre-existing bugs found while verifying (not introduced by
## this change), both fixed within this same pass per user decision

Fixing the per-pid material lookup made `ViscoelasticMaterial.pk2_voigt`
actually get called on PSA elements for the first time (previously
everything used the PET/J2Plasticity material, which happened not to
crash). That call immediately surfaced:

**a) Wrong call signature/state shape** -- `viewer.py` called
`mat.pk2_voigt(F, {}, gp_state)` (3 args, flat 5-element `gp_state`,
empty `params`) uniformly for every material. `ViscoelasticMaterial.pk2_voigt`
needs `(F, params, h_prev, dt)` where `h_prev` is `(M+1,3,3)` and
`params` must carry the *base* material's own params (mu/lambda_m/K for
Arruda-Boyce). Fixed: `viewer.py` now detects `ViscoelasticMaterial`
(duck-typed via `hasattr(mat,'base'/'M'/'g_i')`), unflattens the stored
state with `viscoelastic._flat_batch_to_tensor_3d`, passes the pid's own
params, and calls with **`dt=0.0`** -- not an approximation: the stored
state was saved right after the real solve converged at this same `F`,
so `dS_dev = S_dev_el - h_prev[M] = 0` and `dt=0` reproduces that exact
equilibrium stress rather than advancing the material one more
increment.

**b) `_volumetric_stress`/`_extract_lam_mu` hard-coded NeoHookean's own
volumetric derivative** (`p = (-mu + lam*lnJ)/J`) for *any* base
material, and required `E`/`nu` or `mu`/`lambda` -- ArrudaBoyce's
`{mu, lambda_m, K}` params have neither, so this raised `KeyError: 'E'`.
**Lucky structural fact**: the actual solve's finite-strain path
(`q4_visco_simo_fs_jax._simo_pk2`) uses the Simo & Hughes (1998)
logarithmic volumetric split `U(J) = (K/2)(lnJ)^2`, i.e.
`S_vol = K*lnJ*C^-1` -- and substituting `mu=0, lam=K` into the
*existing* `p = (-mu+lam*lnJ)/J` formula reduces to exactly that. So
`_extract_lam_mu` now returns `(0.0, K)` when `'K' in params` and no
`'lambda'` is present, reusing the same K-preference pattern already
applied to `simo_fs_args` earlier this session -- this is the same
formula the real solve uses, not a second approximation.

**Scope note**: per user decision, only `_volumetric_stress`
(`_extract_lam_mu`) was fixed to be base-aware; a broader audit of
`viscoelastic.py` for other NeoHookean-only assumptions was explicitly
left out of scope.

**Why the actual 90°/side solve was never affected**: `ex12`/`ex13`
drive the PSA elements through `Q4_VISCO_SIMO` ->
`q4_visco_simo_fs_jax.compute_single` -> `simo_fs_args` (already fixed
earlier this session), never through the numpy `pk2_voigt`/
`pk2_tangent_voigt_batch` methods this fix touches. Those numpy methods
were dead code for this model until the viewer started calling them
just now.

## Verification

- Per-pid stress sanity (steps 10/50/-1 of the real 101-step
  `ex12_result.pkl`): PET (pid=1) stress climbs 2.6 -> 11.5 -> 21.7 MPa
  (von Mises mean) as folding progresses; PSA (pid=2) climbs
  0.067 -> 0.39 -> 0.98 MPa (physically sensible for a ~0.5 MPa-modulus
  layer under large strain); STEEL (pid=3, `E` deliberately set to
  `1e-6`) stays exactly 0 -- all finite, all distinguishable by
  material, which is the direct regression check for the per-pid bug
  fix.
- Qt offscreen smoke test (`QT_QPA_PLATFORM=offscreen`): loaded the real
  result, step slider present with correct range (0-100 for 101 steps),
  switched all four field families (disp/stress_vm/principal_stress_1/
  strain_xx), scrubbed steps 0/50/100 and confirmed the plot title's
  time matches (0.0050s / 0.5025s / 1.0000s), toggled a PID checkbox --
  no exceptions.
- `verification.run_all`: 9/9 PASS.
- `pytest tests/`: 145 passed, 1 xfailed (unchanged from the
  correctness-pass baseline).
- Re-ran `ex12_abaqus_inp_plate_fold.py` end to end with the new
  `state=`/`material_objects=` recording: 205.64s (no meaningful
  regression from the ~204-208s baseline; `state` is a copy of an array
  that already exists, no new computation). Result file size grew
  5MB -> **75MB** (101 steps x 1800 elements x state array) -- worth
  knowing if saving many long runs.

## Out of scope (per plan and per user's explicit scope decision)

- No HDF5-backend stress support (documented pickle-only).
- No broader audit of `viscoelastic.py` for other NeoHookean-only
  volumetric/deviatoric assumptions beyond `_extract_lam_mu`.
- 1-point (element-center) stress evaluation approximation in
  `viewer.py` unchanged -- pre-existing, unrelated to file loading.
