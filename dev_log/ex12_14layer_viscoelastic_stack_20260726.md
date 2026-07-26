# ex12: 14-layer multi-material stack + viscoelastic adhesive (2026-07-26)

## Change

`examples/gen_ex12_inp.py`: display mesh raised from 4 mesh rows /
1 material (PET) to **14 mesh rows / 2 alternating materials**:
- odd rows (1,3,5,...,13): `SUBSTRATE` — same elastic-plastic PET params
  as before (E=4000MPa, nu=0.3, sigma_y0=80MPa, H=400MPa).
- even rows (2,4,...,14): `ADHESIVE` — compliant OCA-like layer,
  E=50MPa/nu=0.45, `*VISCOELASTIC` Prony series (g1=0.6/tau1=0.1,
  g2=0.3/tau2=1.0) + `*TRS, DEFINITION=WLF` (T_ref=25, C1=17, C2=51.6),
  built via `dispsolver/material/viscoelastic.py::ViscoelasticMaterial`.
- Each row now gets its own `*ELSET`/`*SOLID SECTION` (`DISP_L01..DISP_L14`)
  instead of one shared `ELSET=DISPLAY`.

Total thickness unchanged (0.5mm) — 7 substrate + 7 adhesive rows, an
illustrative alternating layup, not a validated real stackup (AGENTS.md
§1.4 already flags this codebase has no real physical stackup yet).

## Bug found and fixed while wiring this up

`dispsolver/io/model_builder.py` — both the COMPOSITE (`*MATERIAL` scoped)
and legacy-flat VISCOELASTIC branches of `_build_materials()` **overwrote**
`mat_params` with `{"base":..., "prony":..., "wlf":...}`, discarding the
base material's own `E`/`nu` (or `mu`/`lambda`) params. Since
`ViscoelasticMaterial._base_batch()` calls `self.base.pk2_tensor_batch(F_batch, params)`
using that same runtime `params` dict, any viscoelastic pid crashed with
`KeyError: 'E'` the first time it was actually evaluated (`NeoHookean._lame`).
Fixed by merging `**base_mat_params` into the final `mat_params` dict in
both branches, so the base's params survive alongside the visco-specific
ones. This was a latent bug — nothing in the existing test suite exercised
a real end-to-end viscoelastic material through `model_builder.py`'s .inp
path before now.

## Result (this session, not yet resolved)

After the fix, the model parses and runs, but hits **element inversion in
the ADHESIVE layer (pid=2) at t~0.2875 (~25.9deg/side, ~51.75deg
combined)** — much earlier than the 4-row/1-material baseline's full
90deg/side. `detF` goes negative in elements 1620/1621 (pid 2, y-row
adjacent to a stiffer SUBSTRATE row). This looks like a sandwich-bending
shear-locking/strain-concentration effect: soft (E=50MPa) adhesive rows
between stiff (E=4000MPa) substrate rows, combined with a thin/elongated
element aspect ratio (~0.036mm row height vs 0.25-1mm x-width -> ~7-28:1
aspect), concentrate shear into the compliant layer under bending far
more than the previous single-homogeneous-material case ever did.

**Not fixed at first** — candidates for follow-up, not yet tried:
mesh grading in y (finer near stiffness transitions, mirroring the
existing x-direction grading fix from §4.12), reduced-integration/
hourglass-control element formulation for the thin compliant rows, or a
less extreme substrate/adhesive stiffness ratio. Do not assume the
14-layer generator code itself needs more changes before diagnosing which
of these actually fixes it — the graded-x fix precedent (§4.12) suggests
mesh, not material-stiffness, is the more likely lever.

## Fix (same session, follow-up): Q4_VISCO_SIMO instead of plain Q4

Root cause of the inversion was element formulation, not mesh/stiffness:
the ADHESIVE layer (pid=2) was assigned plain bilinear `"Q4"`
(`ex12_abaqus_inp_plate_fold.py:94`), which shear/volumetric-locks badly
in a thin, soft layer sandwiched between much stiffer substrate rows.

`"Q4_EAS"` (Simo-Rifai EAS Q4, `dispsolver/element/q4_eas.py`) was
considered first but **does not work with `ViscoelasticMaterial`**: its
dispatch in `dynamic.py:2792-2793` is gated on
`isinstance(mat_adapter.material, J2Plasticity)`, so a `ViscoelasticMaterial`
pid silently falls through to the plain-Q4 batch path (no error, no EAS
benefit) — and even if forced through, `q4_eas.py`'s
`compute_eas_j2_contributions` calls `material.pk2_voigt(Ft2, params, state_gp)`
with only 3 args, while `ViscoelasticMaterial.pk2_voigt` requires a
mandatory 4th `dt` argument (`TypeError`).

Instead, `"Q4_VISCO_SIMO"` (`dynamic.py:2725-2761`,
`dispsolver/element/q4_visco_simo_fs_jax.py`) is a finite-strain,
pluggable-base Simo viscoelastic element with **F-bar volumetric-locking
control** (`q4_visco_simo_fs_jax.py:40,211,223`), already wired
end-to-end for `ViscoelasticMaterial` (`isinstance(vmat, ViscoelasticMaterial)`
required, `dynamic.py:2734-2735`) via `ViscoelasticMaterial.simo_fs_args()`
(`viscoelastic.py:426`).

Change: `ex12_abaqus_inp_plate_fold.py:94`
`element_type={1: "Q4_COROTATIONAL", 2: "Q4"}` →
`element_type={1: "Q4_COROTATIONAL", 2: "Q4_VISCO_SIMO"}`.

**Result**: full 90°/side (180° combined) closure, 101 steps, **zero
cutbacks**, 1-2 Newton iterations/step, 307.43s wall time, `n_inverted==0`
throughout — matches the AGENTS.md §1.0 success criteria, now with a real
14-layer alternating SUBSTRATE/ADHESIVE (viscoelastic) stack instead of
the single-homogeneous-material baseline. `pytest
tests/test_convergence_fixes.py tests/test_rigid_plate_tie.py` still 6
passed.

**Takeaway for future multi-material layups**: when a pid uses
`ViscoelasticMaterial`, always pair it with `"Q4_VISCO_SIMO"`, not plain
`"Q4"` or `"Q4_EAS"` — the latter two either lock badly or silently
degrade to the non-EAS path for this material type.

## Rename: SUBSTRATE/ADHESIVE -> PET/PSA (same session, same properties)

Renamed the two alternating materials to their real display-stack names
(`gen_ex12_inp.py`): `SUBSTRATE` -> `PET`, `ADHESIVE` -> `PSA`
(pressure-sensitive adhesive) — same physical params, no property change.
Layup is now literally `PET-PSA-PET-PSA-...` x7 pairs = 14 rows. Re-ran
end to end: still full 90deg/side, 101 steps, zero cutbacks, 289.57s,
`pytest` 6 passed. No solver/material code touched, naming only.

## PSA material upgrade: linear-elastic -> Arruda-Boyce hyperelastic + faster relaxation (same session)

User requested PSA be modeled as Arruda-Boyce hyperelastic targeting
E=0.5MPa/nu=0.490 (near-incompressible), with viscoelastic relaxation
tuned to ~20% stiffness drop within ~10s (vs. the earlier, arbitrary
OCA-like Prony constants).

Derived Arruda-Boyce params (`gen_ex12_inp.py`, `*HYPERELASTIC,
ARRUDA-BOYCE` data row = `mu, lambda_m, D`):
- `mu = E/(2(1+nu)) = 0.5/(2*1.49) = 0.16785 MPa`
- `K = E/(3(1-2nu)) = 0.5/(3*0.02) = 8.3333 MPa` -> `D = 2/K = 0.24 MPa^-1`
  (builder maps `K = 2/D` back, `model_builder.py` hyperelastic branch)
- `lambda_m = 3.0` — **assumed** locking-stretch shape parameter, not
  measured/given; flag if a real value becomes available.

Prony (single term): `g1=0.20, tau1=3.33s` -> relaxed fraction at t=10s
= `g1*(1-exp(-10/3.33)) ~= 19.3%`, i.e. ~20% stiffness drop by ~10s as
requested. WLF shift params left unchanged (T_ref=25, C1=17, C2=51.6 --
not requested to change).

Composite `*MATERIAL` scoping in `model_builder.py` already supports
hyperelastic-base + viscoelastic-wrap (the `has_hyper` branch builds
`ArrudaBoyce`, then the shared `if has_visco:` wrap applies regardless of
which branch built the base) and `ViscoelasticMaterial.simo_fs_args()`
already has an `"arruda"` case (`viscoelastic.py:446-448`,
`q4_visco_simo_fs_jax.py:89`) — no new solver code needed, `Q4_VISCO_SIMO`
element assignment from the earlier fix carries over unchanged.

**Result**: full 90deg/side closure, 101 steps, zero cutbacks, 287.55s,
`pytest` 6 passed. **Caveat**: mesh-quality NOTICE appeared near the end
(`min det(J)~0.31`, 42-44 elements "warped" — approaching but not
reaching inversion, `n_inverted` still 0) in the last few steps. This is
new relative to the PET/PSA-with-linear-elastic run (no such notice
there) — the softer, more compliant Arruda-Boyce PSA at large rotation is
closer to the locking/inversion boundary than the previous linear-elastic
PSA was. Not a failure, but worth watching if lambda_m or the
substrate/PSA stiffness ratio changes further.
