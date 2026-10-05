# A4 — StepPlan (LOWER): plan (2026-10-05)

Spec row A4: `StepPlan` shared by 2D and 3D; BC / load application via
`dispsolver/loads` (B1); delete the per-solver `_update_step_bcs` /
`_compute_step_external_loads` bodies. Done when 2D and 3D produce identical
`f_ext` on twin decks; deck06 / Hertz / reader tests green; `run_all` 13/14.
Committing A4 is sync point **S3** (agent B starts B5).

## What the two solvers do today (read 2026-10-05)

BCs (`_update_step_bcs`, identical logic, dim 2 vs 3):
- `UserFunctionBC` (`evaluate_node(coords, t)`) per node.
- Displacement BC with an amplitude (object, or name resolved in
  `step.parent_model.amplitudes`): value = target * amp(t).
- No amplitude: STATIC/QUASISTATIC ramps from the step-start displacement,
  `u0 + (target - u0) * clamp(t / time_period)`; other procedures: target.

Loads (`_compute_step_external_loads`):
- CLOAD (`get_force_vector`): nodal, times the STATIC default ramp.
- Pressure: 3D integrates on the REFERENCE geometry (surface path C3D8
  only; element-face path C3D8 / C3D20). 2D (`solver/traction.SurfaceTraction`)
  integrates on the DEFORMED geometry X+u with u frozen at the increment
  start (a lagged follower load). **2D and 3D disagree once the model deforms.**
- Gravity / body force: 3D consistent `integral(N dV)`, gravity times the
  element density; 2D raises NotImplementedError.

## Decisions

- Pressure is a follower load when the step's NLGEOM is on and a
  reference-geometry load when it is off (Abaqus/Standard default for
  pressure). Both dimensions evaluate it once per increment on the geometry
  at the increment start (today's 2D scheme; no load stiffness). This may
  shift 3D NLGEOM pressure results: measure, report, do not absorb.
- Integration through `dispsolver/loads` (B1) for every element type it
  covers, which also gives 2D its surface (*DSLOAD) path and 3D its
  non-hex faces.
- 2D body / gravity loads stay NotImplementedError in A4 (plane thickness
  convention unsettled: `body_force_weights` uses unit thickness).
- Names resolve through `model/regions.py` (flattened system), as today.

## Shape

`dispsolver/model/step_plan.py`:
- `lower_step(step, sys, mesh, dim, nlgeom) -> StepPlan`, built once per
  step at the start of `solve()`.
- `StepPlan.prescribed(t, u_step_start) -> (dof_indices, values)`.
- `StepPlan.f_ext(t, u) -> ndarray` (u only used by follower terms).
- Terms keep the `source_location` of the BC / load that made them.

Both solvers call it and keep only the increment / Newton machinery.
