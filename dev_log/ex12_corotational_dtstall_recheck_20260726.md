# ex12_rigid_plate_display_fold_corotational.py: dt-stall re-check (2026-07-26)

## Context

AGENTS.md §1.2 documented (2026-07-25) that this script's `RigidBodyPart`
Dirichlet-BC path stalled around ~7.9deg/side: Newton iteration count
plateaus at 9-10 (vs `target_iters=8`), so `AdaptiveDtController` decays
`dt` toward `dt_min` with no recovery, and full closure was never reached
that session.

## Finding: does not reproduce

Ran the script unmodified (no code changed in it or `dt_controller.py`
this session) end to end:
- `dt` stayed pinned at `dt_max=0.01` almost the entire run.
- Newton iterations ranged 4-7 throughout (never sustained >=9).
- **Full 90deg/side (180deg combined) closure reached, 101 steps, zero
  cutbacks, 313.09s wall time.**

No root cause for the discrepancy was identified or investigated further
(would require bisecting across the session's other changes -- Q4_VISCO_SIMO
wiring, model_builder/abaqus_parser fixes, dt-stall none of which touch
this script's code path). Recorded as an observation, not a fix.

## New finding: kink_detected=True

`fold_success_verdict()` at full closure reports:
`u_shape_ok=False  n_inverted=0  kink_detected=True  closure_both_ok=True
plate_gap_mm=6.00`

The two plates close correctly (`closure_both_ok=True`), but the
hinge-zone bottom-surface curvature has a sharp crease rather than the
smooth rounded U required by AGENTS.md §1.0's success criterion --
`check_smooth_curvature()` (`dispsolver/solver/diagnostics.py:125`)
flags `max_curv_ratio` over its warn threshold.

**User correction (2026-07-26): a sharp kink here is expected, normal
physics for this model, not a bug to fix.** This script's display is a
single homogeneous `J2Plasticity` layer (no real bend-radius-limiting
multi-layer stack, unlike the 14-layer PET-PSA model in
`ex12_abaqus_inp_plate_fold.py`) with a uniform `nx_disp=80` mesh -- with
nothing in the model actually enforcing a minimum bend radius, folding
this simplified structure to 90deg/side has no physical reason to stay
smooth; a crease at the hinge is the expected result, not evidence of a
meshing or solver defect. **Do not port the §4.12 graded-mesh fix here
chasing kink_detected -- that would be solving a diagnostic flag, not an
actual problem.** `check_smooth_curvature()`'s warn threshold is tuned
for the 14-layer stack's physical assumptions (§1.0), not this
single-material script; a "u_shape_ok=False" verdict here doesn't mean
the run is wrong.

## Not done this session

No further action taken on `kink_detected` -- see correction above.
AGENTS.md §1.0 and §1.2 updated to reflect current dt-stall status; do
not cite the old "~7.9deg stall" note as current.
