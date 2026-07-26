# Assembly performance: 301s -> 204s on ex12 (2026-07-26)

Addresses the "known remaining performance issue" left open in
AGENTS.md §4.11.

## Measured, not assumed

§4.11's note was written against an older configuration (it cites
pid=2 / 480 plate elements / 58% of a Newton iteration). The model has
changed a lot since — 14-layer PET-PSA, Q4_VISCO_SIMO, 1800 elements —
so the first step was to re-profile rather than optimize from a stale
number. Added a `max_steps` argument to
`run_folding_from_result` for this: unlike shrinking `t_total` it leaves
the angle ramp untouched, so the profiled steps are the ones a real run
would take.

cProfile over 20 steps / 96 assemblies of the current ex12 model:

| symbol | calls | self | cum |
|---|---|---|---|
| `_element_contributions` | 11520 | 3.4 s | **23.8 s** |
| `scipy...lil.__setitem__` etc. | ~1.0e6 | **9.5 s** (4 fns) | — |
| `_assemble_multi_material_batch` | 96 | 2.4 s | 61.1 s |
| `_solve_step_impl` | 20 | 17.7 s | 102.3 s |

11520 = 96 x 120, exactly the plate element count — confirming the plate
does go through the per-element fallback on every assembly.

## Two fixes

**1. Sequential fallback: lil_matrix -> COO triplets.**
The fallback scattered each element stiffness into a `lil_matrix` with
64 Python-level `__setitem__`/`__getitem__` pairs (~1e6 calls per solve),
then converted to COO anyway — while every other branch in the same
function already emits COO triplets that the caller concatenates once.
Now it emits triplets directly. Identical arithmetic; `coo_matrix` sums
duplicate indices exactly as the `+=` did.

**2. `NeoHookean.pk2_tangent_voigt_batch`.**
`_assemble_multi_material_batch` dispatches to its vectorised branch on
`hasattr(material, 'pk2_tangent_voigt_batch')`. NeoHookean had
`pk2_tensor_batch` and `tangent_voigt_batch` but not the composite the
dispatcher looks for, so it fell through to the per-element path. Added
the composite method — no new assembly branch, the existing tested batch
path now picks the plate up.

De-risked before writing it, because Voigt-convention mismatches have
already bitten this codebase (see `viscoelastic._base_batch`'s comment
about JAX vs numpy shear conventions breaking Newton):
- verified `pk2_tensor_batch`/`tangent_voigt_batch` reproduce
  `pk2_voigt`/`tangent_voigt` to 2e-11 on a random F;
- verified both the sequential kernel and the batch branch use
  **reference-configuration B-bar** (`q4.B_bar_matrix` vs the
  precomputed `_B_bar_all`), so this is not a formulation change —
  `self.elem_coords` and `_precompute_reference_geometry()` are both set
  once at setup and never updated, including in `ul_mode`.

## Results

20-step A/B on the same model, toggling the changes via `git stash`:

| build | 20 steps |
|---|---|
| baseline | 81.6 s / 82.7 s |
| COO only | 77.9 s |
| COO + batch | **61.5 s** |

Full 101-step ex12 run: **301.25 s -> 203.58 s, 32% faster.**

Physics unchanged:
- 20-step displacement field vs baseline: max abs diff 4.0e-12 on
  max|u| = 11.04, i.e. relative 3.6e-13 — floating-point summation order
  only.
- Full run still reaches 90 deg/side with the same interlayer-slip
  result: 81.16 um tip staircase, PSA carrying 99.5%.
- `verification.run_all`: 9/9 PASS.

## Test-baseline correction (important)

Throughout this session runs were reported as "6 passed" / "22 passed"
from the `tests/test_convergence_fixes.py tests/test_rigid_plate_tie.py`
subset named in AGENTS.md §3. The **full** suite is
`135 passed, 10 failed`. All 10 failures were confirmed pre-existing by
A/B stash comparison, not by assumption:

- `test_solver.py` — 5 (`NR should converge, got -40`)
- `test_rbe2.py` — 3 (`assert len(result.constraints) == 1`; rigid bodies
  land in `rbe2_constraints`, not `constraints`. Unrelated to this
  session's commenting-out of `rbe2_elements` — that field is not what
  these assert on.)
- `test_linear_visco_hybrid.py` — 2 (`assert -20 >= 0`)

AGENTS.md previously mentioned only the 3 `test_rbe2.py` failures, which
understated the baseline. Corrected there.

## Still open

The per-element fallback itself remains for any material exposing
neither `J2Plasticity`, `pk2_tangent_voigt_batch`, nor a JAX kernel —
Yeoh and ArrudaBoyce have no numpy batch primitives to compose, so they
would still take the slow path if used as a standalone (non-wrapped)
part material. Not hit by the current models.
