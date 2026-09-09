# Handoff — element formulation, UL, Numba, and the ~21% fold wall (2026-09-07)

Session goal: make `examples/ex13_unified_model_io.py` fold the 14-layer
PET/PSA/GLASS display to a teardrop reliably. It did not get there. This
records what was fixed, what was measured, which hypotheses were **refuted**,
and where the next session should start.

---

## 1. Bugs found and fixed (all verified)

| # | Bug | Evidence | Fix |
|---|---|---|---|
| 1 | **Newton state double-update** in `_solve_step_impl`. The 2026-09-06 tie fix added `u_k = u_temp.copy()` after line search but left the old `u_k += du` at the bottom of the loop, so every non-converged iteration applied the step ~twice. Worse, `max_step`/`max_displacement_corr` were applied *after* the commit, so the safety clamps never limited the committed state. | A/B with the clamps relocated: `verification.run_all` went 9/14 → 12/14 (patch test, 3-point bending, elastica load-step all started passing); `tests/test_solver.py` 5 known failures → 0. | Clamps moved before the line search; duplicate `u_k += du` deleted. `dispsolver/solver/dynamic.py` |
| 2 | **PSA bulk modulus 100× too large.** `K = 83.333` MPa against `mu = 0.016779` → K/μ ≈ 4967 instead of ~50. | `.inp` carried `D1 = 0.024` (= 2/K); target for E=0.05 MPa, ν=0.49 is `D1 = 2.4`. | `K = 0.83333` in `fold_model_config.py`; `.inp` regenerated. |
| 3 | **Arruda-Boyce μ used as if it were the initial shear modulus.** For this code's series (`_AB_C`), `G0 = μ · 1.074550` at λ_m = 3. | Layer was 7.5 % stiff (E_eff = 0.0537 MPa instead of 0.05). | `mu = 0.015614` → E_eff = 0.05000 MPa exactly. |
| 4 | **Surface-tie Augmented Lagrange never converged** — λ was updated once per step with no outer iteration. | Real gap (not the misleading "TieNodes N/N active" count) grew 0 → 34 mm by 68°. | Outer AL loop in `run_folding_from_result` (re-solve at the same θ until gap < tol or 8 tries). Gap then held ≈ 0.02 mm. |
| 5 | **Mesh grading asymmetric at the tie boundary** — the docstring promised a cluster spanning `hinge_half_gap ± width`, `zone_dx()` jumped straight to the coarse `plate_body_dx` at `ax >= half_gap`. | — | Symmetric `hinge_hi = half_gap + cluster_width` zone. `dispsolver/mesh/display_builder.py` |
| 6 | **Plate invisible in every PNG**; hinge pivots not drawn. Plate nodes are ≥ 10000 and the plotter skipped all such elements as "dummy master nodes". | — | Plate detection fixed; pivots drawn as hollow blue circles (`rbe2_constraints` argument). `dispsolver/export/plotter.py` |

---

## 2. Element work

### 2.1 The "hybrid" elements in this repo were not hybrid

`q4_hybrid_jax`'s hybrid/EAS/SRI kernels are gated on
`isinstance(mat, J2Plasticity)`, so for the PSA (Arruda-Boyce + Prony) **every**
element name fell through to a plain displacement Q4. Measured at the free-span
aspect ratio (AR 8.3): `Q4_HYBRID_EAS`, `Q4_COROTATIONAL_HYBRID_EAS`,
`Q4_HYBRID_SRI`, `Q4_COROTATIONAL_HYBRID_SRI`, `Q4_HYBRID`,
`Q4_COROTATIONAL_SRI`, `Q4_COROTATIONAL_EAS` all returned **exactly** the same
number (36.00× too stiff) — the signature of one shared fallback. Independently,
`dev_log/session_20260730_reduced_integration_failure.md` already recorded that
`q4_hybrid_jax`'s "hybrid" function has zero pressure content.

### 2.2 New elements (all validated against the JAX reference)

| Element | File | Accuracy vs beam theory | Notes |
|---|---|---|---|
| `CPE4I` (incompatible modes, Simo–Armero Q1/E4) | `dispsolver/element/q4_visco_eas_jax.py` | 1.01 @ AR 8.3, 1.00 @ AR 1, unchanged to K/μ = 50 000 | Theory-reviewed: patch test exact (Σw·F̃ = 0), objective, no factor-2 in work conjugacy, `K_ua = K_auᵀ` to 1.8e-15 |
| `CPE4H` / `CPE4IH` (true mixed u-p, element-constant pressure) | same file | CPE4H 1.61 @ AR 3.33, 1.02 @ AR 1.5; CPE4IH 0.89 @ AR 3.33 | Pressure verified = K(J−1) exactly. Best compressive tangent (min eig −0.019 vs CPE4I −0.571) |
| Numba `CPE4H` (+UL) | `dispsolver/element/q4_visco_hybrid_up_numba.py` | force 1.4e-15, tangent 2.8e-6 vs JAX | Pressure residual is linear in p → closed form `p = K·∫(J−1)dV / V`; recomputing p inside each FD perturbation makes the FD Jacobian the **already-condensed** tangent |
| Numba co-rotational `SRI` + J2 | `dispsolver/element/q4_corotational_sri_j2_numba.py` | force 6.6e-13, tangent 2.4e-6 (1.1e-2 in shear: FD vs analytic C) | The PET/GLASS element; previously JAX-only (the existing `q4_sri_numba` is linear-elastic, no plasticity — wiring it would have silently dropped plasticity) |

Two HIGH defects found by review and fixed in `CPE4I`:
1. `ALPHA_MAX·tanh(a/ALPHA_MAX)` is **not** a saturation — it shrinks every
   iterate by ~(a/ALPHA_MAX)²/3, so `f_a` could never reach zero and the element
   was 0.8–2.5 % over-stiff with a tangent whose constraint was violated.
   Replaced with a p-norm guard (identity to O(a⁸)); `‖f_a‖` went 1e-5 → 5.7e-13.
2. Condensed force was missing `− K_ua K_aa⁻¹ f_a`. Added.

### 2.3 Element choice, measured (production mesh)

| Element | Bending accuracy | Compressive tangent min eig (−5 % / −20 %) |
|---|---|---|
| `Q4_VISCO_SIMO` (F-bar, was default) | 38.3× @ AR 8.3 | −0.021 / −0.185 |
| `CPE4I` | 1.01 | −0.571 / −9.18 (worst) |
| `CPE4H` | 1.61 @ AR 3.33 | **−0.019 / −0.122** (best) |
| `CPE4IH` | 0.89 @ AR 3.33 | ≈ CPE4I |

`Q4_COROTATIONAL_EAS` (was the PET default) **fails to converge at the
production AR ≈ 12.5** while SRI and plain corotational both converge on the
same harness — very likely the unexplained "crashes at ~22°/side" in
`dev_log/session_20260730_reduced_integration_failure.md`. PET/GLASS switched to
`Q4_COROTATIONAL_SRI` (1.22–1.23× at every AR, converges everywhere).

Hybrid needs ≥ 2 element rows through the adhesive layer to be accurate
(0.597 with 1 row, 0.922 with 2) — a mesh requirement, not a bug.

---

## 3. Updated Lagrangian — the structural gap

`ul_large_rotation_mode = True`, and the corotational/J2 family uses it
(`F_n_elem = self._ul_F_n[e]`), but **every viscoelastic kernel was TL-only** —
no `F_n` argument at all. So in one model the PET/GLASS layers ran UL while the
PSA layers were evaluated from the flat reference all the way through the fold,
with no per-increment `det(F_inc) > 0` protection. This is independent of
element type, mesh, AR, pivot and tie — which matches the wall not moving when
any of those changed.

Fixed this session: `F_n_gps` added to `q4_visco_simo_fs_jax.compute_single`,
`q4_visco_eas_jax.compute_single_eas` and `compute_single_hybrid`; all now
return `F_n_new`. The dispatch builds `coords_ref = coords + u_ref`,
`u_inc = u − u_ref`, passes `F_n`, stores `F_n_new`, and falls back to TL per
element when the updated reference degenerates (`det J < 1e-4`).

---

## 4. Hypotheses tested and REFUTED — do not redo these

| Hypothesis | Result |
|---|---|
| Element shear locking causes the flat free span | Element error 38× → 1.01× moved the sag from 20.6 % to 43.5 % of the Hermite reference but **did not move the wall at all** |
| Hinge pivot position (x) is wrong | 0.8 / 1.4 / 1.5 mm all fail at the same point |
| Hinge pivot height (y) is wrong | y = 0 and y = 0.25 fail identically; pivot verified to sit exactly on the tie plane (display bottom row and plate top are both at y = 0 in the generated `.inp`) |
| Mesh grading asymmetry at the boundary | Fixed; wall moved 23 % → 24.3 % only |
| Tie should be an exact constraint (MPC elimination) instead of penalty | **Worse** — failed at 13 % (1.62°) vs 21–23 % for penalty. Treat as *inconclusive*, not proof of over-constraint: the prescribed-DOF implementation re-set the whole BC array each step and may conflict with the RBE2 condensation |
| The free span never curves | False. Measured bottom-contour `uy(x)` is a smooth symmetric bowl at every angle; the flat look is a plot-scale artifact (0.05–0.1 mm sag on an 80 × 40 mm axis) |
| PET/GLASS elements return zero force | False — that was a test artifact from initialising J2 state to zeros instead of `F_p_inv = I`. The solver initialises correctly via `initial_internal_vars()` |

---

## 5. Where the wall actually is

Failure is always the **bottom PSA row at the outer plate tip** (x ≈ −39,
pid 1, y = 0…0.03), not the hinge. At the failure step the bottom node rises
0.9209 mm while the node 0.03 mm above rises 0.8831 → current thickness
0.03 − 0.0378 = **−0.0078 mm**, i.e. the layer is squeezed through zero. Shear
strain there ≈ 127 %.

Reading: the stiff upper layers cannot follow the rigid plate's rotation, and
the whole mismatch lands on the thinnest, softest row at the largest moment arm.

---

## 6. State of the tree

- Element config now: PET/GLASS `Q4_COROTATIONAL_SRI`, PSA `CPE4H`,
  PSA layer 2 rows (0.015 mm), free-span `dx = 0.05` (AR 3.33), 20 400 elements,
  `tie_method = "penalty"`, PSA E = 0.05 MPa exact.
- `tie_method` config option added: `"penalty"` (default) | `"kinematic"`.
- **The two new Numba paths are DISABLED by default.** The full-Numba run
  (`scratch/ex12_numba3.log`, PID 1918) produced `Max Res.Force = 4.19e+25`
  and `Energy.Err = 6.08e+19` on the very first assembly, against ~`1.5e+03`
  for the same step on the JAX path — i.e. ~10^22 too large. Both kernels pass
  a single-element comparison against JAX (force 1e-13 / 1e-15), so the defect
  is something that only appears in the real mesh: element node ordering or
  orientation, non-contiguous array views, or the UL bookkeeping interacting
  with the corotational path. They are now gated behind
  `solver.enable_new_numba_elements` (default `False`), so the tree runs the
  previously-validated JAX path unless that flag is set explicitly.
  Setting `--threads` with `--elem_jit numba` additionally raises
  `RuntimeError: Cannot set NUMBA_NUM_THREADS ... currently have 6, trying to
  set 16`; run without the flag.

  **First thing to do with these kernels**: a mesh-level A/B, not another
  single-element test — assemble one step of the real model twice (JAX vs
  `enable_new_numba_elements=True`) and compare `f_int` and `K_T` element by
  element to find which pid/element diverges. Enable one kernel at a time.

Verification scripts written this session (all under `scratch/`):
`psa_locking_test.py` (element accuracy vs beam theory per AR/material),
`locking_ar_test.py` (J2 elements), `eas_element_checks.py` (rigid-body
invariance, eigen-spectrum, alpha saturation), `check_center_sag.py`
(free-span sag vs the Hermite reference, from a saved result).

---

## 7. Next steps, in order

1. **Read `scratch/ex12_numba3.log`.** First full run with UL active on the PSA
   layers. The specific thing to check: does the bottom PSA row at x ≈ ±39 still
   invert, and does the run pass ~21 %?
2. If the wall persists, it is not the element and not the tie geometry. The
   remaining untested solver-side gap is **stabilization**: `self.v` is hard-zeroed
   in `static_mode`, so `AbaqusViscousStabilization` (F = c·M·v) is structurally
   inert no matter how it is configured. Abaqus's static `*STABILIZE` uses a
   pseudo-velocity `(u_k − u_n)/dt` instead; implementing that is the next
   concrete lever for traversing a limit point.
3. Finish the Numba coverage the user asked for: `CPE4I`, `CPE4IH`, corotational
   EAS, and the `Q4_EAS` batch UL gate (`dynamic.py`'s
   `elem_jit == "numba" and not self.ul_mode`) — the single Numba EAS kernel
   already accepts `F_n`, only the batch assembler does not thread it through
   and does not return `F_n_new`.
4. Open items from the theory review not yet addressed: `_ALPHA_MAX = 0.05` is
   not a strain bound (it permits an F perturbation of 1.9 in the thin direction
   of a 0.25 × 0.03 mm element, i.e. GP-level inversion inside the clamp) and is
   not objective; and the known finite-strain EAS compressive instability is
   unguarded (monitor `min eig(K_aa)`, fall back to F-bar per element).
