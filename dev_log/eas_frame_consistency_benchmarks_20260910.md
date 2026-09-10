# EAS frame-consistency — four defects found by two real Abaqus benchmarks (2026-09-10)

Continues `dev_log/eas_stabilization_modernization_20260909.md` and the
element-by-element Abaqus 1:1 audit of
`dev_log/plan_abaqus_element_consolidation_20260908.md`, under the standing
directive **"영구 예외 없음. 전면 재검토할 것임. abaqus 요소로 1:1 개발할
것임."**

Two official Abaqus Benchmarks Manual reproductions —
`verification/abaqus_benchmarks/nlgeo_cantilever.py` (geometrically nonlinear cantilever, tip
transverse load, CPS4I) and `verification/abaqus_benchmarks/cook_membrane.py` (Cook's membrane,
near-incompressible neo-Hookean, CPE4IH) — were both failing outright. Each
turned out to be a **different** defect, and all four found here are the same
*class* as AGENTS.md §4.14's F4/F6: **a frame/work-conjugacy inconsistency that
is exactly invisible on the axis-aligned, Total-Lagrangian configurations every
existing unit test uses.**

Nothing is committed (standing instruction: commit once, after the redesign
lands).

---

## 0. Summary

| # | file | defect | detection | status |
|---|---|---|---|---|
| B1 | `dispsolver/element/q4_eas_jax.py` | UL `K_aa` was **not** the Jacobian of `f_alpha` — PK2 from the total `F` contracted against operators on config *n*, with no push-forward. 97.6% wrong. | FD Jacobian of the element's own residual on the real failing element | **fixed** |
| B2 | `dispsolver/element/q4_visco_eas_jax.py` `_residuals` (CPE4I) | `f_u` used the pushed `S_n` while `f_a` used the raw `S_v`, and `B_L` was built from the **compatible** `F_inc_c` instead of the enhanced `F_inc`. Broke the `K_au = K_ua^T` the condensation assumes. | autodiff `K_au` vs `K_ua.T`; FD `d f_e/d u` vs `K_e` | **fixed** |
| B3 | `dispsolver/element/q4_visco_eas_jax.py` `_residuals_h` (CPE4H/CPE4IH) | same three-way frame split, plus `r_p` integrated on config *n* while the stress rows integrate on config 0. | same harness | **fixed** |
| B4 | `dispsolver/element/q4_visco_hybrid_simo_numba.py` | FD element tangent used a **fixed absolute** `h = 1e-6`, so its relative error scales as `h/L_elem` and doubles on every mesh refinement. | A/B against the exact JAX tangent at identical `u` | **fixed** |

Everything below is reproduced numerically; nothing is inferred.

---

## 1. B1 — `q4_eas_jax.py`: the UL enhanced-mode Newton had a 97.6%-wrong Jacobian

### Symptom

`bmk_nlgeo_cantilever.py`, `Q4_EAS`, `elem_jit='jax'`, `nlgeom=True`: stalls at
`t = 0.2118` (nx=10, 10 cutbacks) / `t = 0.1587` (nx=20, 15 cutbacks). Replaying
the dumped inputs of the worst element (element 8, index 8 of
`scratch/eas_dump4.npz`) through `compute_eas_j2_contributions_jax_status`
standalone returns `status = inf` — element-local non-convergence — and a
hand-rolled 200-iteration trace of the same alpha-Newton shows `|f_a|` decaying
only *linearly* and then wandering, with `K_aa` well-conditioned and positive
throughout. That combination (positive-definite, well-conditioned tangent +
non-quadratic, erratic convergence) is the signature of a **wrong** Jacobian,
not an ill-conditioned one.

### Root cause

In UL mode the kernel builds

```
Ft_inc = Fc + alpha·Fenh          # incremental, on config n
Ft_mat = Ft_inc @ F_n             # total, for the material
S_v, C_v = tangent_voigt_jax(Ft_mat)   # PK2 / material tangent, config 0
G       = _compute_G(Ft_inc, Fenh)     # dE_inc/dalpha,          config n
f_a     = G.T @ S_v * w                # <-- two different reference frames
```

`S_v` is conjugate to the **total** Green–Lagrange strain on config 0; `G`,
`B_L`, `Fenh` and `w = detJ` (on the step-*n* `coords`) are all conjugate to the
**incremental** strain on config *n*. Contracting them directly is AGENTS.md
§4.14's finding F4 — already fixed in `q4_visco_simo_fs_jax.py`,
`q4_visco_eas_jax.py` and `q4_visco_hybrid_reduced_jax.py`, but this file was
never on that list. It is identity-valued when `F_n = I`, which is why every TL
test passed.

### Evidence

Element 8 of the real failing increment (accumulated reference rotation
≈ 39°, `||F_n − I||_max = 0.638`), comparing the coded `K_aa` against a central
finite-difference Jacobian of the element's own `f_alpha`:

| | `‖K_code − K_fd‖ / ‖K_fd‖` | asymmetry of `K_fd` | undamped local Newton, `\|f_a\|` per iteration |
|---|---|---|---|
| **before** | **0.976** | **1.51** (complex eigenvalues — not a gradient of any potential) | `3.5e-1 3.7e-1 3.8e-1 3.0e-1 2.3e-1 …` never converges |
| **after** | **4.6e-10** | 7.8e-11 | `7.6e+3 2.5e-2 1.1e-9` — quadratic, 2 iterations |
| control, `F_n = I` (TL) | 2.5e-10 *(unchanged by the fix)* | — | — |

The TL control is the important row: **the bug is exactly zero on an
axis-aligned/TL reference**, so no patch test, no rigid-rotation canary and no
existing unit test could ever have seen it.

### Fix

`_push_forward(S_v, C_v, F_n_k)` in `q4_eas_jax.py`. Since
`E_mat = F_n^T E_inc F_n` exactly (because `Ft_mat = Ft_inc @ F_n`) and
`dV_0 = dV_n / det F_n`:

```
S_n = T^T S   / det(F_n)          T = Voigt operator of  E ↦ F_n^T E F_n
C_n = T^T C T / det(F_n)
```

Pushing the **tangent** as well as the force is what restores
`K_aa = d f_alpha / d alpha`. Applied at all three sites that consume the
material response (line-search probe `_f_alpha`, `_alpha_body`, final
assembly). Identity when `F_n = I`, so TL is bit-identical.

### Result — `verification/abaqus_benchmarks/nlgeo_cantilever.py`

Exact Bisshopp & Drucker (1945) elastica for this load: `theta_tip = 81.963°`,
`y_tip = 8.1072`, `L − x_tip = 5.5523`.

| nx | element | backend | **before** `uy` | **before** cutbacks | **after** `uy` | **after** err% | **after** cutbacks |
|---:|---|---|---|---:|---|---:|---:|
| 10 | Q4_COROTATIONAL | jax | 1.3697 | 0 | 1.3697 | −83.10 | 0 |
| 10 | Q4_COROTATIONAL | numba | 1.3697 | 0 | 1.3697 | −83.10 | 0 |
| 10 | **Q4_EAS** | jax | **FAILED** (t=0.2118) | 10 | **7.7799** | **−4.04** | **0** |
| 10 | **Q4_EAS** | numba | **FAILED** (t=0.2118) | 10 | **7.7799** | **−4.04** | **0** |
| 20 | Q4_COROTATIONAL | jax | 4.0889 | 0 | 4.0889 | −49.56 | 0 |
| 20 | Q4_COROTATIONAL | numba | 4.0889 | 0 | 4.0889 | −49.56 | 0 |
| 20 | **Q4_EAS** | jax | **FAILED** (t=0.1587) | 15 | **8.0702** | **−0.46** | **0** |
| 20 | **Q4_EAS** | numba | **FAILED** (t=0.1587) | 15 | **8.0702** | **−0.46** | **0** |

("before" reproduced by monkeypatching `_push_forward` to the identity —
`scratch/`-local runner, not committed. Q4_COROTATIONAL is byte-identical
before and after, confirming the change touches only the EAS kernel.)

This also reproduces the benchmark document's own **CPS4 vs CPS4I** finding
under an independent loading case: the fully-integrated element is 83% too
stiff at nx=10 and still 50% too stiff at nx=20, while the incompatible-mode
element is within 4% at nx=10 and 0.5% at nx=20.

### Note on the "chaotic call-order sensitivity" reported earlier

The earlier session observed the same inputs returning `status = inf` when
called alone but `9.97e-9` when called after an unrelated warm-up call, and
raised the possibility of a JAX/XLA compilation-order bug. **There is no XLA
bug.** With a Jacobian that is 97.6% wrong the local Newton has no descent
property at all, so its iterate path is genuinely chaotic in the dynamical
sense: the reported `9.97e-9` sits just under the `_ALPHA_CONV_TOL = 1e-8`
exit threshold, i.e. the loop was terminating on a knife edge where any
last-bit perturbation (a different XLA fusion choice between two compilation
contexts is more than enough) flips the outcome. After the fix the step drops
to ~1e-15 in two iterations and the sensitivity is gone. The step-size exit
criterion introduced on 2026-09-10 (`||d(alpha)||_inf` vs `_ALPHA_CONV_TOL`,
replacing the dimensional `f_a_norm < 1e-10`) is **sound and kept** — it was
reporting a real failure, not causing one.

---

## 2. B2 — `q4_visco_eas_jax.py` `_residuals` (CPE4I): `K_au = K_ua^T` was false

### Symptom

`bmk_cook_membrane.py`, `CPE4I`: the **global** Newton line search collapses
(`[DIAGNOSTIC] Line search collapsed at iter 4`) while the element-local Newton
reports clean convergence. n=4 stalled at `t = 0.0373`; every mesh density
failed.

### Root cause

`compute_single_eas_status` condenses with

```
K_uu = jax.jacobian(_fu, 0);  K_ua = jax.jacobian(_fu, 1);  K_aa = jax.jacobian(_fa, 0)
K_au = K_ua.T          # <-- assumed, with a comment claiming "verified to 1.8e-15"
K_e  = K_uu - K_ua @ solve(K_aa_reg, K_au)
```

That symmetry holds only if `f_u` and `f_a` are gradients of **one** potential.
They were not, for two independent reasons:

1. `f_u` contracted the pushed-forward `S_n` while `f_a` contracted the raw
   `S_v` — the file's own comment recorded this as "out of scope here". The
   partial F4 fix of 2026-09-08 is what broke the symmetry.
2. `B_L` was built from the **compatible** `F_inc_c`, not the enhanced
   gradient. With `F = F_c + alpha·Fenh` (Simo–Armero deformation-gradient
   enhancement) `dE/du = sym(F^T dF_c)` genuinely involves the enhanced `F`;
   using the compatible one silently drops the alpha-dependent half. This
   half is **visible in pure TL**, which is why the TL row below is nonzero.

### Evidence

One real Cook's-membrane element, moderate deformation; TL and a 25°-rotated
UL reference:

| | `‖K_au − K_ua^T‖/‖K_ua‖` | `‖K_e − K_fd‖/‖K_fd‖` | asym(`K_fd`) | local `status` |
|---|---|---|---|---|
| TL, **before** | 4.00e-2 | 3.01e-2 | 3.58e-2 | 1.9e-11 |
| TL, **after** | **3.4e-16** | **5.8e-7** ¹ | 2.3e-9 | 1.9e-11 |
| UL 25°, **before** | 7.78e-2 | **1.28** | 4.90e-2 | **6.7e-1** (did not converge) |
| UL 25°, **after** | **8.5e-16** | **5.0e-7** ¹ | 1.2e-9 | **4.2e-11** |

¹ `5e-7` is the central-difference truncation floor at `h = 1e-6`, not a
residual error.

Note the local `status` column: in UL the element-local Newton *itself* was
failing (`6.7e-1`, i.e. no convergence) before the fix, and converges to
`4.2e-11` after.

### Fix

Move the enhancement into the **incremental** frame — where `Fenh`, `B_L` and
`w` already live — and take one push-forward for the whole GP:

```
F_inc = F_inc_c + alpha·Fenh          # enhanced, config n
F_tot = F_inc @ F_n                   # total, for the material
S_n   = F_n S F_n^T / det(F_n)        # one push-forward
BL    = _BL_columns(F_inc, …)         # enhanced gradient, not compatible
G     = sym(F_inc^T Fenh_j)
f_u  += BL.T @ S_n * w ;  f_a += G.T @ S_n * w
```

Both residuals now differentiate one potential, so `K_au = K_ua^T` holds
exactly and the condensation is valid.

---

## 3. B3 — `_residuals_h` (CPE4H / CPE4IH): the same split, three ways

The mixed u–p element had the identical defect plus a third frame: `r_p`
integrated `(J−1)` with the config-*n* weight `w` while the stress rows
integrated on config 0. Same harness, same element:

| element | mode | `‖K_qu − K_uq^T‖/‖K_uq‖` before → after | `‖K_e − K_fd‖/‖K_fd‖` before → after |
|---|---|---|---|
| CPE4IH | TL | 5.22e-3 → **9.3e-16** | 4.68e-3 → **8.7e-10** |
| CPE4IH | UL 25° | **5.42e-1** → **1.3e-15** | **1.80e-1** → **1.3e-9** |
| CPE4H | TL | 1.09e-15 → 1.1e-15 | 4.7e-10 → 4.7e-10 |
| CPE4H | UL 25° | 8.80e-3 → **1.1e-15** | 8.70e-3 → **4.5e-10** |

CPE4H in TL was already exact — with `use_eas=False` the enhanced half of the
mismatch vanishes (`alpha ≡ 0`), which is precisely why the TL patch tests
never caught this. **CPE4IH under UL was 54% asymmetric with an 18%-wrong
condensed tangent**, and CPE4IH is the element the display fold uses for its
PSA layers.

Fix: `dV_0 = w / det(F_n)` for the pressure row, the same single push-forward
and enhanced `B_L`/`G` as B2. All four configurations are now exact to machine
precision.

---

## 4. B4 — `q4_visco_hybrid_simo_numba.py`: a finite-difference step that was not mesh-invariant

### Symptom

`bmk_cook_membrane.py`, `Q4_VISCO_SIMO`, `elem_jit='numba'`, n=32: element
inversion on the **first** increment, `stalled at t=0.0000, cutbacks=16`. The
same model with `elem_jit='jax'` solves in zero cutbacks. n=4/8 were clean and
n=16 needed one cutback — a progression with mesh density, not a cliff.

### Root cause

This kernel builds its element tangent by **forward finite difference** of the
internal force (a deliberate, documented choice — a material-only tangent was
measured at ~50% error for this F-bar near-incompressible element), with a
**fixed absolute** step `h = 1e-6`. A forward difference's truncation error is
`O(h·f'')`, so *relative to* `K` it scales as `h / L_elem` and therefore doubles
on every uniform refinement.

### Evidence

Both backends assembled at the identical `u = 0`, Cook's membrane:

| n (elem/side) | 4 | 8 | 16 | 32 |
|---|---|---|---|---|
| `L_elem` | ~12 | ~6.0 | ~3.0 | ~1.5 |
| `‖dK‖/‖K‖` **before** | 9.27e-8 | 1.73e-7 | 3.33e-7 | **6.49e-7** |
| `‖dK‖/‖K‖` **after** | 2.77e-8 | 2.61e-8 | 2.48e-8 | **2.43e-8** |
| worst entry (rel) before → after | 5.7e-4 → 1.7e-4 | 1.2e-3 → 1.8e-4 | 2.5e-3 → 1.6e-4 | **5.1e-3 → 1.6e-4** |
| asym(`K_numba`) before → after | 1.4e-7 → 4.1e-8 | 2.7e-7 → 4.0e-8 | 5.2e-7 → 3.8e-8 | **1.0e-6 → 3.8e-8** |
| asym(`K_jax`) (exact reference) | 2.2e-16 | 3.4e-16 | 6.1e-16 | 1.2e-15 |
| `‖df_int‖` (both backends) | 2.0e-16 | 2.6e-16 | 3.3e-16 | 4.7e-16 |

The "before" row is exactly `2×` per refinement, and matches the predicted
`h/L_elem` to within 3% at both ends (n=4: predicted 8.3e-8, measured 9.3e-8;
n=32: predicted 6.7e-7, measured 6.5e-7). `f_int` agreed to 2e-16 throughout —
**the force kernel was never in question, only the differencing.** At n=32 the
worst single entry was already 0.5% off and the tangent had visibly lost
symmetry; with `K/mu = 10000` the condition number is high enough for that to
destroy the first Newton step.

### Fix

Per-column step, the standard choice of Dennis & Schnabel, *Numerical Methods
for Unconstrained Optimization and Nonlinear Equations* (1983) §5.4:

```
h_j = sqrt(eps_mach) * max(|u_j|, L_elem)
```

with `L_elem` the longer element diagonal (rotation-invariant, unlike a
bounding-box extent). This balances truncation against roundoff at
`~sqrt(eps) ≈ 1.5e-8` **relative**, independently of mesh size and units —
which is exactly what the "after" row shows: flat at 2.4–2.8e-8 across a
factor-8 range of element sizes.

`h = 0.0` (the new default) selects this; an explicit positive `h` still forces
the old fixed-step behaviour.

**`DISPFOLD_CACHE_VERSION` bumped `v5` → `v6`** (`dispsolver/_jit_cache.py`) —
this changes a `@njit(cache=True)` kernel and adds a module-level constant
(`_SQRT_EPS`).

### Result

| n | backend | before | after |
|---|---|---|---|
| 16 | numba | 6.8631, **1 cutback** | 6.8631, **0 cutbacks** |
| 32 | numba | **FAILED** (t=0.0000, 16 cutbacks) | **6.9167** (+0.24% vs ref), 1 cutback |
| 32 | jax (unchanged reference) | 6.9181 | 6.9181 |

---

## 5. Cook's membrane — full sweep

Reference curve digitized from the official convergence figure
(`verification/abaqus_benchmarks/reference/cook2d_convergence.png`), `U2` of the top-right
corner. `Q4_VISCO_SIMO` ~ CPE4H, `CPE4I` ~ CPE4IH.

| n | element | backend | U2 | ref | err% | cutbacks |
|---:|---|---|---:|---:|---:|---:|
| 4 | Q4_VISCO_SIMO | jax | 6.2053 | 6.15 | +0.90 | 0 |
| 4 | Q4_VISCO_SIMO | numba | 6.1868 | 6.15 | +0.60 | 0 |
| 4 | CPE4I | jax | 6.3224 | 6.55 | −3.48 | 0 |
| 4 | CPE4I | numba | 6.3224 | 6.55 | −3.48 | 0 |
| 8 | Q4_VISCO_SIMO | jax | 6.7063 | 6.70 | +0.09 | 0 |
| 8 | Q4_VISCO_SIMO | numba | 6.7030 | 6.70 | +0.05 | 0 |
| 8 | CPE4I | jax | 6.6983 | 6.75 | −0.77 | 0 |
| 8 | CPE4I | numba | 6.6983 | 6.75 | −0.77 | 0 |
| 16 | Q4_VISCO_SIMO | jax | 6.8648 | 6.85 | +0.22 | 0 |
| 16 | Q4_VISCO_SIMO | numba | 6.8631 | 6.85 | +0.19 | 0 ¹ |
| 16 | CPE4I | jax | 6.8478 | 6.85 | −0.03 | 0 |
| 16 | CPE4I | numba | 6.8478 | 6.85 | −0.03 | 0 |
| 32 | Q4_VISCO_SIMO | jax | 6.9181 | 6.90 | +0.26 | 0 |
| 32 | Q4_VISCO_SIMO | numba | 6.9167 ¹ | 6.90 | +0.24 | 1 ¹ |
| 32 | CPE4I | jax | 6.9068 | 6.90 | +0.10 | 0 |
| 32 | CPE4I | numba | 6.9068 | 6.90 | +0.10 | 0 |

¹ post-B4 values, measured separately after the FD-step fix; the sweep table
above was captured between the B2/B3 and B4 fixes and read `1` cutback at n=16
and `FAILED` at n=32 for that one row.

Before this session **every CPE4I row failed** (n=4 stalled at `t = 0.0373`).
All rows are now within 3.5% of the digitized reference at every mesh density,
and CPE4I is less locked than Q4_VISCO_SIMO at the coarse mesh (6.3224 vs
6.2053 against a converged ≈6.90), reproducing the document's own
CPE4H-vs-CPE4IH ordering.

Caveat retained from `bmk_cook_membrane.py`'s docstring: our `CPE4I` carries
**no hybrid pressure DOF** (unlike Abaqus's CPE4IH) and our volumetric penalty
is Simo–Taylor rather than Abaqus's `(1/D1)(J−1)^2`, so an exact digit match is
not expected — only the convergence trend and the locking ordering are.

---

## 6. Which EAS kernel actually runs (answers "does numba need the same fix?")

Traced by wrapping all three entry points and running the cantilever:

```
elem_jit=numba  uy=7.7799 cutbacks=0  calls -> {numba_batch: 0, numpy_seq: 0, jax_vmap: 1}
elem_jit=jax    uy=7.7799 cutbacks=0  calls -> {numba_batch: 0, numpy_seq: 0, jax_vmap: 1}
```

- **`q4_eas_numba.py` needs no fix and could not have been fixed usefully**:
  `dynamic.py:4092` gates the Numba EAS batch on `not self.ul_mode`, so under
  NLGEOM it is unreachable and `elem_jit='numba'` silently uses the JAX kernel.
  That is why the numba and jax rows of the cantilever table are identical to
  the last digit.
- **`q4_eas.py`'s sequential fallback is not exercised** (0 calls). It only
  runs when the JAX vmap produces NaN, which no longer happens. Its local
  Newton was separately confirmed by reading to be self-consistent (it composes
  `Fc = Fc @ F_n` *first*, so `G`, `S` and `B_L` all refer to one gradient) and
  it degrades gracefully rather than latching `inf`.

**Open, flagged, not fixed:** `q4_eas.py` uses a config-*n* weight `w` with a
config-0 strain measure — the same missing `1/det(F_n)` volume factor, i.e. an
F4-class residue. Its element-local Newton is unaffected (the Jacobian is
self-consistent), and the path is dead in every current model, so it was left
alone rather than changed without a benchmark that exercises it.

---

## 7. The generalizable lesson (extends AGENTS.md §4.14)

§4.14 already says "verify a new element on a TILTED reference, not just an
axis-aligned rectangle". These four add a second, sharper test that catches
things a rotated-reference sweep alone does not:

> **Check that the element's own tangent is the Jacobian of its own residual,
> by finite difference, on a rotated reference — and, for any condensed/mixed
> element, check every symmetry the condensation *assumes* (`K_au = K_ua^T`,
> `K_qu = K_uq^T`) instead of trusting a comment that says it was verified.**

Both B2 and B3 were introduced *by* a previous partial F4 fix: pushing the
stress forward in one residual and not the other is worse than not pushing it
at all, because it silently invalidates the symmetry the condensation is built
on. A comment asserting "verified to 1.8e-15" survived the change that
falsified it.

### Divergence deliberately left in place

`q4_visco_eas_numba.py` is the Numba mirror of the CPE4I kernel changed in B2
and has **not** been brought onto the new formulation. It is unreachable
today — `dynamic.py:3765` gates it behind `enable_new_numba_elements`, whose
default is `False` everywhere in `dispsolver/` (only `scratch/` experiments set
it) — so `elem_jit='numba'` for CPE4I runs the JAX kernel, which is why the
numba and jax CPE4I rows of §5 are identical to the last digit. **Anyone
turning that flag on must port B2's change first**, or the two backends will
silently disagree on the formulation, not just on roundoff.

And B4 adds: **a finite-difference tangent needs a step scaled to the problem,
and a fixed absolute step is a latent mesh-refinement bug.** Four sibling
Numba kernels still carry the identical `h: float = 1e-6` default —
`q4_plastic_numba.py`, `q4_visco_eas_numba.py`,
`q4_visco_hybrid_reduced_numba.py`, `q4_visco_hybrid_up_numba.py`. They were
**not** changed here because none is exercised by these two benchmarks and the
measurement that justified the change does not exist for them; the scaling law
and the fix pattern above make that a mechanical follow-up.

---

## 8. Regression

### Targeted element tests

```
pytest tests/test_cpe4_element.py tests/test_ul_tl_consistency.py \
       tests/test_cpe4rh_patch.py tests/test_eas_jax_verify.py -q
54 passed in 159.94s
```

### Full suite

```
pytest tests/ -q
3 failed, 284 passed, 4 skipped, 4 xfailed, 48 warnings in 4191.15s
```

| failure | pre-existing? | attribution |
|---|---|---|
| `test_abaqus_config.py::test_config_presets` | yes | documented baseline |
| `test_prescribed_skip.py::test_prescribed_skip_solution_equivalence` | yes | documented baseline |
| `test_3d_numba.py::test_3d_dod_multimaterial_assembly` | **new, but not from this work** | commit `96f8adb` (concurrent 3D-solids work) changed `DynamicSolver3D.assemble_system` to return a 3-tuple `(K, f_int, has_error)` for its "Smart Cutback on element inversion" feature; `tests/test_3d_numba.py:162` still unpacks 2 → `ValueError: too many values to unpack (expected 2)`. Every file involved (`dispsolver/solver3d/`, `dispsolver/element3d/`, `tests/test_3d_numba.py`) is untouched by this work, which is 2D-only. **Left for the owner of that change** rather than patched from here. |

Note: two commits (`96f8adb`, `a4276a7`) landed mid-session from concurrent
3D-solids work, so the `2 failed / 265 passed` baseline recorded in `prd.json`
is superseded — the pass count grew to 284 because that work also added tests.

### `python -m verification.run_all`

```
VERIFICATION COMPLETE: 12/14 passed in 152.8s
Failed: 2-Point Bending (Gulati Elastica)
        Convergence Cantilever Mesh (Q4 B-bar)
```

**Identical to the baseline committed at HEAD** (`verification/results/`):
`git show HEAD:verification/results/verification_report.md` also reads
`12 passed, 2 failed` with the same two benchmarks, and the Q4 B-bar
`order_fit` matches to all 16 digits (`4.471560358967913` committed vs
`4.471560e+00` measured). **No new failures and no changed values** — expected,
since neither failing benchmark exercises an EAS, CPE4I/H/IH or
Q4_VISCO_SIMO-numba path.

### Debug scaffolding removed

`DEBUG_DU_PRINT`, `DEBUG_EAS_DUMP` and `DEBUG_VISCO_EAS_DUMP` are gone from
`dispsolver/solver/dynamic.py` (grep-verified: zero occurrences repo-wide).
