# 3D element library defect audit (2026-09-12)

**Read-only audit.** No file under `dispsolver/element3d/`, `dispsolver/solver3d/`,
`dispsolver/constraint3d/`, `dispsolver/material3d/`, `dispsolver/mesh3d/`, or
`dispsolver/model/` was modified by this audit. Every probe script lives under
`scratch/audit3d/` (throwaway, not part of the repo). Method follows the 2D
session's own playbook (AGENTS.md §4.9/§4.14/§4.16): every finding below is a
**measured number** from a probe that was actually run, not an impression from
reading code.

**Caveat — this audit ran against a live, actively-changing tree.** The
concurrent 3D session was mid-edit for the entire duration (adding a
`stress_init`/geostatic-stress feature across 5 files). Two things changed
*while this audit was running*:
- `dynamic3d.py`'s SDV-commit bug (flagged by the prior read-only pass,
  `dev_log/plan_2d_3d_interface_alignment_20260912.md` §1.5) was fixed
  mid-session — see §4 "already fixed" below.
- `dynamic3d.py` and `dispsolver/solver3d/assembly_utils.py` picked up
  additional uncommitted changes partway through.

Every finding below states what was true **at the time it was measured**, with
enough detail (file:line, exact numbers) that anyone can re-verify against
current HEAD. A companion, currently-live, purely mechanical bug (not a
correctness defect) is reported first because it blocks re-verifying anything
else.

---

## 0. Executive summary

**Most serious confirmed defect: the 3D `C3D8_EAS` (C3D8I) element's 9-mode
enhanced-assumed-strain machinery is dead code.** Both the production Numba TL
kernel and its JAX twin compute the enhanced-strain condensation matrices
(`K_ua`, `K_aa`) and even an `alpha_opt` value, but **never feed the enhanced
strain back into the stress/internal-force calculation** — `f_int` is
bit-identical (JAX: `2.8e-17` absolute difference) to a plain compatible
(locking-prone) C3D8 element. The condensed tangent this element returns is
then not even the correct Jacobian of its own (unenhanced) residual: measured
**10.4% relative error** against a finite-difference check. Despite the
docstring claiming this element "Completely eliminates 3D Shear Locking," it
provides **zero** shear-locking mitigation for the converged solution and
supplies a **wrong Newton tangent** on top. This is a worse defect than any
single one 2D found (F4-F6, B1-B4) — those were subtle sign/frame/transpose
errors; this is a complete absence of the core mechanism, on the element this
session's own docstring says the whole 9-mode scheme exists for.

Four more independently confirmed, quantified defects, roughly in severity
order:

1. **3D J2 plasticity material tangent is always the elastic tangent** —
   never updated to the consistent/algorithmic elastoplastic tangent once
   yielded. Measured **53.5% relative error**, shear-diagonal off by ~10x, on
   a modestly plastic strain state. (`dispsolver/material3d/numba_materials.py`,
   `MAT_J2_PLASTICITY` branch.)
2. **Corotational C3D8 tangent is not the Jacobian of its own residual** —
   measured **99.5% relative error**. `f_int` itself is correct (isotropic to
   machine precision, rigid-rotation canary clean), but `∂R/∂u_elem` is
   entirely absent from the returned `K`. (`c3d8_corotational_numba.py:390-406`.)
3. **Hybrid ("C3D8H") element bypasses the material dispatcher entirely** —
   hardcodes linear elasticity regardless of assigned material (J2, Prony,
   anything) — plus is **currently non-functional at all** (undefined
   `gp_idx`, a NameError on every call, from the same in-progress edit).
   (`c3d8_hybrid_numba.py:394` and surrounding.)
4. **F-bar C3D8 tangent omits the F-bar volumetric-consistency term** —
   measured **20.5% relative error** against its own FD Jacobian. Unlike the
   EAS/hybrid cases, this element's *residual* is correct and objective
   (isotropy error `3.9e-15` to `8.0e-15`); only the tangent is incomplete.
   (`c3d8_fbar_tl_numba.py:206-207, 279`.)

Plus a confirmed, already-known-from-prior-pass architectural defect
(dispatch-table disagreement routing `C3D4`/`C3D10M`/plain `C3D8` silently
onto the 8-node hex F-bar kernel) that this audit independently re-confirmed
still live at the time of writing, and one defect this audit found and the
concurrent session fixed **during** the audit (SDV commit-on-copy).

---

## 1. Inventory and what was run

| Element (as dispatched) | File | C6 rotated-ref sweep | C7 global isotropy | C8 FD Jacobian | C9 condensation symmetry | C10 JAX≡Numba | Verdict |
|---|---|---|---|---|---|---|---|
| C3D8I / C3D8_EAS (k=0, production) | `c3d8_eas_tl_numba.py` | not needed (defect found upstream of geometry) | not needed | **10.4% err** | N/A — condensation exists but never touches `f_int` | JAX bit-identical to plain C3D8, confirming the same defect there | **DEFECT — enhancement is dead code** |
| C3D8_CR / COROTATIONAL (k=1) | `c3d8_corotational_numba.py` | pass, ratio 1.0 at AR 1/7/15/30, all angles | pass, `2e-15`–`4e-15` | **99.5% err** | N/A | no JAX counterpart exists | **DEFECT — missing `∂R/∂u`** |
| C3D8H / HYBRID (k=2) | `c3d8_hybrid_numba.py` | not run (element non-functional) | not run | 114% err (only measurable after patching the `gp_idx` crash in-memory) | N/A — no two-field condensation exists despite the name | no JAX counterpart | **DEFECT — non-functional + material-dispatch bypass** |
| else / C3D4 / C3D10M / plain C3D8 (k=3, F-bar TL) | `c3d8_fbar_tl_numba.py` | not separately run (isotropy covers the relevant invariance) | pass, `3.9e-15`–`8.0e-15` | **20.5% err** | N/A | `c3d8_fbar_jax.py` exists, not cross-checked this pass | **PARTIAL — residual correct, tangent incomplete** |
| Tetra4ANPElement / C3D4_ANP | `c3d4_anp_jax.py` / `c3d4_anp_numba.py` | not run | not run | not run | N/A | not run | **not reachable via solver dispatch (dead code) — not independently probed this pass** |
| Tetra10Element / C3D10M | `c3d10m_jax.py` / `c3d10m_numba.py` | not run | not run | not run | N/A | not run | **not reachable via solver dispatch (dead code) — not independently probed this pass** |
| Plain C3D8 (reference kernel, not independently dispatched) | `c3d8_numba.py` | — | — | — | — | — | used only as ground truth for the EAS/hybrid comparisons above; small-strain linear, exact `f=Ku` by construction |
| Material dispatcher | `material3d/numba_materials.py` | — | — | J2: **53.5% err** (tangent) | — | — | **DEFECT — J2 tangent never updated post-yield** |

C6/C7/C8/C9/C10 follow the naming in AGENTS.md §4.16's 2D contract suite
(`tests/element_contract/`); no equivalent suite exists for 3D yet (confirmed
by the concurrent architecture-alignment pass, §2.4 of
`dev_log/plan_2d_3d_interface_alignment_20260912.md`).

---

## 2. Confirmed defects, in detail

### 2.1 [CRITICAL] C3D8_EAS (C3D8I): enhanced strain is computed and discarded

**Where:** `c3d8_eas_tl_numba.py:139-295` (production Numba TL kernel, k=0 —
the default and fallback element type in `dynamic3d.py:_initialize_elements`
and `_setup_numba_topology`), and identically in the JAX twin
`c3d8_eas_jax.py:88-163`.

**What was measured** (`scratch/audit3d/probe_eas_no_enhancement.py`,
`probe_eas_fd_jacobian.py`; also independently reproduced in the corotational
fork's control comparisons):

- JAX `Hexa8EASElement.compute_element_stiffness_and_force`'s `f_int` is
  identical to a plain compatible C3D8 element's `f_int` to **`2.78e-17`**
  absolute difference (machine precision) on a distorted element under a
  generic small nodal displacement.
- The Numba TL kernel's `f_int` differs from plain C3D8 only by an amount
  that scales as `O(u^2)` relative to `|f_int| ~ O(u)` — i.e. exactly the
  residual expected from comparing a genuinely Total-Lagrangian
  (Green-Lagrange) kernel against a small-strain-linear one, with **zero**
  contribution attributable to the enhanced modes (confirmed by sweeping
  `u`-scale over 1e-3 to 1e-6 and showing `rel_diff/scale` stays constant at
  `~4.8`, the signature of an `O(scale^2)` absolute difference, not an
  independent O(scale) EAS effect).
- `_compute_c3d8_eas_tl_element_umat_numba` never receives an `alpha`
  parameter and never updates one across calls (no such parameter exists in
  its signature at all) — `K_ua`/`K_aa` are accumulated (lines 277-278) and
  used **only** in the static condensation `K_uu - K_ua @ inv(K_aa) @ K_ua.T`
  (line 292), which touches `K_elem` only, never `f_int` (line 275,
  `f_int += (B_L.T @ S_voigt) * dV`, uses only the compatible `B_L`). The JAX
  twin literally computes `alpha_opt = -inv_K_aa @ (K_ua.T @ u_elem)`
  (`c3d8_eas_jax.py:161`) and returns it — but `f_int` (line 154,
  `f_int += B.T @ stress * dV`) was already fully computed above that line
  using only the compatible `strain_comp = B @ u_elem` (line 144). `alpha_opt`
  is computed for nothing.
- **C8 FD Jacobian check** (`probe_eas_fd_jacobian.py`, distorted geometry,
  moderate deformation `u ~ 0.05·randn(24)`): the condensed `K` this kernel
  returns is **10.4%** away from the true finite-difference Jacobian of the
  very `f_int` it assembles (`max|K_analytic-K_fd| = 35.28`,
  `max|K_fd| = 340.5`). This is expected once you know `f_int` never depends
  on `alpha`: subtracting `K_ua @ inv(K_aa) @ K_ua.T` from `K_uu` is only
  valid when `alpha` is being driven to satisfy its own equilibrium
  `∂(strain energy)/∂alpha = 0` inside the same Newton step — here nothing
  does that, so the subtraction just corrupts the tangent by a
  quantity unrelated to the actual residual (confirmed at the raw level too:
  `K_eas` vs `K_c3d8` differ by 13.8% even though `f_eas ≈ f_c3d8`).

**Impact:** every model in this codebase that uses `C3D8I`/`C3D8_EAS` (the
*default* element type whenever `elem_type` isn't recognized —
`_initialize_elements`'s final `else:` branch, and `_setup_numba_topology`'s
matching `else: k=3`... actually resolves to k=0 only for the two literal
strings; anything else falls to k=3/F-bar, see §2.5) gets **zero** shear
locking mitigation from this element in bending, and a systematically wrong
(and, per the general condensation-symmetry logic, not merely "slightly off"
but wrong by a term of the same order as the whole condensation contribution)
Newton tangent. For a 3D fold problem (thin, high-aspect-ratio elements
through the display thickness — exactly the AGENTS.md §4.1 locking regime),
this element will behave like a locked, plain-compatible hex, contradicting
its own docstring and any design decision made on the assumption that it does
not lock.

**Suggested fix** (precise enough to apply directly): give the kernel a
persisted `alpha` (9 floats per element, alongside `elem_sdvs` — same pattern
as history variables), and inside the Gauss-point loop, after `B_tilde` is
built, add the enhanced contribution to strain before calling
`material_dispatch_3d`: `E_voigt_enhanced = E_voigt + B_tilde @ alpha_prev`
(or, for the finite-strain case, augment `F` directly per Simo & Armero
1992's enhanced-`F` construction — this needs care and is the harder part).
Then solve the element-local equilibrium `f_alpha = integral(B_tilde^T S dV)
= 0` via a per-element Newton loop (since `S` is nonlinear in `alpha` for any
non-trivial material), converging `alpha` before returning `f_int`/`K`, and
only then perform the static condensation using the *converged* `K_aa`/`K_ua`
at that `alpha`. This is a materially larger change than a one-line fix —
budget for it accordingly; it is the 3D equivalent of what 2D's
`q4_eas_jax.py`/`cpe4_jax.py` do via `frame.py`'s config ledger (AGENTS.md
§4.16), which the 3D session has not yet adopted (per
`dev_log/plan_2d_3d_interface_alignment_20260912.md` §3.1's own recommendation
to give 3D the ledger "before it has a UL path to retrofit" — this EAS defect
is a second, independent reason to do that soon).

### 2.2 [CRITICAL] 3D J2 plasticity: tangent never updates past first yield

**Where:** `dispsolver/material3d/numba_materials.py:118-192`
(`MAT_J2_PLASTICITY` branch of `material_dispatch_3d`).

**What was measured** (`scratch/audit3d/probe_j2_tangent.py`): for
`E=1000, nu=0.3, sigma_y0=1.0, H=100`, at a strain state that trial-yields
(`eq_plastic_strain` after the step = `0.036`, confirming the plastic branch
executes), the returned `C_mat` is:

```
diag(C_mat returned)     = [1346.2, 1346.2, 1346.2, 384.6, 384.6, 384.6]
diag(C_fd, true Jacobian)= [ 874.2,  881.9,  881.9,  38.3,  38.3,  38.3]
relative error           = 53.5%   (shear terms off by ~10x)
```

Root cause: `C_mat` is built once (lines 129-135) from pure elastic `lam`,
`mu` and never touched again — the `f_trial > 0` branch (lines 176-192)
computes the correctly radially-returned `S_voigt` but has no code path that
modifies `C_mat`/`C_tangent` for the plastic case. A sanity check at a
genuinely-elastic strain state confirms `eq_plastic_strain = 0.0` there (so
the check above is specifically exercising the plastic branch, not an
artifact of always being elastic).

**Impact:** any 3D model using J2 plasticity beyond first yield gets a
Newton tangent that overstates stiffness (especially in shear) by a large,
state-dependent amount — degraded convergence at minimum, and this is the
*same defect class* AGENTS.md §4.16 already found and characterized for 2D's
`plastic.py` (86% wrong tangent there, from a Voigt-index bug in an otherwise-
present plastic tangent). The 3D case is simpler to characterize (no plastic
tangent branch exists at all) but arguably more severe since it isn't even a
partial/miscoded plastic tangent — it's simply absent.

**Suggested fix:** implement the standard isotropic J2 consistent
(algorithmic) tangent for the radial-return branch — e.g.
`C_ep = C_e - (6*mu^2*d_gamma)/norm_s * P_dev + 2*mu*(...)*n⊗n` (see any
standard reference, e.g. Simo & Hughes 1998, already cited in AGENTS.md's own
theory table) — and return that instead of the elastic `C_mat` whenever
`f_trial > 0`. The needed intermediate quantities (`d_gamma`, `q_trial`,
`norm_s`, `mu`, `H`) are all already computed in the existing branch
(lines 177-192); this is a localized, well-scoped fix.

### 2.3 [HIGH] Corotational C3D8: tangent omits ∂R/∂u

**Where:** `c3d8_corotational_numba.py:390-406` (also confirmed present in
the current, still-mid-edit file).

**What was measured** (fork `a5307d726f7a98688`, full detail in that fork's
report, reproduced in summary here):

- Rigid-rotation-only canary (zero strain, 45/90/120° rotation): `f_local`
  max `9.5e-13` to `1.4e-12` — clean.
- Global-axis isotropy (reference + displacement rotated together, 5 angles
  × 3 axes): relative error `2.2e-15` to `3.7e-15` — clean, **not** the
  Q4_COROTATIONAL_SRI-class absolute-orientation defect 2D found.
- Rotated-reference-only AR sweep (AR 1/7/15/30, angle 0-90°): force-norm
  ratio exactly `1.0` everywhere — clean.
- **C8 FD Jacobian**: `max|K_analytic-K_fd| = 473.9` vs `max|K_fd| = 476.4`,
  **99.5% relative error**. Root cause: `K_global = R @ K_local @ R^T` is
  built with `R` evaluated once per call and never differentiated
  (`∂R/∂u_elem` measured by FD at `~0.27`, not negligible). This is the same
  defect *class* as AGENTS.md §4.4's "modified Newton" (dropping the
  rotational geometric-stiffness term) — but at ~10x the magnitude of that
  documented, accepted 2D baseline (~11%, §4.16).

**Impact:** `f_int` (hence the converged solution) is trustworthy; Newton
convergence rate under this element is not — expect materially more
iterations, cutbacks, or line-search dependence than the tangent's symmetry
would suggest, specifically in the large-rotation fold regime this element
is meant for.

**Suggested fix:** add the standard corotational geometric-stiffness
correction capturing `∂(R·f_local)/∂u_elem` (Crisfield 1997 / Wriggers 2008
ch. 4, spin-operator formulation — already in AGENTS.md's reference table for
the 2D corotational element) — or explicitly document this as an accepted
modified-Newton tradeoff (as §4.4 does for 2D) only *after* measuring its
real cost on an actual fold run, since 99.5% is far outside 2D's own accepted
11% baseline for the same tradeoff.

### 2.4 [HIGH] Hybrid C3D8 ("C3D8H"): non-functional + bypasses material dispatch

**Where:** `c3d8_hybrid_numba.py:394` (undefined `gp_idx`) and the whole
function body (`material_dispatch_3d` imported at line 32 but never called —
confirmed by `grep -c material_dispatch_3d` showing exactly one hit, the
import).

**What was measured** (fork `a47b34d1f3c1ca91b`):

1. **Currently crashes on every call**, part of the concurrent session's
   in-progress edit: `gp_idx` is referenced (line 394) but never assigned in
   this file. Numba's `njit` raises `NameError` at compile time for any
   invocation. Confirmed still present as of this writing (re-grepped after
   the fork's report).
2. Stress is hardcoded inline as linear elasticity (lines 283-294, 383-404),
   with `mu`/`K` decoded from `props[0]`/`props[1]` via the same
   magnitude-sniffing heuristic flagged elsewhere
   (`dev_log/plan_2d_3d_interface_alignment_20260912.md` §1.3, "Neo-Hookean
   is not Neo-Hookean"). A material tagged `MAT_J2_PLASTICITY` is silently
   given linear elasticity instead — confirmed (after an in-memory-only test
   patch of the `gp_idx` line, never written to disk) by comparing `f_int`
   against a plain elastic material with equivalent `E,nu`: bit-identical.
3. **C8 FD Jacobian** (same in-memory-patched test): 114% relative error,
   same `∂R/∂u` omission as §2.3 (this element also uses a corotational
   frame internally).
4. No two-field pressure/displacement condensation exists despite the
   "hybrid" name — pressure is folded in via a single-field volume-averaged
   `B_vol_bar` (mean-dilatation/B-bar style). So there is no CPE4IH-class
   condensation-symmetry defect to check here — but the Abaqus-style name
   `C3D8H` (which denotes a genuine independent-pressure-DOF hybrid element)
   is a naming misrepresentation, not just a numerical concern.
5. Its own inversion metric (`J_vol = 1 + eps_vol_mean`, a linear
   approximation to `det(F)`, line ~304) is self-consistent with its own
   small-strain-in-local-frame formulation — lower severity than the
   detJ/detJ0-vs-detF confusion found elsewhere, not the same bug.

**Suggested fix:** (a) add `gp_idx = 4*gi + 2*gj + gk` inside the `gk` loop
before line 394 (mechanical, one line, matches the pattern already used in
`c3d8_eas_tl_numba.py`/`c3d8_fbar_tl_numba.py`); (b) route the stress
computation through `material_dispatch_3d` instead of the hardcoded inline
linear-elastic formula, the same way the EAS/F-bar TL kernels already do;
(c) either rename this element (it isn't Abaqus's C3D8H) or implement an
actual independent-pressure-DOF condensation to earn the name.

### 2.5 [MEDIUM] F-bar C3D8: tangent omits the F-bar volumetric-consistency term

**Where:** `c3d8_fbar_tl_numba.py:206-207` (`scale_vol = (detF0/detF)**(1/3)`,
`F_bar = scale_vol * F`) and `:279` (`K_mat = B_L^T C_tangent B_L`).

**What was measured** (fork `a47b34d1f3c1ca91b`):

- Global-axis isotropy: `3.9e-15` to `8.0e-15` at all tested angles — this
  element's *residual* is genuinely frame-invariant TL, confirmed clean.
- **C8 FD Jacobian**: `max|K_analytic-K_fd| = 52.2` vs `max|K_fd| = 255.0`,
  **20.5% relative error** (K exactly symmetric, `2.8e-14`). Root cause:
  `scale_vol` is a function of `u` at every Gauss point (through both local
  `detF` and centroidal `detF0`), but the analytic tangent never
  differentiates it — the standard F-bar "Q-bar"/volumetric-consistency
  correction (de Souza Neto et al. 1996) is missing.
- **`detF` choice is correct by design, not a bug**: `material_dispatch_3d`
  is called with `detF0` (line 230), and since `F_bar` is scaled so
  `det(F_bar) ≡ detF0` exactly, this *is* `det(F_bar)`, the actual current
  determinant of the deformation gradient being used — consistent with F-bar
  theory (volumetric response evaluated at one averaged J), unlike the
  reference-Jacobian-ratio confusion elsewhere.

**Impact:** lower severity than §2.1-2.4 — the residual (hence converged
answer) is correct and objective; only Newton's convergence rate suffers, and
by a smaller margin (20.5% vs 53-114-99.5% elsewhere).

**Suggested fix:** add the `∂scale_vol/∂u` contribution to `K_mat` via the
chain rule through `∂detF/∂u_j` (at the GP) and `∂detF0/∂u_j` (at the
centroid) — a real but bounded analytic derivation, not requiring the
larger EAS-style redesign.

### 2.6 [MEDIUM, previously flagged, independently re-confirmed] Dispatch tables still disagree; C3D4/C3D10M/plain-C3D8 still silently route to F-bar hex kernel

Already identified architecturally by
`dev_log/plan_2d_3d_interface_alignment_20260912.md` §1.4. Independently
re-confirmed still current (`dynamic3d.py:96-110` vs `:156-168`, both read
directly at time of writing): `_initialize_elements`'s table and
`_setup_numba_topology`'s table recognize different string sets (`C3D8_FBAR`
in the first, not the second; `C3D8_CR` in the second, not the first), and
`_setup_numba_topology`'s `else: k = 3` sends **any** unrecognized
`elem_type` — including `C3D4`, `C3D10M`, and plain `C3D8` — to the 8-node
hex F-bar TL kernel, which indexes 8 nodes from a connectivity array that,
for a tet mesh, only has 4. No dispatch-integrity assertion exists for 3D
(no `_assert_dispatch_integrity()` analogue). Not independently re-measured
with a live tet mesh this pass (the interface-alignment doc's citation is
sufficient and this audit's own reading matches it exactly); flagging here
only to confirm it is still live, not stale.

### 2.7 [LOW, usability] Material dict construction crashes unconditionally for any dict with a "type" key

**Where:** `dynamic3d.py:142-145` (first pass, expects `mat["type"]` to
already be an int and `mat["props"]` to be an array) versus `:189`
(`mat_obj.get("type", "").lower()`, expects `mat["type"]` to be a string).
**Both run, in that order, for the same input dict.** Confirmed by direct
construction: `DynamicSolver3D(mesh, materials={pid: {"type": "j2_plasticity",
...}})` raises `ValueError: invalid literal for int()` at line 143 (string
where int expected); `DynamicSolver3D(mesh, materials={pid: {"type":
MAT_J2_PLASTICITY, "props": [...]}})` gets past that but then raises
`AttributeError: 'int' object has no attribute 'lower'` at line 189. **A
dict-based material specification cannot currently be constructed at all**
via `DynamicSolver3D.__init__`, regardless of which convention is used — only
an attribute-based `mat_obj` (`.mat_type`/`.props`, or `.C_mat`) works. Not a
correctness defect (nothing computes a wrong answer), but it means the
"friendly" dict-of-string-type-name convention documented in the second
branch is entirely dead/unreachable code today. Fix: pick one convention and
delete the other, or type-check before dispatching between them (e.g.
`isinstance(mat.get("type"), str)` vs `isinstance(mat.get("type"), int)`).

---

## 3. Confirmed correct (measured, not assumed)

- **F-bar C3D8 residual**: frame-invariant to `3.9e-15`–`8.0e-15` under
  combined reference+displacement rotation (§2.5).
- **Corotational C3D8 residual**: frame-invariant to `2.2e-15`–`3.7e-15`
  under the same test, clean on the rigid-rotation canary, and clean across
  an AR-1-to-30 rotated-reference sweep (§2.3) — this element does **not**
  have the Q4_COROTATIONAL_SRI-class absolute-orientation shear-sampling
  defect that 2D found in its analogous element.
- **F-bar `detF0` choice**: correct by construction, not a bug (§2.5).
- **No hand-rolled finite-difference tangents exist anywhere in
  `dispsolver/element3d/`** (grepped for `1e-6`/FD-step patterns — zero
  hits), so the B4-class "FD step size not mesh-invariant" defect (AGENTS.md
  §4.15/§4.16) cannot exist in 3D — there is no FD-based tangent code to
  carry it.

## 4. Already fixed during this audit (not by this audit)

**SDV commit-on-copy** (`dynamic3d.py:291-321`): the interface-alignment
pass's original finding — that `self.elem_sdvs[elem_indices]` is NumPy
advanced indexing and therefore always a copy, so state updates never
persisted back into `self.elem_sdvs` regardless of the `update_state` flag —
was independently re-derived by this audit (confirmed by an isolated NumPy
micro-test: `np.shares_memory(a, a[fancy_idx])` is `False`) and then observed
to have been **fixed by the concurrent session while this audit was running**:
current `dynamic3d.py:319-321` now reads

```python
if update_state:
    # NumPy advanced indexing creates a copy. We must explicitly write the modified sub_sdvs back.
    self.elem_sdvs[elem_indices] = sub_sdvs
```

Independently re-verified end-to-end via `scratch/audit3d/probe_sdv_persistence.py`
(a real `DynamicSolver3D` with a single `C3D8I` element, `MAT_J2_PLASTICITY`):
`elem_sdvs` was `0.0` everywhere before any assembly, became `0.1342`
(`eq_plastic_strain`) after one `assemble_system(..., update_state=True)`
call at a plastic strain level, and `0.2461` after a second such call — i.e.
state now genuinely accumulates across converged increments, and (separately
confirmed by the `.copy()` branch, `dynamic3d.py:293`) is **not** mutated by
trial/rejected assemblies. **Do not re-report this as an open defect without
re-checking current `dynamic3d.py:285-326` first** — it may already be
correct by the time anyone reads this.

The **`detJ/detJ0`-as-`detF` inversion-guard bug** flagged in
`dev_log/plan_2d_3d_interface_alignment_20260912.md` §1.5, cited at
`c3d8_eas_tl_numba.py:252-253`: **re-checked directly against the current
file and found NOT present** — current lines 253-259 compute the true
`det(F)` from the actual deformation gradient and pass that (correctly) into
`material_dispatch_3d`. Either the prior doc's citation was already stale
when written, or the concurrent session fixed it before this audit started;
either way, treat that specific §1.5 claim as **not current** and re-derive
before acting on it. (The F-bar kernel's `detF0` usage, §2.5 above, is a
different, deliberately-centroidal case and was separately confirmed
correct-by-design.)

## 5. Reviewer pass — design defects, unsafe patterns, stale comments

Scope added by team-lead: not numeric probes, but a code-reviewer read of
`element3d/`, `solver3d/`, `material3d/`, `constraint3d/` for obvious design
defects, unsafe patterns (silent `except`, no-value fallbacks), and comments
that assert something the code doesn't actually do (the AGENTS.md §4.15
class: a "verified to X" comment that outlived the change that falsified it).
Grepped for `except:`/`except Exception`/`verified`/`TODO`/`FIXME`/`silently`/
`fallback` across all four packages; findings below, all confirmed by reading
the surrounding code, not just the grep hit.

### 6.1 [HIGH] `assemble_system`'s bare `except Exception` silently swaps the entire formulation

**Where:** `dynamic3d.py:277-365`. The whole fast/Numba assembly path — mesh
classification lookup, all four element kernels, constraint assembly — is
wrapped in one `try: ... except Exception as e: traceback.print_exc();
print("Fastpath failed! Falling back to python loop.")`. On **any** exception
— a real numerical failure, a typo, an import error, or (as directly observed
during this audit) the concurrent session's own in-progress-edit `NameError`
— execution falls through to the slow Python-loop path (`:367-421`), which:

- **ignores every assigned material entirely.** The fallback loop
  (`:388-391`) only recognizes `mat_obj.C_mat` (a full 6×6 elasticity
  matrix attribute) — the `mat_type`/`props`, dict-`"type"`, and
  string-`"type"` conventions used everywhere else in this file (§2.7) are
  not checked here at all. Any material without a literal `.C_mat` attribute
  silently gets `self.C_mat_default` (constructor default `E=200000, nu=0.3`
  unless overridden), regardless of what was actually assigned.
- **never persists SDVs** — the slow path uses `self.element_states`
  (a separate `QuadraturePointState3D` list per element, JAX-instance based)
  which is a completely different state representation than
  `self.elem_sdvs` (§4's fix only touches the fast path).
- **returns no signal to the caller that this happened** beyond a print to
  stdout. `solve_step`/`solve`/any test harness sees a normally-returned
  `(K_global, f_int_global)` and proceeds as if nothing changed.

**Reproduced directly during this audit** (not hypothetical): while probing
SDV persistence (`scratch/audit3d/probe_sdv_persistence.py`, before the
workaround for the concurrent session's `NameError` was added), the solver
silently fell back and ran the model as plain default-linear-elastic instead
of the assigned J2 plasticity — `elem_sdvs` stayed at `0.0` the whole run,
consistent with the fallback path never touching it, and `f_int` reflected
`E=200000` behavior, not the assigned `E=1000, sigma_y0=1.0`. This is
exactly the AGENTS.md §4.8 signature ("converges cleanly, wrong answer, no
error to see") reproduced live.

**Fix:** narrow the `except` to the specific conditions the fallback is
actually meant to handle (if any are load-bearing — a case-by-case audit is
needed, since it's unclear this fallback was ever meant to catch anything
but historical bring-up bugs), and at minimum surface a hard warning/flag on
`self` (e.g. `self.used_fallback_formulation = True`) that callers and tests
can assert against, rather than a print-only side channel.

### 6.2 [MEDIUM] Two more silent `except Exception` blocks around the linear solve

**Where:** `dynamic3d.py:483-488` and `:675-679` (`solve_step` and
`_solve_step_with_hooks`, respectively — two independent copies of the same
pattern). Both wrap `pardiso_spsolve(K_bc, r_bc)` in `except Exception:
reg_diag = 1e-4 * np.eye(self.num_dofs); du = np.linalg.solve(K_bc.toarray()
+ reg_diag, r_bc)`.

Two separate concerns: (a) any exception — not just a singular-matrix
failure — silently triggers this path, with the original exception
discarded (no `as e`, no log); (b) `reg_diag = 1e-4` is a fixed absolute
constant with no relation to the actual stiffness scale of the problem
(contrast AGENTS.md's own solver table, which describes 2D's PARDISO path as
using "adaptive ridge regularization" — this 3D fallback is not adaptive).
For a stiff model (`E~10^5`-`10^8`, the constructor's own default), `1e-4`
is negligible and the fallback will simply fail again with a barely-perturbed
singular matrix; for a very compliant model it could dominate the true
physics. Also, `K_bc.toarray()` on a large sparse system converts to dense —
a potential memory/performance cliff with no size guard, silently triggered
by any exception in the primary solve.

**Fix:** catch the narrower `Exception`/`RuntimeError` PyPardiso actually
raises on singularity (check `pypardiso`'s own exception types), log the
original error, and scale the regularization to the matrix's own diagonal
(e.g. `reg_diag = eps * mean(|K.diagonal()|) * I`), matching the "adaptive"
description already used for 2D.

### 6.3 [MEDIUM] A third, independently-disagreeing element-classification table

Already flagged (§2.6, from the prior architecture pass) that
`_initialize_elements` (:96-110) and `_setup_numba_topology` (:156-168)
disagree on which element-type strings map to which kernel. Reading both
in full during this pass surfaces a **third** disagreement: `_initialize_elements`
maps **both** `"C3D8_FBAR"/"FBAR"` and `"C3D8H"/"C3D8_HYBRID"/"HYBRID"` to
the exact same class, `Hexa8FbarElement()` (:101-104) — i.e. in the slow/
fallback path, "hybrid" silently *is* F-bar, not a hybrid formulation at
all — while the fast path's `_setup_numba_topology` table (:156-168) sends
them to two different kernels (`k=3` F-bar vs `k=2` hybrid). So which
formulation `"C3D8H"` actually gets depends on whether §5.1's silent
fallback has fired that call or not — a model could receive a different
element formulation **mid-run**, from one Newton iteration to the next, with
no indication anywhere in the output. This compounds §5.1: the fallback
isn't merely "same formulation, slower" — it can silently change which
kinematics the hybrid element actually gets.

### 6.4 [LOW] Duck-typed `hasattr(constraint, "reproject_deformed")` — silent no-op for a constraint that doesn't implement it

**Where:** `dynamic3d.py:347, 413`. A future constraint type that needs
large-rotation re-projection (per AGENTS.md §4.13's "MUST be called at the
start of every Newton-Raphson assembly" rule, written for the 2D tie but
architecturally identical for 3D) will silently skip re-projection if it
doesn't happen to define this exact method name — no error, no warning. Not
a currently-triggered defect (both constraint types checked implement it, per
a quick read of `constraint3d/surface_tie3d.py`), but it's the same
no-value-fallback shape as §5.1/§5.2; worth requiring the method on an ABC
(`model/constraint.py`'s `AssemblableConstraint` already exists as an ABC
per the prior architecture pass — `reproject_deformed` could be added there
as a required or explicitly-optional method rather than duck-typed).

### 6.5 [LOW] Declared-but-undispatched material type, silent by numeric flag only

**Where:** `material3d/numba_materials.py:16` (`MAT_VISCOELASTIC_PRONY = 3`
declared, with an SDV count at `:28-29`) vs `:74-204`'s if/elif chain, which
has no `mat_type == 3` branch — falls to `else: error_flag = 2` (:201-204).
Already identified architecturally by the prior architecture-alignment pass
(§1.3); re-confirmed still current. The failure mode is purely numeric
(`error_flag=2`, silently returning zero stress/tangent) with no exception
and no assertion anywhere that every declared `MAT_*` constant has a
dispatch branch — exactly the class of defect 2D's `_assert_dispatch_integrity()`
(AGENTS.md §4.16) was built to make impossible at import time. No such
assertion exists for the 3D material dispatcher.

### 6.6 No stale "verified"/"checked" comments found misrepresenting current code

Grepped all four packages for `verified`/`Verified`/`VERIFIED` and
`TODO`/`FIXME`: **zero hits** in `element3d/`, `solver3d/`, `material3d/`,
`constraint3d/`. Unlike 2D's B2/B3 (a comment claiming "verified to 1.8e-15"
that had been falsified by an intervening change, AGENTS.md §4.15), the 3D
codebase does not currently carry any comment asserting a numeric
verification result that could go stale — most likely because, per the
concurrent architecture pass's own finding, no rotated-reference/FD-Jacobian
verification has been done here yet at all (§2 confirms several kernels fail
exactly the checks 2D uses those comments to record). One comment worth
flagging for the opposite reason — it is accurate and should be treated as
a model for what these files are currently missing: `dynamic3d.py:320`
(`"NumPy advanced indexing creates a copy. We must explicitly write the
modified sub_sdvs back."`) — added by the concurrent session mid-audit,
verified correct by this audit (§4). Encourage more comments of exactly this
shape (a stated invariant plus the reason), and none of the "verified to
X" shape without an accompanying, re-runnable check.

---

## 6. Theoretical verification against real Abaqus documentation

Scope added by team-lead, method following
`dev_log/plan_abaqus_spirit_element_refactor_20260911.md` §1 (which
established the sources and confidence-tag convention reused here):
**V(doc)** = fetched page, section/quote available; **V(lit)** = established
in cited literature; **UNVERIFIED** = could not confirm — no guessing.
Primary source fetched this pass: Abaqus 2016 Analysis User's Guide (AUG)
§28.1.4 "Three-dimensional solid element library",
`ceae-server.colorado.edu/v2016/books/usb/pt06ch28s01ael03.html` — **V(doc)**,
full element list quoted directly from the fetched page. Supporting citations
for incompatible modes reuse the already-**V(doc)**-tagged quotes from the
prior 2D session's own fetch of TG §3.2.5 (re-fetching would be redundant;
that document's citations are reused verbatim, not re-derived).

### 6.1 Which of our 3D element names are real Abaqus elements

| Our name | Real Abaqus element? | Abaqus's actual description (AUG §28.1.4, V(doc)) |
|---|---|---|
| `C3D8` | **Yes** | "8-node linear brick" |
| `C3D8I` / `C3D8_EAS` | **Yes** (as `C3D8I`) | "8-node linear brick, incompatible modes" — 13 internal DOF (TG §3.2.5, V(doc), reused from the 2D session's fetch) |
| `C3D8H` / `C3D8_HYBRID` / `HYBRID` | **Yes** (as `C3D8H`) | "8-node linear brick, hybrid **with constant pressure**" — a genuine two-field (u, p) mixed formulation with an independent pressure DOF per element |
| `C3D8R` | **Yes** | "8-node linear brick, reduced integration with hourglass control" — **not currently dispatched by our solver at all** (absent from both of `dynamic3d.py`'s classification tables) |
| `C3D8_FBAR` / `FBAR` | **No** | Does not appear anywhere in AUG §28.1.4's element list, nor in the Theory Guide chapter searched. "F-bar" is a general academic technique (de Souza Neto et al. 1996), not an Abaqus element or section-controls name — confirmed absent by the fetched page ("No mention of 'F-bar' or 'FBAR' appears anywhere in this documentation"). |
| `C3D8_CR` / `C3D8_COROTATIONAL` / `C3D8_FBAR_CR` | **No** | Same conclusion the 2D session already reached for the general architecture (`plan_abaqus_spirit_element_refactor_20260911.md` §1.2, V(doc), TG): Abaqus's continuum elements under NLGEOM are formulated in the *current configuration* directly, with rotation removed **per-integration-point from the incremental ΔF** (TG §1.4.3/§1.5.4) — there is no per-element rigid-body-triad extraction for continuum solids anywhere in the fetched Theory Guide. "The only documented element-level co-rotational formulation is Abaqus/Explicit small-strain shells" (already V(doc)-cited). This applies to 3D exactly as it did to 2D's `Q4_COROTATIONAL`: no Abaqus continuum counterpart exists, at any dimension. |
| `C3D4` | **Yes** | "4-node linear tetrahedron" |
| `C3D4H` | **Yes** | "4-node linear tetrahedron, hybrid with linear pressure" |
| `C3D4_ANP` | **No** | Not in Abaqus's element list under any name. "Average Nodal Pressure" is the name of Bonet & Burton's 1998 academic technique (*Communications in Numerical Methods in Engineering*, confirmed via literature search), not an Abaqus element designation. Whether Abaqus's own `C3D4H` internally uses ANP-equivalent nodal pressure averaging is **UNVERIFIED** — could not confirm from the fetched Theory Guide pages (the specific hybrid-formulation theory section was not successfully retrieved this pass); do not assume either way. |
| `C3D10` | **Yes** | "10-node quadratic tetrahedron" |
| `C3D10M` | **Yes**, but see §5.2 | "10-node **modified** tetrahedron, **with hourglass control**" — note this is explicitly *not* the same thing as plain `C3D10` in Abaqus's own naming; the "M" and "with hourglass control" are load-bearing parts of the name, not decoration. |

**Two of our four "exotic" 3D element type-strings claim Abaqus elements
that do not exist** (`C3D8_FBAR`, `C3D8_CR`/`COROTATIONAL`) — the exact
"SRI/CPE4S" pattern the 2D session already found and is removing. Unlike
2D's case, ours don't even collide with a real different Abaqus element of
a similar name (there's no risk of confusion with an actual `C3D8FBAR`); they
are simply invented labels.

### 6.2 Do the *implementations* match the technique their own name claims — independent of whether the name is a real Abaqus element

This is the more serious question, and the answer is **no, for three of the
elements audited**, compounding the naming problem: even the elements whose
*name* is real Abaqus (`C3D8I`, `C3D10M`) or a real, named academic
technique (`C3D4_ANP` → Bonet & Burton 1998) do not implement what their
own name claims.

- **`C3D8I` (our `c3d8_eas_tl_numba.py`/`c3d8_eas_jax.py`)**: real Abaqus
  name, real 13-DOF incompatible-mode formulation *conceptually* (9 strain
  modes + condensation matrices are present in the code) — but §2.1 already
  measured that the enhanced strain is never used to compute stress/f_int.
  **Separately worth flagging here**: even if that were fixed, Abaqus's own
  Theory Guide (TG §3.2.5, V(doc), quote reused from the 2D session's fetch)
  explicitly warns that accumulating incompatible modes against the **total**
  deformation gradient is "a fatal flaw in the formulation for problems
  involving large incremental deformations," and that Abaqus's actual fix is
  to express the enhancement **incrementally** (against ΔF each increment),
  not accumulate a total enhanced F. Our kernel is Total Lagrangian with no
  incremental-frame machinery at all (§0 of the companion interface-alignment
  doc: "3D is purely Total Lagrangian... no `F_n`, no push-forward"). **So a
  correct fix for §2.1 cannot simply "turn alpha on" against the current TL
  formulation** — it would reproduce the exact fatal-flaw case Abaqus's own
  documentation identifies, at large fold rotations, unless the incremental
  (UL-style, config-ledger-based) approach the 2D session already built
  (`element/kinematics/frame.py`) is adopted first. This is a second,
  independent reason (beyond the interface-alignment doc's own §3.1
  argument) to give 3D the config ledger before touching this element.
- **`C3D4_ANP` (our `c3d4_anp_numba.py`/`_jax.py`)**: read in full this pass.
  It is a **plain, single-point-integrated, fully compatible linear
  tetrahedron** — standard `B`-matrix, `strain = B@u`, `stress = C@strain`,
  `f_int = B^T@stress@Ve`. There is **no nodal-volume computation, no
  pressure averaging across elements sharing a node, no deviatoric/
  volumetric stress split** — none of the machinery Bonet & Burton's actual
  ANP method requires (nodal volumes, average nodal pressure smoothing to
  decouple the pressure field from the raw element-wise volumetric strain).
  This is the exact same defect *pattern* as §2.1 (a name claiming a
  locking-mitigation technique; code implementing none of it) on a
  completely different element family — three occurrences of the identical
  pattern now (`C3D8I`, `C3D4_ANP`, and next bullet) is enough to call it
  systemic, not incidental. (Not independently re-verified against
  `DynamicSolver3D` runtime, since this element is dispatch-dead per §2.6 —
  the finding is from direct code reading, which is sufficient here: there
  is no ANP machinery in the file to have missed by reading too little of
  it.)
- **`C3D10M` (our `c3d10m_numba.py`/`_jax.py`)**: read in full this pass. It
  is a **plain, fully-integrated, unmodified 10-node quadratic tetrahedron**
  — standard quadratic Tet10 shape functions, the standard 4-point Gauss
  rule, no hourglass control code anywhere in the file, no "modified"
  interpolation of any kind. This is bit-for-bit the same formulation shape
  as a plain `C3D10`, just relabeled `C3D10M`. Real Abaqus `C3D10M` is a
  materially different element from `C3D10` — the name change and "with
  hourglass control" qualifier are not decorative (AUG §28.1.4, V(doc)).
  Also dispatch-dead per §2.6, so not independently re-run at solver level.

**Summary judgment for §6**: of the six 3D element type-strings this
codebase names after Abaqus or a named published technique, **two are not
Abaqus elements at all** (§5.1) and **three of the remaining four implement
a plainer/naive formulation than the one their name claims** (§5.2,
`C3D8I`/`C3D4_ANP`/`C3D10M`) — only `C3D8H`'s *name* is both real and,
per §2.4, its *actual* single-field B-bar implementation is a real,
recognized technique in its own right (mean-dilatation/uniform-strain,
related to but not identical to Abaqus's two-field hybrid) — still a
naming mismatch (§2.4 already covers this), but at least the technique it
does implement is a legitimate one, unlike the three cases above where the
claimed technique is simply absent.

## 7. Not independently investigated this pass

- `c3d4_anp_jax.py`/`c3d4_anp_numba.py` (Tetra4 ANP) and
  `c3d10m_jax.py`/`c3d10m_numba.py` (Tetra10) — confirmed unreachable via
  `DynamicSolver3D` dispatch (§2.6), so no live model currently exercises
  them; not worth a C6-C11 sweep until they are actually wired into
  `_setup_numba_topology`'s classification table.
- JAX≡Numba cross-check (C10) for F-bar (`c3d8_fbar_jax.py` exists,
  parallel to the Numba TL kernel) — not run this pass; given the F-bar
  Numba kernel's own defect (§2.5) is a tangent-only issue, a JAX comparison
  would mainly be informative if JAX's tangent is *complete* (unlike EAS,
  where the JAX/Numba comparison was the smoking gun) — worth doing as a
  quick follow-up.
- A repeat of §2.1-2.5's C8 checks after the concurrent session's
  `stress_init` edit fully lands (once `element3d` imports cleanly again) —
  everything here was measured either via the source-patching workaround
  (`scratch/audit3d/_bootstrap.py`) or via `DynamicSolver3D`'s except-guarded
  fallback path; a final confirmation against the finished, importable
  tree is worth one more pass.
