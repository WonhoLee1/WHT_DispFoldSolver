# Precise PDASS / semi-smooth Newton design for 3D contact (2026-09-15)

**Status: design only. No `dispsolver/` code changed in this session.** This
follows `dev_log/contact_beyond_abaqus_research_20260915.md`'s
recommendation to treat the penalty-regularized Alart-Curnier/PDASS
promotion as the concrete, well-scoped follow-on. That document already
established (§1) that `SurfaceContactConstraint3D.assemble()`'s augmented
branch evaluates the exact Alart-Curnier NCP function and that
Hüeber & Wohlmuth (2005) prove PDASS ≡ semi-smooth Newton with **local**
(per-node) elimination of the multiplier. This document derives, from
scratch, exactly what that equivalence means for this codebase's actual
variables (`gap`, `lam`, `k_contact`, `k_diag`, `penetration`) and its
actual loop structure (`solve_step()`, `solve_step_augmented()`), rather
than restating the literature at a conceptual level.

**Headline finding, stated up front because it resolves the task's central
question precisely**: the per-node residual/tangent formula in
`assemble()`'s active branch is **already exactly correct** and does not
change. The entire algorithmic content of promoting today's outer-Uzawa
augmented Lagrangian to genuine PDASS/semi-smooth Newton is a **one-line
relocation** of an already-correct, already-tested method call
(`update_augmented_multipliers()`) from `solve_step_augmented()`'s outer
loop (called once per fully-converged inner Newton solve) into
`solve_step()`'s own Newton loop (called once per Newton *iteration*, at
both points that loop currently commits a new accepted iterate). No new
math is required in `assemble()`. This is the Simo & Laursen (1992)
single-loop augmented-Lagrangian construction, arrived at independently
here via the NCP-function derivation in §1, then confirmed to match.

---

## 0. Sources actually used (not just cited abstractly)

- S. Hüeber, B. Wohlmuth (2005), *A primal-dual active set strategy for
  non-linear multibody contact problems*, CMAME 194, 3147-3166. Confirmed
  bibliographic details (title/venue/pages) via web search this session;
  full text paywalled — the *formulation properties* used below (local
  per-node elimination, no saddle-point solve for the PDASS/penalty form)
  are re-derived independently in §1 from the underlying NCP function
  already coded in this repository, not copied from an inaccessible PDF.
  This is stated as an honest gap, matching this project's own citation
  discipline (`contact_beyond_abaqus_research_20260915.md` §6).
- M. Hintermüller, K. Ito, K. Kunisch (2002/2003), *The primal-dual active
  set strategy as a semismooth Newton method*, SIAM J. Optimization 13(3),
  865-888 (DOI 10.1137/S1052623401383558). This is the actual origin of
  the PDASS = semismooth-Newton equivalence for a *bound-constrained
  obstacle problem*; Hüeber & Wohlmuth extend it to contact. §1 below
  re-derives the obstacle-problem NCP function from this paper's own
  known structure (`C(u,λ) = λ - max(0, λ - c(u-ψ))`) and maps it onto
  this codebase's gap/penetration sign convention term by term — full
  text also not fetched (paywalled/blocked), so the mapping is this
  document's own derivation, cross-checked against the codebase's already
  -measured, already-passing behavior (`assemble()`'s existing sign
  convention, `tests/test_augmented_lagrangian_contact.py`'s measured
  numbers) rather than against the paper's own worked example.
- R. Qi, J. Sun (1993), *A nonsmooth version of Newton's method*, Math.
  Programming 58, 353-367 — generalized-Jacobian/semismooth-Newton
  convergence theory, used in §1.4 for the switching-point question.
- J.C. Simo, T.A. Laursen (1992), *An augmented Lagrangian treatment of
  contact problems involving friction*, Computers & Structures 42(1),
  97-116 — the origin of nesting the Uzawa multiplier update inside the
  Newton loop itself (rather than around a fully-converged outer solve)
  for contact specifically. Found and confirmed via search this session;
  cited here because §2's control-flow recommendation is *exactly* this
  construction, independently re-derived, not a novel proposal.
- Direct reads of `dispsolver/constraint3d/surface_contact3d.py` (full
  file, 479 lines), `surface_contact3d_deformable.py` (full file, 245
  lines), `pressure_overclosure.py` (full file), and
  `dispsolver/solver3d/dynamic3d.py` lines 404-856 (`assemble_system`,
  `_contact_active_set`, `solve_step`, `solve_step_augmented`), plus
  `tests/test_augmented_lagrangian_contact.py` (full file) and the
  `benchmark_element/benchmark_3d_contact.py` augmented-Lagrangian call
  site — all read this session, line numbers below are exact.

---

## 1. The exact residual/tangent, per node, active and inactive — derived, not asserted

### 1.1 The NCP function, mapped onto this codebase's own sign convention

This codebase's convention (`pressure_overclosure.py`'s module docstring,
`surface_contact3d.py`'s `assemble()`): `gap = (x_s - x_master)·n - offset`
(≥0 = separated, <0 = penetrating), `penetration = -gap` (≥0 while
penetrating). The one-sided normal contact complementarity condition is

    p_i ≥ 0,   gap_i(u) ≥ 0,   p_i · gap_i(u) = 0                     (C)

(p_i = normal contact pressure at slave node i, u = global displacement
vector). Hintermüller-Ito-Kunisch's NCP function for an obstacle
constraint `v ≥ ψ` with multiplier `μ` is `C(v,μ) = μ - max(0, μ - c(v-ψ))`
for any fixed `c>0`; substituting the contact analogue `v-ψ → gap_i(u)`,
`μ → p_i` gives, **for this codebase's own variables**:

    C_i(u, p) := p_i - max(0, p_i - c·gap_i(u))
               = p_i - max(0, p_i + c·penetration_i(u))                (1)

using `c = self.k_contact` and `penetration_i(u) = -gap_i(u)`. **This is
exactly the quantity already computed in `get_active_set()` (line 245,
`p = self._lam[s_nid] + self.k_contact * (-gap)`) and in `assemble()`
(line 302, `f_mag = self._lam[s_nid] + self.k_contact * (-gap)`)** — the
codebase's `self._lam[s_nid]` already plays the role of `p_i`, and the
argument of `max(0, ·)` inside (1) is already computed and stored as
`f_mag` before the `if f_mag <= 0.0: continue` check. This confirms, by
direct term-by-term substitution rather than by citing that it exists,
`contact_beyond_abaqus_research_20260915.md`'s §1 claim.

### 1.2 Solving `C_i = 0` for `p_i` given `u` — the local elimination, worked out

Set `C_i(u,p)=0` and case-split on the sign of the argument of `max(0,·)`:

- **Case A — `p_i + c·penetration_i(u) > 0` (ACTIVE, exactly today's
  `f_mag > 0` check).** Then `max(0,·)` takes the linear branch, and
  `C_i=0` becomes `p_i - (p_i + c·penetration_i(u)) = -c·penetration_i(u) = 0`.
  Since `c>0`, this forces **`penetration_i(u) = 0` exactly** — i.e. the
  active branch of the *exact* NCP equation is a zero-compliance geometric
  constraint on that node's normal displacement component, not a
  finite-stiffness spring. `p_i` itself is left **undetermined by (1)** in
  this branch — Hüeber-Wohlmuth's "read off directly from the node's own
  equilibrium equation" means exactly this: `p_i` becomes whatever value
  makes that node's own row of the momentum-balance residual vanish, i.e.
  a Lagrange-multiplier/reaction-force role, not a coded penalty-force
  formula.
- **Case B — `p_i + c·penetration_i(u) ≤ 0` (INACTIVE, today's
  `f_mag ≤ 0` check).** `max(0,·)=0`, so `C_i=0` gives **`p_i = 0`**
  exactly — matching today's `continue` (zero force, zero stiffness).

**This is the answer to the crux question, stated precisely:** the *exact*
PDASS active branch is **not** the same as this codebase's finite-`k_contact`
hard-penalty formula — it is a genuine zero-compliance Dirichlet constraint
on a (generally 3-DOF-coupled, `u_s·n = const`) linear combination at that
node, which this codebase cannot represent as a plain single-DOF
`fixed_dofs` entry except in the axis-aligned-plane special case, and which
would otherwise require either (a) a real per-active-node scalar unknown
(a genuinely *local*, block-diagonal — not global — saddle-point
augmentation, one extra row/column per active node, still cheap because
the coupling is 1×1 per node) or (b) driving the *effective* `k_diag` to a
very large value only on active nodes each iteration, which reintroduces
exactly the conditioning problem augmented Lagrangian exists to avoid.
**Neither (a) nor (b) is recommended** — see §1.3.

### 1.3 The regularized (Simo-Laursen) resolution this codebase should use instead

Keep the existing finite-`k_contact`, displacement-only architecture
exactly as coded, but do not treat `p_i` (`self._lam[s_nid]`) as frozen
for an entire outer cycle. Instead, update it **once per Newton iteration**
using its own already-coded fixed-point map

    p_i^{(k+1)} = max(0, p_i^{(k)} + ω·c·penetration_i(u^{(k)}))       (2)

(exactly `update_augmented_multipliers()`'s existing
`p_target = max(0, lam_old + k_contact*(-gap)); lam_new = lam_old + omega*(p_target-lam_old)`,
unchanged) — and then, **using this just-updated, now-fixed `p_i^{(k+1)}`**,
assemble the *next* Newton iterate's residual/tangent with the existing
formula:

    f_mag_i(u) = p_i^{(k+1)} + c·penetration_i(u)          (active branch, f_mag_i>0)
    d(f_mag_i)/d(gap_i) = -c   ⇒  k_diag = c                (unchanged — see §1's own comment
                                                               deriving why the two sign flips cancel)
    f_mag_i(u) = 0                                          (inactive branch, f_mag_i≤0)

**These are literally the same lines already in `assemble()` (lines
290-317) — nothing there changes.** What changes is *when* `p_i` is
written: today, `update_augmented_multipliers()` is called only from
`solve_step_augmented()`, once per fully-converged inner `solve_step()`
call (an *outer* Uzawa cycle). The redesign calls the *same, unmodified*
`update_augmented_multipliers()` once per Newton *iteration*, from inside
`solve_step()`. This is not an approximation of exact PDASS invented for
this document — it is precisely the Simo & Laursen (1992) construction
(nest the multiplier update inside the Newton loop rather than around it),
and it is precisely what Hüeber-Wohlmuth's "one single Newton iteration
handles the active-set search and the multiplier together" property
reduces to in a scalar-penalty, no-extra-DOF architecture: the multiplier
still only ever needs its own node's `gap_i(u)`, updated at Newton's own
rate instead of at Uzawa's (much slower) outer rate.

**Honest caveat, stated because the task requires it:** this is the
*regularized* PDASS, not the exact one from §1.2 — a single Newton
iterate still carries an `O(1/c)` compliance bias in `penetration_i`,
identical in form to plain penalty. The zero-compliance property is
recovered only once the per-iteration multiplier sub-iteration (2) has
itself converged, which — per §4's falsifiable prediction — should now
happen *within* one `solve_step()` call's own iteration budget, not via a
separate outer wrapper.

### 1.4 The switching-point question — generalized Jacobian at `f_mag_i = 0`

`max(0,x)` is non-differentiable only at `x=0`. Qi & Sun's (1993)
semismooth-Newton theory: for a locally Lipschitz `F`, Newton's method
using **any** element of the generalized (Clarke) Jacobian at each
iterate converges locally superlinearly provided `F` is semismooth at the
solution and every generalized-Jacobian element there is nonsingular; at
points where the true solution has **strict complementarity** (`p_i*=0`
XOR `gap_i(u*)=0`, never a genuine tie), the *specific* subgradient
chosen at an intermediate iterate that happens to land exactly on the kink
does not affect the asymptotic rate — it only needs to be *a* valid
element of `[0,1]` (Clarke's generalized Jacobian of `max(0,x)` at
`x=0`), applied consistently.

This codebase's own convention, `if f_mag <= 0.0: continue` (line
303/316), places the boundary point `f_mag_i=0` on the **inactive** side
— i.e. it implicitly selects the subgradient element `0` (not `1`) at
the switch. This is a legitimate, self-consistent choice within `[0,1]`
and **should not be changed**: Qi-Sun's theory gives no basis to prefer
the other endpoint, and changing an already-tested sign/branch convention
here for no measured benefit would repeat exactly the class of defect
`AGENTS.md` §4.14-4.17 spent an entire session cataloguing (silently
changing a convention with real numerical consequences based on
unverified intuition).

**On whether SDI is the right handling at the switch, or whether something
additional is needed:** SDI (`dynamic3d.py` lines 669-680,
`is_sdi = current_active_set != prev_active_set`) is the numerically
correct practical treatment for a different reason than the generalized-
Jacobian choice above: Qi-Sun's theory concerns which *subgradient value*
to use in a **single** iteration's linearization; SDI concerns what to do
about the *actual residual discontinuity* that occurs **across** two
consecutive iterates when the active-set classification flips — a
structural fact about the complementarity condition (§3.3 of
`contact_beyond_abaqus_research_20260915.md` reaches the identical
conclusion for Nitsche and soft-law contact: this discontinuity exists
regardless of which NCP treatment is used). **No additional mechanism is
needed**: SDI already exempts exactly the iteration where the switch
occurs, and — critically for §2 below — SDI's own `current_active_set`
computation already reads `self._lam` fresh every call, so once `p_i` is
updated every iteration (§1.3), an iteration where the active set flips
*because the multiplier moved* (not just because `u` moved) is
automatically covered by the same, unmodified SDI check. No new "did the
multiplier flip status" tracking is required.

---

## 2. The exact control-flow change to `solve_step()` / `solve_step_augmented()`

### 2.1 Where the multiplier update must NOT be placed — a precision point the naive design gets wrong

A first instinct is to put the `p_i`-update as a side effect inside
`SurfaceContactConstraint3D.assemble()` itself, guarded by
`self.augmented_lagrange`. **This is wrong and must not be done**: within
one Newton *iteration* of `solve_step()`, `assemble_system()` — and hence
`constraint.assemble(u)` — is called **multiple times** against different
trial displacement vectors that are not all accepted: once for the
"official" residual/tangent at the current accepted `u_k` (line 652),
once more for the iteration-1 full-step-acceptance probe (line 710), and
up to **six** more times during the Armijo line search (line 749, one
per halving). If the multiplier update fired inside `assemble()`, `p_i`
would be advanced once per *call*, not once per *accepted iterate* — a
result that depends on how many line-search halvings happened to run
(an incidental implementation detail), which is neither reproducible nor
theoretically justified, and risks exactly the kind of divergence
`update_augmented_multipliers()`'s own docstring already documents
happening once before (the pre-2026-09-13 sign bug's 1.422×/cycle
geometric blow-up) for an unrelated reason. `assemble()` must remain a
pure function of `(u, self._lam)` with **no** mutation of `self._lam` —
exactly as coded today.

### 2.2 The actual, minimal change

Add one new `solve_step()` parameter, `augment_relaxation: float = 1.0`
(matching `solve_step_augmented()`'s own existing default), and insert one
call **at each of the two places `solve_step()` currently commits a newly
accepted iterate** to `u_k` — not inside `assemble()`:

1. **Iteration-1 fast path**, `dynamic3d.py` line 712 (`u_k = u_trial`,
   immediately before the `continue`): insert the update call using this
   `u_trial` as the newly-committed iterate.
2. **Line-search-accepted path**, line 769 (`u_k = best_u`, after the
   Armijo loop): insert the same call using `best_u`.

Both call sites reduce to the same one-line helper:

```python
for c in self.constraints:
    if getattr(c, "augmented_lagrange", False):
        c.update_augmented_multipliers(u_k, omega=augment_relaxation)
```

(`u_k` here is whichever local variable was just assigned — `u_trial` at
site 1, `best_u` at site 2.) **`update_augmented_multipliers()` itself is
not modified** — it already computes exactly `p_target = max(0, lam_old +
k_contact*(-gap)); lam_new = lam_old + omega*(p_target-lam_old)` (§1.3's
equation (2)) and already mutates `self._lam` in place. No new state is
needed beyond what `SurfaceContactConstraint3D` already carries.

Consequence for `current_active_set`/SDI (line 673, evaluated
*immediately after* this point on the *next* loop pass): it will now read
the just-updated `self._lam`, so `is_sdi` correctly reflects an active-set
change driven by the multiplier update as well as by `u` — automatically,
with no change to `_contact_active_set()` or the SDI bookkeeping itself
(§1.4).

### 2.3 What happens to `solve_step_augmented()`

Its outer `for augment_iter in range(1, max_augment_iters+1): ...` loop
(lines 834-850) becomes redundant for correctness once §2.2 is in place:
after `solve_step()` returns converged, `update_augmented_multipliers()`
has already been driving `p_i` toward its fixed point every iteration, so
`max_pen` at the first outer check should already satisfy `< augment_tol`
essentially always (§4's falsifiable prediction). **Recommendation:
keep `solve_step_augmented()` as a thin wrapper** (do not delete it —
`benchmark_element/benchmark_3d_contact.py` calls it directly), but change
its role from "the mechanism that converges augmentation" to "a
diagnostic pass-through plus a safety net": call `self.solve_step(dt=dt,
f_ext=f_ext, max_iters=max_iters, max_sdi_iters=max_sdi_iters,
augment_relaxation=augment_relaxation)` once, and only re-enter the
`for augment_iter in ...` loop if `max_pen >= augment_tol` after that
single call (which §4 predicts should not trigger on the existing
regression tests). This keeps the public signature and return tuple
`(converged, n_augment_cycles, last_inner_iters)` unchanged for the
existing benchmark call site while making `n_augment_cycles == 1` the
expected, checked outcome rather than an accident.

---

## 3. Composing with `DeformableSurfaceContactConstraint3D`

That class currently has **no** `augmented_lagrange`/`_lam` machinery at
all (module docstring states this explicitly). To add PDASS-style
augmentation:

1. `__init__`: add `self.augmented_lagrange: bool = False` (constructor
   parameter, default off, matching the rigid sibling and this project's
   "off-by-default new mechanism" pattern, `AGENTS.md` §4.13), and
   `self._lam: Dict[int, float] = {nid: 0.0 for nid, *_ in self.pairs}`
   (built from `self.pairs` after `_build_pairs()` runs, since pairing —
   not all candidate slaves survive `position_tolerance` — determines
   which nodes actually need a multiplier entry).
2. `get_active_set()`: add the identical augmented branch the rigid
   sibling has (line 238-247) — `p = self._lam[s_nid] + self.k_contact *
   (-gap); active if p > 0` — using this class's own
   `_current_gap_and_normal()` for `gap` instead of the rigid plane's
   direct dot product. No other change.
3. `assemble()`: add the identical augmented branch (mirroring lines
   290-306) computing `f_mag = self._lam[s_nid] + self.k_contact *
   (-gap)`, `k_diag = self.k_contact`, else fall through to the existing
   `self.law.evaluate(penetration)` branch when `augmented_lagrange` is
   off. The 5-node stencil/weights (`[1, -N1, -N2, -N3, -N4]`) and the
   `k_ab = k_diag * w_a * w_b` stiffness pattern (lines 191-230) are
   **completely unchanged** — the augmented branch only changes how the
   scalar `f_mag`/`k_diag` pair is computed, exactly as it does in the
   rigid-plane sibling; the stencil doesn't know or care which branch
   produced them.
4. Add `update_augmented_multipliers(u, omega=1.0)`, a near-verbatim port
   of the rigid sibling's version (lines 427-478), substituting
   `self._current_gap_and_normal(u, s_nid, m_face, xi, eta, sign)` for the
   rigid version's direct `gap = dot(xs - point, normal)` computation.
5. Wire the same §2.2 call sites in `solve_step()` — no change needed
   there beyond what §2 already specifies, since the loop already
   iterates `self.constraints` generically via `getattr(c,
   "augmented_lagrange", False)`, not by class name.

**Does PDASS's local-elimination argument interact with this class's own
already-documented modified-Newton simplification (frozen, undifferentiated
normal in the tangent)?** No — the two are orthogonal, and this is worth
stating precisely rather than asserting: §1.2's local-elimination argument
is about **which unknown (`p_i` vs. a global multiplier vector) gets
solved how** — it says nothing about how exactly `gap_i(u)`'s own
dependence on `u` is differentiated. The modified-Newton choice already
made in this file (module docstring: differentiate `d(gap)/du` through
the shape-function-weighted position term, but hold `d(normal)/du = 0`)
is a **separate, prior** simplification of that same `gap_i(u)` function
that both the plain-penalty and the newly-added augmented branch consume
identically — swap in whatever (possibly-frozen-normal) `gap` this file's
`_current_gap_and_normal()` already returns, and every formula in §1.3
composes unchanged. The two simplifications stack without any additional
correctness argument required, for the same reason a change to which
material law feeds a `B`-operator doesn't require re-deriving the
`B`-operator itself.

---

## 4. Verification plan — precise, falsifiable predictions

**Test 1 — `tests/test_augmented_lagrangian_contact.py::test_augmented_lagrangian_converges`,
re-run first.** This test's *existing* loop (10 manual
`solver.solve_step()` + `contact.update_augmented_multipliers()` calls in
Python, bypassing `solve_step_augmented()` entirely) exercises the
primitives directly and should be **numerically unaffected** by this
redesign (it already does, by hand, what §2.2 automates) — use it as a
non-regression check that `update_augmented_multipliers()` itself is
untouched (still `penetrations[-1] < 0.1 * penetrations[0]`, same ~0.771
ratio/cycle). **Add a new test** calling `solver.solve_step_augmented(dt=1.0,
max_iters=60)` (or a direct single `solve_step(..., augment_relaxation=1.0)`)
**exactly once** on the same fixture (`_build_cube_contact("AUGMENTED_LAGRANGE")`)
and assert:
  - `max_penetration` after that **single** call is `< 0.1 *` the
    plain-penalty case's converged penetration at the same `k_contact`
    (the falsifiable signature that augmentation actually happened,
    inline, without the outer loop).
  - The returned inner iteration count is small and **comparable to a
    single plain (non-augmented) `solve_step()` call's own iteration
    count on the same fixture** (expect roughly the same order, e.g.
    single digits to ~20) — **not** the ~10-cycle × (per-cycle inner
    iters) total work the current outer-loop test structure implies.
    A result that only converges by *also* letting
    `solve_step_augmented()`'s outer loop run several `augment_iter`
    cycles (i.e. `n_augment_cycles > 1` in the return tuple) is the
    specific, checkable sign the redesign did **not** achieve its
    purpose — treat `n_augment_cycles == 1` as the pass/fail bar per §2.3,
    not merely "eventually converges within `max_augment_iters`."

**Test 2 — `benchmark_element/benchmark_3d_contact.py`'s Hertz
curved-block, `AUGMENTED_LAGRANGE` row** (`_solve_with_cutback(...,
augmented=True)`, calling `solver.solve_step_augmented(dt=1.0,
max_iters=max_iters)` per load step). Re-run and check: (a) the contact
half-width-vs-Hertz-reference ratio does not regress from whatever this
row's already-measured baseline is; (b) the **total** number of inner
Newton iterations summed across the whole load ramp drops substantially
relative to the pre-redesign run — since the outer-cycle multiplication
disappears, the expected new total should approach the same order as the
`PENALTY` (non-augmented) row's own total iteration count on the same
benchmark, not a multiple of it.

**Test 3 — the two-stacked-cubes `DeformableSurfaceContactConstraint3D`
case referenced in the task brief.** Run once in plain-penalty mode at a
representative `k_contact` to record a baseline `max_penetration`; then
run again with §3's augmented branch enabled at the **same** `k_contact`
and confirm `max_penetration` drops well below the plain-penalty baseline
(demonstrating the `O(1/k)` bias is actually being removed, not just that
Newton still converges) — and re-run `sanity_report()`/
`check_region_tracking()`-class checks (`AGENTS.md` §4.9) to confirm the
two bodies remain genuinely coupled through contact, not just
independently converging to their own prescribed states. A run that
"converges" (Newton succeeds) but shows `max_penetration` unchanged from
the plain-penalty baseline is the specific, checkable sign that the new
branch in `assemble()`/`get_active_set()` was wired incorrectly (e.g. the
augmented branch never actually gets selected), matching this project's
own standing discipline (`AGENTS.md` §4.9) that Newton convergence alone
is not evidence of correctness.

---

## 5. Honest gaps in this design pass

1. Hüeber & Wohlmuth (2005) and Hintermüller-Ito-Kunisch (2002/2003) full
   texts were not accessible this session (paywalled); §1's derivation of
   the NCP function and its case-split is this document's own re-derivation
   from the known structure of the obstacle-problem NCP function and this
   codebase's already-coded `assemble()`/`get_active_set()` formulas, cross
   -checked for self-consistency rather than against the papers' own
   worked proofs.
2. §1.3's regularized (per-iteration-Uzawa) construction is attributed to
   Simo & Laursen (1992) by structural resemblance (confirmed the paper
   exists, title/venue/year, via search) — its own full numerical
   convergence-rate results were not independently re-read this session.
3. The exact PDASS form (§1.2, zero-compliance active branch via a true
   local per-node extra unknown) is characterized but deliberately **not**
   recommended for implementation here — if a future task specifically
   wants zero-compliance contact (not just augmented-Lagrangian-quality
   convergence), that would need a real design pass of its own (a
   block-diagonal 1-extra-DOF-per-active-node augmentation of the linear
   system), not an extension of this document's recommendation.
