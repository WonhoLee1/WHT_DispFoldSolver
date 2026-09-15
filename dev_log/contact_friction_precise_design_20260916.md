# Precise design: penalty-regularized Coulomb friction for 3D contact (2026-09-16)

**Status: design only. No `dispsolver/` code changed in this session.** Phase 3
of the contact roadmap (`dev_log/contact_development_report_20260915.md`
"다음 단계 (우선순위 순)": item 1 Dual Mortar/surface-to-surface and item 2
nonlinear/soft penalty laws are already implemented and verified; "3. Phase
3: 마찰(Coulomb friction) — 아직 시작 안 함" is next, per the project owner's
own "우선순위로 진행" instruction).

Read in full this session, and built on directly rather than restated:
`benchmark_element/reference_abaqus_docs/ctc_friction.txt` (2553 lines;
quoted below by its own section headings and exact formulas, not
paraphrased), `dispsolver/constraint3d/surface_contact3d.py` (570 lines),
`dispsolver/constraint3d/surface_contact3d_deformable.py` (376 lines),
`dispsolver/constraint3d/surface_contact3d_s2s.py` (491 lines),
`dispsolver/constraint3d/pressure_overclosure.py` (172 lines),
`dev_log/contact_pdass_precise_design_20260915.md` (full),
`dev_log/contact_surface_to_surface_precise_design_20260915.md` (full, esp.
its "Deferred entirely" section), `dev_log/contact_stabilization_precise_
design_20260915.md` (full), `dev_log/contact_abaqus_grade_design_20260915.md`
(the `k_ref`/`softness_scale` auto-derivation sections, §A.1/§B.7), and
`dispsolver/solver3d/dynamic3d.py` lines 380-880 (`assemble_system`,
`_contact_active_set`, `_update_contact_chatter_state`, `solve_step`,
`solve_step_augmented`'s docstring) — read directly this session, not taken
from a predecessor doc's line numbers, per the task's own caution that those
shift session to session. Every variable name below (`gap`, `penetration`,
`f_mag`, `k_diag`, `self._lam`, `k_contact`, `N1..N4`, `xi`, `eta`, `sign`,
`self.pairs`) is that file's own.

**Headline finding, stated up front because it resolves the task's central
questions precisely:**

1. Friction needs its **own** persistent per-node state (an "elastic slip"
   history, analogous to a plastic internal variable), and that state must
   be committed at a **different** set of call sites than
   `update_augmented_multipliers()`/`update_gate_state()` use — not the
   three per-**iteration** commit points, but the three
   `assemble_system(..., update_state=True)` commit points, i.e. only on a
   genuinely **converged increment** (§5). Reusing the PDASS per-iteration
   convention for friction history would be a correctness bug, not a design
   choice — it would let a rejected/non-converged Newton trial permanently
   corrupt the physical slip history, exactly the class of error this
   project's own `elem_sdvs`/`update_state` convention already exists to
   prevent for material plasticity.
2. The stick/slip switch is mathematically the **same return-mapping
   structure** this codebase already implements for `J2Plasticity`
   (`material/plastic_jax.py`) — trial elastic force, a yield-like surface
   `|f_t| <= mu*p`, and a radial return when it's violated. This is not an
   analogy of convenience; it is derived below (§2) term-for-term and used
   to write the consistent tangent by reusing the same
   "elastic-projector-minus-orthogonal-correction" form the plasticity
   tangent already has.
3. Frame consistency for the deformable master face (§4) is answered by
   **convected (parametric) anchor tracking** — storing the friction
   "stuck point" as its own `(xi_anchor, eta_anchor)` material coordinate on
   the master face, evaluated against the *current* deformed configuration
   every call, exactly the way `xi_m, eta_m` already are for the normal
   gap. This sidesteps the AGENTS.md §4.14-class defect (a fixed *global*
   anchor would misread rigid rotation of the master face as spurious slip)
   without inventing any new geometric primitive.
4. Recommended first-pass scope: **`SurfaceContactConstraint3D` (rigid
   plane) and `DeformableSurfaceContactConstraint3D` only.**
   `SurfaceToSurfaceContactConstraint3D` (dual-mortar) is explicitly
   deferred — its own module docstring and predecessor design doc already
   defer friction there, for reasons that get *harder*, not easier, once
   friction's own state-history requirement (point 1) is layered on top of
   an already-aggregated weak gap (§8).

---

## 0. What Abaqus's own source actually says, quoted precisely

From `ctc_friction.txt` (line numbers as extracted from the plain-text
dump; section titles are the document's own):

> "Using the Basic Coulomb Friction Model... two contacting surfaces can
> carry shear stresses up to a certain magnitude across their interface
> before they start sliding relative to one another; this state is known
> as sticking. The Coulomb friction model defines this critical shear
> stress, τ_crit, at which sliding of the surfaces starts as a fraction of
> the contact pressure, p... (τ_crit = μp)." (lines 138-166)

> "For a three-dimensional simulation there are two orthogonal components
> of shear stress, τ1 and τ2... Abaqus combines the two shear stress
> components into an 'equivalent shear stress,' τ_eq, for the stick/slip
> calculations... τ_eq = sqrt(τ1² + τ2²)." (lines 180-208)

> "Shear Stress Versus Elastic Slip While Sticking... some incremental slip
> may occur even though the friction model determines that the current
> frictional state is 'sticking'... The relationship shown... is analogous
> to elastic-plastic material behavior without hardening: κ corresponds to
> Young's modulus, and τ_crit corresponds to yield stress; sticking
> friction corresponds to the elastic regime, and slipping friction
> corresponds to the plastic regime." (lines 1747-1769) — **this is
> Abaqus's own text drawing exactly the plasticity analogy point 2 above
> uses**, not an analogy invented for this document.

> "Stiffness Method for Imposing Frictional Constraints in Abaqus/Standard...
> a penalty method that permits some relative motion of the surfaces (an
> 'elastic slip') when they should be sticking... Abaqus continually
> adjusts the magnitude of the penalty constraint to enforce this
> condition." (lines 1788-1806)

> "The stiffness method... requires the selection of an allowable elastic
> slip, γ_crit... Abaqus/Standard calculates γ_crit as a small fraction of
> the 'characteristic contact surface length,' l̄_char... The allowable
> elastic slip is given as **γ_crit = F_f · l̄_char**, where F_f is the slip
> tolerance; the default value of F_f is **0.005**." (lines 1850-1921) —
> the exact formula and default this design's §3 reuses.

> Anisotropic friction section: "if τ_cand lies outside the critical shear
> stress surface, Abaqus/Explicit sets the contact shear stress equal to
> τ_crit on the critical shear stress surface where the normal to the
> critical shear stress surface passes through τ_cand... and sets the
> direction of incremental slip to be normal to the critical shear stress
> surface." (lines 886-906) — for the **isotropic** case this project
> targets, the critical surface is a circle, so "normal to the circle at
> the point nearest τ_cand" degenerates to "the τ_cand direction itself,"
> i.e. the classical radial return direction §2 derives independently below
> is consistent with Abaqus's own general (anisotropic) statement in its
> isotropic special case — cross-checked, not assumed.

Two scope-defining consequences:

1. Abaqus's own default is the **penalty (stiffness) method** for
   Abaqus/Standard, exactly this project's own established mechanism class
   (penalty contact, no new Lagrange-multiplier DOF) — friction should stay
   in that same family, not introduce a mixed formulation.
2. Abaqus's own text frames the stick/slip switch as *literally* an
   elastic-plastic-without-hardening return map. This project already has
   a working, tested implementation of exactly that structure
   (`material/plastic_jax.py`'s finite-strain multiplicative J2 return
   map, `AGENTS.md` §2's table) — reusing that structure's *shape* (not its
   code) is the least-novel, best-precedented way to build this.

---

## 1. The physical model, mapped onto this codebase's own variables

Per active contact node `i` (a normal-contact `SurfaceContactConstraint3D`
slave node with `f_mag > 0`, i.e. `p_i := f_mag` already computed by the
existing normal branch in the *same* `assemble()` call — friction never
recomputes the normal pressure independently, per the task's own framing):

- `n` — the node's current contact normal (already computed:
  `self.normal` for the rigid-plane class, or the per-pair `n` from
  `_current_gap_and_normal()` for the deformable class).
- `(t1, t2)` — an orthonormal basis of the tangent plane at `n` (§4 gives
  the exact construction for each class).
- `s = (s1, s2)` — the **trial elastic tangential slip displacement**,
  in the `(t1, t2)` basis, measured from a persistent per-node **anchor**
  (§4/§5) representing "the material point this node was stuck to the last
  time it was sticking."
- `f_t_trial = k_t * s` (2-vector) — trial tangential penalty force,
  `k_t` derived in §3.
- `p_i` — the node's current normal contact pressure (read, not
  recomputed, from the normal branch already executed this call).

**Stick/slip switch** (Coulomb, Abaqus's own `τ_crit = μp`):

```
if |f_t_trial| <= mu * p_i:                       # STICK
    f_t = f_t_trial = k_t * s
else:                                              # SLIP
    f_t = mu * p_i * (s / |s|)                     # radial return, direction = trial direction
```

This is precisely Abaqus's own elastic-plastic-without-hardening picture
(§0's quote): `k_t` plays the role of the elastic (spring) modulus, `mu*p_i`
plays the role of the (pressure-dependent, hence non-constant) yield
stress, and the SLIP branch is a **radial return** onto the yield circle
— the identical operation `plastic_jax.py`'s return-mapping performs onto
the (smooth, tanh-blended) J2 yield surface, specialized here to an exact
(not smoothed) circular yield surface in 2D deviatoric-tangential space,
because Abaqus's own basic Coulomb model has no smoothing device to
replicate (the project's existing J2 smoothing exists for a *different*
reason — autodiff-friendliness through `eigh`, AGENTS.md §4.3 — which does
not apply here since this derivation is by hand, not `jax.grad`).

---

## 2. Consistent tangent — derived, not asserted, and shown to be the same shape as the existing plasticity tangent

Let `s = (s1, s2)`, `|s| = sqrt(s1²+s2²)`, `ŝ = s/|s|`. Two branches:

**STICK** (`k_t*|s| <= mu*p_i`):

```
f_t = k_t * s
d(f_t)/d(s) = k_t * I_2x2                      # isotropic elastic stiffness
```

**SLIP** (`k_t*|s| > mu*p_i`):

```
f_t = mu * p_i * ŝ
d(f_t)/d(s) = (mu*p_i/|s|) * (I_2x2 - ŝ ⊗ ŝ)   # radial-return consistent tangent
```

**This is the exact structural form of a J2-type consistent tangent**:
an isotropic term along the flow direction that vanishes in the SLIP
branch (the material offers zero resistance to *further* sliding in the
direction it is already sliding — same reason plastic flow removes
stiffness along the flow direction), leaving only `mu*p_i/|s|` of
resistance in directions orthogonal to `ŝ` (the "radius" of the yield
circle acting like a secant curvature). Two continuity checks, done
algebraically rather than asserted:

1. **Force is C0 continuous at the switch.** At `|s| = s_crit :=
   mu*p_i/k_t` (the switch point), `f_t_stick = k_t*s_crit*ŝ = mu*p_i*ŝ =
   f_t_slip` — identical. Unlike **normal** contact activation (where the
   force itself jumps from exactly `0` to `k_contact*penetration`),
   friction's stick→slip transition is a force-continuous, tangent-
   discontinuous event — the same C0/C1 distinction between yield-point
   crossing and contact activation in ordinary elastoplastic FE, and it is
   *milder* than the normal-contact discontinuity SDI was built for (§6).
2. **The tangent itself is discontinuous at the switch.** At `|s|=s_crit`,
   `K_stick = k_t*I`, but `K_slip = (mu*p_i/s_crit)*(I - ŝ⊗ŝ) = k_t*(I -
   ŝ⊗ŝ)` — the *radial* (along-`ŝ`) component matches `k_t` exactly at the
   switch (no jump there), but the tangent along `ŝ` itself drops from
   `k_t` to `0` discontinuously. This is the quantity §6 uses to decide
   whether SDI needs extending.

**Coupling to the normal DOFs.** `f_t = mu*p_i*ŝ` in the SLIP branch also
depends on `p_i`, and `p_i` depends on the same node's own normal
displacement (`d(p_i)/d(gap) = k_contact` or `dp_dh` from
`pressure_overclosure.py`'s `evaluate()`). An exact tangent would therefore
carry an off-diagonal normal↔tangential coupling block,
`d(f_t)/d(p_i) * d(p_i)/d(u_normal) = mu*ŝ ⊗ (k_contact * n)`. **This
design recommends dropping that coupling term as a stated modified-Newton
simplification**, exactly the precedent already established twice in this
codebase for a materially harder problem than this one: AGENTS.md §4.4
(dropping `dT8/du`'s rotational geometric-stiffness term for the
corotational-J2 element, "costs a couple extra Newton iterations... not
full quadratic convergence") and
`surface_contact3d_deformable.py`'s own module docstring (freezing
`d(normal)/du` while still differentiating `d(gap)/du` through the
position term). Freezing `p_i` for the tangential sub-block only (never for
the force itself, which always reads the *current* `p_i` fresh each call)
keeps `K` block-structured per node — a normal 3×3-ish block plus an
independent 2×2-in-plane tangential block — instead of a fully coupled
5×5-per-node system, at the cost of a few extra Newton iterations near a
stick/slip transition, the same tradeoff already accepted elsewhere in this
codebase.

---

## 3. Penalty stiffness `k_t` — auto-derived, matching this project's own no-magic-number convention, cross-checked against Abaqus's own formula

**Primary derivation (buildable now, reuses existing machinery
verbatim).** This project's `k_contact` is itself never a bare magic
number when the caller wants auto-derivation — `dev_log/contact_abaqus_
grade_design_20260915.md` §A.1/§B.7 already establishes `k_ref` (a
representative element stiffness, `E_eff * L_elem`, minimum over the slave
nodes' parent elements) and a dimensionless multiplier convention
(`softness_scale`, `1.0` reproducing exactly today's hard-contact default).
Friction should draw from the **same** `k_ref`/multiplier pattern, not
invent a second stiffness basis:

```
tangential_stiffness_ratio: float = 1.0      # dimensionless, mirrors softness_scale's convention
k_t = tangential_stiffness_ratio * self.k_contact
```

`k_contact` here is whatever the normal contact constraint already resolved
(auto-derived from `k_ref` per §A.1/§B.7, or a caller-supplied override) —
**no separate absolute number for friction stiffness needs to exist**;
`tangential_stiffness_ratio=1.0` (default) makes the tangential penalty
exactly as stiff as the normal one, which is the same default relationship
Wriggers (2006, *Computational Contact Mechanics*, already cited in this
project's own theory table for the surface-tie penalty formulation) uses
for isotropic penalty contact with no separate tangential calibration
data.

**Cross-check against Abaqus's own quantity (validation, not the primary
derivation).** Abaqus's allowable elastic slip `γ_crit = F_f * l̄_char`
(§0, `F_f=0.005` default) is a *length*; the implied stiffness at a
representative pressure `p_typ` is `k_t_abaqus = mu*p_typ / γ_crit`. This
project already has a characteristic length in scope for the exact same
purpose — `L_char`/`L_elem`, the same quantity `contact_abaqus_grade_
design_20260915.md` §A.2's nonlinear-penalty breakpoints (`e`, `d`) and
§B.7's `allowable_penetration` default (1% of `L_char`) already reuse. The
recommended **diagnostic**, printed once per contact pair when friction is
enabled (not a silent auto-override of `k_t` above): compute
`s_crit_effective = mu * p_typ / k_t` (using the primary-derivation `k_t`
and a representative `p_typ`, e.g. the converged normal pressure at the
end of the first successful increment) and compare it against
`gamma_crit_abaqus = 0.005 * L_char`; **warn** (matching this project's
existing "explicit warning over silent precedence" pattern, e.g.
`contact_abaqus_grade_design_20260915.md`'s 1000× warning) if the two
differ by more than, say, one order of magnitude — a large discrepancy
means `tangential_stiffness_ratio=1.0` is producing an elastic slip far
outside Abaqus's own documented "generally works very well" default range,
and the caller should consider overriding the ratio, not that anything is
silently wrong.

**Why not literally implement Abaqus's own pressure-adaptive formula
(`k_t := mu*p_i/γ_crit`, recomputed every call)?** Because that makes
`k_t` itself a function of `p_i` — i.e. exactly the normal↔tangential
coupling term §2 already recommends dropping for tractability, but now
smeared into the STICK branch's own stiffness too (today, `p_i`'s
dependence on `u` is confined to the SLIP branch's force; making `k_t`
itself pressure-dependent would reintroduce it into the STICK branch as
well). Keeping `k_t` a fixed scalar per contact pair (derived once from
`k_ref`, not recomputed from the instantaneous `p_i`) is consistent with
how `k_contact` itself already behaves (a fixed penalty parameter, not
re-derived every Newton iteration from the current penetration) and keeps
the STICK branch's tangent exactly `k_t*I` with no coupling to worry about
at all — only the SLIP branch (§2) has the (already-flagged,
already-precedented) frozen-`p_i` simplification.

---

## 4. Frame consistency of the tracked slip — the convected-anchor answer

**Rigid-plane class (`SurfaceContactConstraint3D`).** The master "surface"
is a fixed analytical plane for the constraint's entire life (module
docstring: "the anchor projection... is computed ONCE at construction and
never re-solved"). Its own `(t1, t2)` basis can therefore be constructed
**once**, at `__init__`, via a single Gram-Schmidt step off `self.normal`
and any global axis not parallel to it — no rotation to track, ever. The
anchor is simply a **3D point on the plane** (or, equivalently, 2 scalars
in the fixed `(t1,t2)` basis), and "current slip" is just
`dot(xs_tangential_projection - anchor, t1/t2)` — no frame-consistency
question exists for this class at all, because nothing in its master side
ever rotates.

**Deformable class (`DeformableSurfaceContactConstraint3D`).** Here the
master Quad4 face **does** deform and rotate between increments — this is
exactly the situation AGENTS.md §4.14-§4.17 catalogued three separate
times for element kinematics (a quantity silently invisible on an
axis-aligned/non-rotating reference, wrong once the reference genuinely
rotates). **A naive fixed *global* Cartesian anchor point would repeat
that defect class here**: if the anchor were stored as a fixed 3D point in
space, then rigid rotation of the master face between increments (with
zero real sliding) would still change `xs_tangential_projection - anchor`
in the global frame, and get misread as spurious slip — silently wrong,
not caught by any existing test, because every existing contact test
fixture is close to axis-aligned.

**The fix, reusing only existing machinery**: store the anchor as its own
**parametric (material) coordinate on the master face**,
`(xi_anchor, eta_anchor)`, exactly the same kind of quantity `xi_m, eta_m`
already are for the *slave's own* projection (`_current_gap_and_normal()`,
`surface_contact3d_deformable.py` L198-207). At any later call, evaluate
the anchor's **current physical position** the same way the slave's own
projection point is evaluated — `xm_anchor = N1(xi_anchor,eta_anchor)*q1 +
... `, using the **current** (deformed) master vertex coordinates `q1..q4`
— so the anchor "rides along" with the master face's own deformation and
rotation automatically, with no parallel-transport bookkeeping and no new
geometric primitive: it is `_quad4_shape_functions()` called at a second,
independent `(xi, eta)` pair. Slip is then

```
xs_tan       = xs - dot(xs - xm_proj, n) * n            # slave's own current tangent-plane position (existing xm_proj, n)
xm_anchor    = sum(N_a(xi_anchor, eta_anchor) * q[a] for a in range(4))   # anchor's CURRENT physical position
s_3d         = xs_tan - xm_anchor                         # already lies in (or very near) the current tangent plane
s1, s2       = dot(s_3d, t1), dot(s_3d, t2)               # t1, t2 recomputed fresh each call (Gram-Schmidt off current n)
```

Because both `(xi_m,eta_m)` (slave's own frozen small-sliding pairing) and
`(xi_anchor,eta_anchor)` (friction's own, separately-committed anchor, §5)
are parametric coordinates re-evaluated against the *current* deformed
`q1..q4` every call, a pure rigid rotation of the master face between
increments moves `xs_tan` and `xm_anchor` by the *same* rotation, leaving
`s_3d` — and hence `s1, s2` — unchanged to the same order the existing
modified-Newton frozen-normal approximation already accepts. This is not a
new approximation layered on top of the existing one; it is the same
"material-coordinate, re-evaluated-against-current-geometry" idea the
class already uses for `xi_m, eta_m` and `n`, applied once more to a second
point. **No rate/objective-rate machinery (e.g. `rate_form_adapter.py`'s
Jaumann rate, AGENTS.md §4.16) is needed here** — that machinery exists to
objectively rate-integrate a *stress tensor* through a rotating frame for
a rate-form material; a scalar-pair *material coordinate* has no such
integration problem because it is defined once (at commit) and simply
re-evaluated, not incrementally rotated.

---

## 5. Where the friction state actually lives, and precisely where it commits — the key wiring question, resolved

### 5.1 Why the anchor is a *converged-increment* history variable, not a per-iteration one like `self._lam`

`self._lam` (PDASS's normal multiplier) is a **numerical device** — it has
no physical meaning independent of the algorithm; advancing it many times
while Newton searches for equilibrium *within* one increment is exactly the
point (`contact_pdass_precise_design_20260915.md` §1.3's regularized
per-iteration update). The friction anchor `(xi_anchor, eta_anchor)`,
however, represents **actual physical slip history** — "the material point
on the master surface this slave node was last stuck to." Updating it on
a Newton trial that later gets rejected (line-search halving, a cutback
increment) would permanently corrupt that history with motion that never
actually happened. This is the *exact* reasoning this codebase already
applies to `elem_sdvs` (plastic state): `assemble_system()`'s
`update_state` parameter is `False` for every trial call within
`solve_step()`'s Newton loop and `True` only at the three points a step is
actually accepted as converged (`dynamic3d.py` lines ~765, ~781, ~872 —
"Commit state variables (SDV) for the converged increment"). Friction's
anchor is exactly this same category of state and must follow the exact
same rule, **not** `update_augmented_multipliers()`'s per-iteration rule.

### 5.2 The exact wiring, citing the real call sites

`constraint.assemble(u)` today has no `update_state` parameter at all
(only element assembly does); adding one to every constraint's `assemble()`
signature is unnecessary churn. Following this project's own demonstrated
pattern of small, purpose-specific, `hasattr()`-dispatched methods
(`get_active_set`, `update_gate_state`, `update_augmented_multipliers`),
add one more:

```python
def commit_friction_state(self, u: np.ndarray) -> None:
    """Called ONLY when a step has actually converged -- advances the
    per-node friction anchor (xi_anchor, eta_anchor) for any node that
    slipped, per sec2's radial return. No-op if friction is disabled."""
```

and one new dispatch helper in `dynamic3d.py`, mirroring
`_update_contact_chatter_state()`'s own style (L598-633):

```python
def _commit_contact_friction_state(self, u: np.ndarray) -> None:
    for c in self.constraints:
        if hasattr(c, "commit_friction_state"):
            c.commit_friction_state(u)
```

wired into the **three existing `update_state=True` commit points**
(`dynamic3d.py` `solve_step()`, exact lines read this session):

1. Line ~763-766 (`bc_err`/`rel_r` fast-exit branch): immediately after
   `self.assemble_system(u_k, dt=dt, update_state=True)`.
2. Line ~780-782 (`du_norm` fast-exit branch): immediately after
   `self.assemble_system(self.u, dt=dt, update_state=True)`.
3. Line ~870-873 (budget-exhausted-but-actually-converged branch): after
   the same `assemble_system(..., update_state=True)` call.

**Not** the three `update_augmented_multipliers()`/`update_gate_state()`
sites at lines 792-796, 813-817, 854-857 — those fire once per *accepted
Newton iterate*, which is the wrong cadence for a physical history
variable for exactly the reason §5.1 gives. This is a genuinely different
answer than a naive "friction is just another contact multiplier, wire it
in next to `self._lam`" instinct would produce, and is the most important,
non-obvious design decision in this document.

### 5.3 What `commit_friction_state()` actually does

At commit time, for every currently-active node:

```
if |s| (this node's just-converged trial elastic slip, sec1/sec4) > s_crit:   # SLIP occurred this increment
    # Radial return: shrink the anchor-to-slave vector down to exactly the
    # elastic length s_crit, in the SAME direction -- i.e. move the anchor's
    # own (xi_anchor, eta_anchor) so that, evaluated against the NOW-CONVERGED
    # geometry, the remaining stored elastic slip is exactly s_crit*ŝ, not 0
    # and not the full slipped distance. This is the exact plastic-return
    # analogue of "commit the updated plastic strain, not the full trial
    # strain" already performed for J2 state at every converged increment.
    new_anchor_physical_point = xs_tan - s_crit * ŝ_3d
    (xi_anchor, eta_anchor) <- re-project new_anchor_physical_point onto the
                                 CURRENT master face (a single _project_point_to_quad4
                                 call, already-existing machinery)
# else: STICK all increment -- anchor unchanged, elastic slip carries over
```

The re-projection step reuses `_project_point_to_quad4` verbatim (already
imported by this module) — no new geometric solver is needed.

---

## 6. Does SDI need to fire on a stick/slip transition too?

**Yes, recommended, and cheaply.** §2's continuity analysis shows the
stick→slip switch is force-continuous but tangent-discontinuous — milder
than normal-contact activation (which is force-discontinuous), but a real
tangent jump (`k_t*I` → `k_t*(I-ŝ⊗ŝ)`, losing all stiffness along the
current slip direction) is exactly the kind of event that can cause Newton
to overshoot and re-cross the boundary repeatedly if left unflagged,
matching this project's own already-measured chattering failure mode for
normal contact (`contact_stabilization_precise_design_20260915.md` §0).

**How, without changing `get_active_set()`'s existing contract.**
`get_active_set()`'s return value (a `frozenset` of node ids) is consumed
by PDASS (`_lam` updates don't care about tangential state) and must stay
exactly as it is. Add a **parallel** method, `get_friction_state(u) ->
frozenset`, returning `frozenset((nid, "slip") for nid currently slipping)`
(sticking or inactive nodes simply don't appear — mirroring
`get_active_set()`'s own "only active nodes appear" convention). Extend
`_contact_active_set()`'s SDI comparison to union **both**:

```python
def _contact_full_state(self, u: np.ndarray) -> frozenset:
    normal = self._contact_active_set(u)                    # unchanged, existing method
    friction = frozenset().union(*(c.get_friction_state(u) for c in self.constraints
                                    if hasattr(c, "get_friction_state")))
    return normal | friction

# solve_step(): replace `current_active_set = self._contact_active_set(u_k)`
# with `current_active_set = self._contact_full_state(u_k)` -- same frozenset
# equality comparison, same is_sdi logic, unchanged everywhere else.
```

This is additive and low-risk: with friction disabled (no constraint
exposes `get_friction_state`), `_contact_full_state()` degenerates to
exactly `_contact_active_set()`'s existing return value — byte-identical
old behavior for every non-friction model, matching this project's
established "off-by-default new mechanism" convention.

**What is explicitly *not* recommended in this first pass**: a
friction-specific hysteresis/chatter gate analogous to
`chatter_stabilization`'s normal-contact gate. Reasons, stated honestly:
(a) the transition is milder (force-continuous) than the normal-activation
case that gate was built for, so it is not yet *known* to chatter in
practice; (b) `contact_stabilization_precise_design_20260915.md` §5.4's own
staging discipline ("Stage 2 escalation... build only if Stage 1 measurably
fails") is the precedent to follow — build the plain return-map first,
measure whether stick/slip actually chatters on a real fixture (§9's
verification plan), and only design a dedicated gate if it does. Building
one speculatively now would violate this project's own repeatedly-stated
"measured, not assumed" discipline.

---

## 7. Exact residual/tangent assembly — both classes, both branches

### 7.1 `SurfaceContactConstraint3D` (rigid plane, 3×3-per-node block)

Inside `assemble()`'s existing per-node loop, **after** the existing
normal branch has computed `f_mag` (`p_i`) and *before* the `continue` for
an inactive node (friction only applies where normal contact is active —
`f_mag <= 0` already `continue`s, so friction code sits strictly after
that check, reusing `f_mag` as `p_i` with no recomputation):

```python
s1 = dot(xs_tan - anchor_point, t1)
s2 = dot(xs_tan - anchor_point, t2)
s_mag = sqrt(s1*s1 + s2*s2)
s_crit = mu * p_i / k_t
if s_mag <= s_crit:                                   # STICK
    ft1, ft2 = k_t*s1, k_t*s2
    Ktt = k_t * (outer(t1,t1) + outer(t2,t2))         # 3x3, added to slave's own diagonal block
else:                                                  # SLIP
    ft1, ft2 = mu*p_i*(s1/s_mag), mu*p_i*(s2/s_mag)
    shat = (s1/s_mag)*t1 + (s2/s_mag)*t2               # unit 3-vector in the tangent plane
    Ktt = (mu*p_i/s_mag) * (outer(t1,t1)+outer(t2,t2) - outer(shat,shat))

f_t_3d = ft1*t1 + ft2*t2
f_contact[3*s_idx:3*s_idx+3] += -f_t_3d               # SAME sign convention as f_node = -f_mag*normal
# stiffness scatter: add Ktt into the SAME 3x3 block the normal k_diag*outer(normal,normal) already uses
```

The sign convention (`-f_t_3d`, matching `f_node = -f_mag*self.normal`'s
already-verified `f_int = -F_physical` fix, `surface_contact3d.py`
L437-478) applies identically to the tangential term: it is the same
slave-node-only, no-master-DOF block the normal force already uses, just
summed with a second (tangential) contribution rather than requiring a
separate scatter pass.

### 7.2 `DeformableSurfaceContactConstraint3D` (5-node stencil)

Identical STICK/SLIP branch logic, `t1, t2` recomputed fresh each call
(Gram-Schmidt off the already-recomputed `n` from `_current_gap_and_normal`
— consistent with the class's own existing modified-Newton convention of
recomputing `n` for the force but freezing its derivative for the
tangent), `xs_tan`/`anchor_point` per §4. The force distributes via the
**same 5-node stencil weights** `[1, -N1, -N2, -N3, -N4]` already used for
the normal force (`surface_contact3d_deformable.py` L296-301) — a
tangential contact force obeys the identical Newton's-third-law
distribution the normal one does, over the identical physical patch:

```python
for a, idx_a in enumerate(node_indices):     # [s_idx] + m_idx, existing pattern
    w_a = weights[a]                          # [1, -N1, -N2, -N3, -N4], existing pattern
    f_contact[3*idx_a:3*idx_a+3] += -w_a * f_t_3d
```

Tangent: the same `Ktt` (STICK: `k_t*(t1⊗t1+t2⊗t2)`; SLIP: `(mu*p_i/s_mag)*
(t1⊗t1+t2⊗t2-ŝ⊗ŝ)`) scatters with the existing `k_ab = k_diag * w_a * w_b`
outer-product pattern (L326-335), substituting `Ktt` for the current
`k_diag*outer(n,n)` — i.e. the stencil's rank-1-per-DOF-pair structure is
completely unchanged; only which 3×3 matrix gets distributed by it changes
from a rank-1 normal projector to (in STICK) a rank-2 tangential projector
or (in SLIP) a rank-1 tangential projector.

---

## 8. Compatibility with everything already built this session

**With PDASS/AL (`self._lam`)**: friction only **reads** `p_i` (`f_mag`,
or the AL branch's `self._lam[nid] + k_contact*penetration`, whichever the
normal branch already computed) — it introduces its own, separate
`self._lam_friction`-style state (the anchor, §5), never writes to the
normal `self._lam`. The two are composed exactly the way `contact_pdass_
precise_design_20260915.md` §3 already composes AL with the deformable
class's own modified-Newton normal: independently-owned state consumed at
a shared read point, no interaction to derive beyond noting the plug-in
point is `p_i`, wherever it came from.

**With the chattering-stabilization hysteresis gate**: orthogonal, per §6
— the existing normal-activation gate (`chatter_stabilization`,
`hysteresis_band`) is unaffected; friction's own SDI extension (§6) is a
separate, additive union, not a modification of the existing gate's logic.
A future friction-specific gate, if measurement shows one is needed (§6's
honest deferral), would plug in at the same `get_friction_state()` point
without touching the normal gate.

**With the surface-to-surface (dual-mortar) class**: **stays frictionless
in this design**, matching that class's own module docstring and its
predecessor design doc's explicit deferral (`contact_surface_to_surface_
precise_design_20260915.md` §5's "Deferred entirely: Friction.
`DeformableSurfaceContactConstraint3D`'s own scope is already frictionless...
this design stays frictionless too"). §5's per-node anchor/history
mechanism does not obviously generalize to that class's weak, segment-
aggregated `g_tilde[nid]` — a frictional anchor there would need to be a
weak (dual-basis-weighted) history quantity, aggregated the same two-pass
way `g_tilde` itself is, and the SLIP radial-return direction would need
to be computed from an aggregated *weak* tangential slip rather than a
single point's own. This is a real, separate, harder design problem
(mortar frictional contact is a recognized harder sub-field even in the
literature this project's mortar design already draws from, e.g. Puso &
Laursen's mortar-friction papers extend, not trivially reuse, their own
frictionless mortar contact formulation) — not attempted here, flagged for
a possible future design pass, consistent with this project's staged
discipline.

---

## 9. Staged scope — what is buildable now vs. deferred, stated honestly

### 9.1 Buildable now (Stage 1)

- `SurfaceContactConstraint3D`: `friction_coefficient: float = 0.0`
  (0 = frictionless, exact byte-identical fallback to today's class),
  `tangential_stiffness_ratio: float = 1.0` constructor params; per-node
  `self._anchor: Dict[int, np.ndarray]` (3D point, rigid-plane case, §4);
  one fixed `(t1,t2)` pair built once at `__init__`; the STICK/SLIP branch
  in `assemble()` (§7.1); `get_friction_state()` (§6);
  `commit_friction_state()` (§5.3).
- `DeformableSurfaceContactConstraint3D`: identical parameters; per-node
  anchor stored as `(xi_anchor, eta_anchor)` (§4, convected); `t1,t2`
  recomputed per call from the current `n`; the same STICK/SLIP branch
  distributed via the existing 5-node stencil (§7.2); the same two new
  methods.
- `dynamic3d.py`: `_commit_contact_friction_state()` wired at the three
  `update_state=True` sites (§5.2); `_contact_full_state()` replacing
  `_contact_active_set()` at the one `solve_step()` call site that feeds
  `is_sdi` (§6) — both additive, both no-ops when `friction_coefficient=0`
  or absent.

### 9.2 Deferred, explicitly, not built speculatively

- **`SurfaceToSurfaceContactConstraint3D` friction** (§8) — a separate,
  harder mortar-friction design problem; matches that class's own already-
  stated deferral.
- **A dedicated stick/slip chattering gate** (§6) — build only if Stage
  1's own verification (§10) actually measures chattering on a real
  fixture; not assumed necessary in advance.
- **Un-freezing the normal↔tangential coupling** (§2's dropped
  `d(f_t)/d(p_i)*d(p_i)/d(u_normal)` term) — a genuine full-Newton
  improvement, but adds real complexity (breaks the clean per-node 3×3 /
  5-node-stencil block structure into a larger coupled block); defer until
  the frozen-`p_i` modified-Newton approximation is measured to cost more
  Newton iterations than acceptable on a real fold/slide case, matching
  the precedent AGENTS.md §4.4/§4.10 already set for accepting modified-
  Newton first and only upgrading when measured necessary.
- **Anisotropic friction, slip-rate/pressure/temperature-dependent `μ`,
  the optional shear-stress limit `τ_max`** (all present in `ctc_friction.
  txt` but not requested by the task or by any current example) — real
  Abaqus features, not built without a concrete use case, matching this
  project's own standing discipline against speculative infrastructure.
- **A per-node (rather than per-contact-pair) auto-derived
  `tangential_stiffness_ratio`** — today's design takes one caller-set
  scalar per constraint (mirroring `k_contact`'s own convention); a
  spatially-varying auto-derivation is a separable follow-on, not attempted
  here.

---

## 10. Verification plan — falsifiable predictions

**Test 1 — pure sliding block, known analytical friction force.** A single
deformable cube (reuse `tests/test_augmented_lagrangian_contact.py`'s
`_build_cube_contact` fixture geometry) resting on the rigid-plane
constraint under a prescribed constant normal preload `p_i` (drive the
cube down until `p_i` converges to a known value, e.g. via a fixed
prescribed vertical displacement), then apply a monotonically increasing
prescribed tangential (horizontal) displacement to the cube's top face.

*Falsifiable prediction*: while the applied tangential displacement is
below `s_crit = mu*p_i/k_t`, the reaction tangential force at the contact
node should equal `k_t * s_applied` to within the solver's own convergence
tolerance (a direct, closed-form check, no fitting). Once the applied
displacement exceeds `s_crit`, the reaction tangential force should
plateau at exactly `mu*p_i` (to the same tolerance) and stay flat as the
applied displacement continues increasing — the classic stick-then-slip
signature. A result that keeps growing linearly past `s_crit`, or that
never reaches `mu*p_i`, is the specific, checkable sign the branch logic
or the sign convention (§7) is wired incorrectly.

**Test 2 — frame consistency under master-face rotation.** Using the
deformable class, rigidly rotate the master body's prescribed BC by (say)
30° over several increments with **zero** applied tangential load and
**zero** actual relative sliding (the slave body co-rotates rigidly with
it, e.g. both bodies driven by the same rigid rotation BC). *Falsifiable
prediction*: `s_mag` (computed per §4, convected anchor) should stay at
its pre-rotation value (constant, to numerical tolerance) throughout the
rotation — i.e. **zero spurious slip from pure rotation**. Re-run the same
scenario with a naive fixed-*global* anchor (a quick throwaway
modification, only for this diagnostic, not for production) and confirm
it *does* show spurious growing `s_mag` proportional to the rotation angle
— demonstrating the convected-anchor fix is actually necessary, not merely
plausible, on this codebase's own geometry, the same "measured, not
assumed" standard AGENTS.md §4.14 applies to.

**Test 3 — stick/slip transition and SDI interaction.** Ramp the Test 1
tangential displacement through `s_crit` over many small increments with
`debug_sdi=True`. *Falsifiable prediction*: with §6's `_contact_full_state()`
extension, the increment(s) spanning the stick→slip crossing show
`is_sdi=True` (exempted from the normal convergence check), and the run
converges in a comparable iteration count to a nearby non-transitioning
increment. Re-run with the extension reverted (using only
`_contact_active_set()`, i.e. normal-only SDI) as a baseline comparison;
**prediction**: if the un-extended baseline shows measurably more Newton
iterations or an outright non-convergence at exactly the crossing
increment while the extended version does not, that is the concrete
evidence justifying §6's SDI extension (and, per §6, evidence for or
against needing a dedicated chatter gate — if neither version chatters
across many increments, defer the gate per §9.2; if the un-extended
version chatters but the extended one does not, SDI alone was sufficient
and the gate is unneeded; only if *both* chatter should a dedicated gate be
designed).

**Test 4 — committed-state integrity across a rejected trial.** Force a
deliberate cutback (e.g. an artificially tiny `max_iters` on one increment
so it fails to converge and the caller retries with a smaller `dt`), with
the tangential displacement mid-slip at the moment of the forced failure.
*Falsifiable prediction*: `self._anchor`/`(xi_anchor,eta_anchor)` after the
failed-then-retried increment should be **unaffected** by the rejected
trial's own (larger, never-accepted) slip — confirming §5.2's wiring
(commit only at `update_state=True` sites) actually prevents the
history-corruption failure mode §5.1 predicts would occur if wired at the
per-iteration sites instead. Wiring `commit_friction_state()` at the wrong
call sites (as a deliberate negative-control re-run) should visibly fail
this test — the anchor advancing on the rejected trial and producing a
p_ermanently wrong post-cutback elastic-slip baseline — which is the
concrete, checkable signature of exactly the bug §5.1 argues against.

---

## 11. Summary answers (for the requester)

1. **Physical model & SDI**: Coulomb stick/slip (`τ_crit=μp`) is the same
   return-mapping structure as this project's own J2 plasticity — trial
   elastic force, yield-circle check, radial return (§1-2). The transition
   is force-continuous/tangent-discontinuous (milder than normal-contact
   activation); SDI should be extended via a new, additive
   `get_friction_state()`/`_contact_full_state()` union (§6) rather than
   overloading `get_active_set()`'s existing contract. A dedicated
   stick/slip hysteresis gate is deferred until measurement shows one is
   needed (§6, §9.2).
2. **Penalty stiffness**: `k_t = tangential_stiffness_ratio * k_contact`
   (default ratio 1.0), reusing the *already*-auto-derived `k_contact`/
   `k_ref` machinery rather than inventing a second stiffness basis;
   cross-checked (diagnostic warning only) against Abaqus's own
   `γ_crit = 0.005 * L_char` allowable-elastic-slip formula (§3).
3. **Frame consistency**: only the deformable-master class needs it;
   solved by storing the friction anchor as its own convected
   `(xi_anchor, eta_anchor)` material coordinate on the master face,
   re-evaluated against the current deformed geometry every call — the
   same device already used for the slave's own projection point, applied
   once more (§4). No rate/objective-rate machinery needed.
4. **Exact formulas**: given in this codebase's own variable names for
   both the rigid-plane 3×3 block (§7.1) and the deformable 5-node
   stencil (§7.2), STICK and SLIP branches, with the same sign convention
   (`f_int = -F_physical`) the normal force already established.
5. **The key wiring decision**: friction's own state (the anchor) must
   commit only at the three existing `update_state=True` sites in
   `solve_step()` — a converged-increment cadence — not at
   `update_augmented_multipliers()`'s three per-iteration sites, because
   the anchor is physical history, not a numerical device (§5). This is
   the single most important, non-obvious point in this design.
6. **Scope recommendation**: build for `SurfaceContactConstraint3D` and
   `DeformableSurfaceContactConstraint3D` first; explicitly defer
   `SurfaceToSurfaceContactConstraint3D` (mortar friction is a materially
   harder, separate problem, and that class's own docstring already says
   so) (§8, §9).
7. **File**: `dev_log/contact_friction_precise_design_20260916.md` (this
   document). No `dispsolver/` code was written or modified this session.
