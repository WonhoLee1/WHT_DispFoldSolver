# Contact: hardened penalty + soft contact — Abaqus-support-grade design (2026-09-15)

**Status: design only, per task scope — no `dispsolver/` code changed in this
session.** This is Phase 2.5 in the lineage of
`dev_log/3d_contact_implementation_design_20260913.md` (the original phased
plan) and `dev_log/hertz_contact_benchmark_20260913.md` (what actually got
built and measured against it: Phase 1 penalty contact, the SDI fix, the
line-search bug fix, and augmented Lagrangian with its sign-bug fix). Read
both before this document — everything below assumes that context and does
not re-derive it.

Every claim below is sourced either to a specific fetched Abaqus doc file
(`benchmark_element/reference_abaqus_docs/ctc_*.txt`, all fetched 2026-09-13
except the newly re-fetched material noted in §2.1) or to a specific
measured number already in this session's own `dev_log`/`tests` files. Where
a source doesn't publish a number (this happens once, §2.1), that gap is
stated explicitly rather than filled with an invented one.

---

## 0. What this document is answering

The task (from the team lead, translating the user's own request) is three
things:

1. Harden the **penalty method** itself to Abaqus "hard contact" parity —
   beyond what SDI (severe-discontinuity-iteration handling) already fixed.
2. Design **soft contact** (pressure-overclosure laws) — currently only
   `"HARD"` exists; `"SOFT_LINEAR"`/`"SOFT_EXPONENTIAL"` are named
   placeholders in `dispsolver/model/interaction.py`'s docstring with zero
   runtime implementation (`ContactPair.build_runtime_constraint` raises
   `NotImplementedError` for anything but `"HARD"`,
   `dispsolver/model/interaction.py:167-172`).
3. Answer whether a single unifying formulation could subsume hard-penalty
   robustness + soft-contact smoothness + augmented-Lagrangian exactness
   more cheaply than three separate code paths.

### What already exists (verified, not re-litigated here)

- `dispsolver/constraint3d/surface_contact3d.py`'s `SurfaceContactConstraint3D`:
  frictionless, hard, small-sliding, node-to-analytical-rigid-plane, penalty
  contact. `assemble()` computes `f_mag = k_contact * penetration` (line
  ~288) for `gap < 0`, else contributes nothing — this single line is
  exactly the "hard-contact-approximated-by-linear-penalty" law and is the
  one piece of math §2 below generalizes.
- `dispsolver/solver3d/dynamic3d.py`'s `solve_step()`: severe-discontinuity-
  iteration (SDI) handling — an iteration where `_contact_active_set()`
  changes is exempted from the normal convergence check and given a
  separate iteration budget (`max_sdi_iters`), matching Abaqus/Standard's
  own documented SDI treatment. **Verified working** (this session's own
  regression: 46/46 unrelated 3D tests unaffected).
- `SurfaceContactConstraint3D.augmented_lagrange=True` +
  `DynamicSolver3D.solve_step_augmented()`: Uzawa-style augmented Lagrangian.
  **Verified working on the single-C3D8-element case only**
  (`tests/test_augmented_lagrangian_contact.py`: `max_penetration` contracts
  geometrically, ratio ≈0.771/cycle, nodal force-balance residual ≈1e-10)
  after the 2026-09-13 sign-bug fix. **Not yet re-verified on the harder
  multi-node-activation Hertz curved-block case** — that benchmark's Newton
  solve was diverging even with plain penalty before augmented Lagrangian
  entered the picture (`dev_log/hertz_contact_benchmark_20260913.md`), so
  augmented Lagrangian has never actually been exercised on the case it was
  meant to fix. This gap is carried forward into §4's phased plan.

---

## Part A — Hardening the penalty method to Abaqus parity

### A.1 Automatic default penalty stiffness (replace the magic `1e8`)

**Source**: `ctc_contactconstraints_std.txt` — "Abaqus/Standard will, by
default, set the penalty stiffness to 10 times a representative underlying
element stiffness." The augmented-Lagrange default is 1000× the same
quantity. Table 1 in the same file: Lagrange multipliers are only forced in
*if* `k_penalty > 1000·k_e` — i.e. Abaqus treats 1000× the element stiffness
as the ceiling past which plain penalty/augmented-Lagrange become
ill-conditioned enough that it silently switches methods.

**Current state**: `ContactPair.build_runtime_constraint`
(`dispsolver/model/interaction.py:181-192`) falls back to a hardcoded
`1.0e8` when no `penalty_stiffness` is passed, with its own comment
admitting "we don't have a clean single representative element stiffness
scalar exposed by `DynamicSolver3D` today." Confirmed by grep: no such
estimator exists anywhere in `dispsolver/solver3d/` or `dispsolver/material3d/`.
`tests/test_augmented_lagrangian_contact.py` already works around this by
hand at the call site (`k_rep = E * L; penalty_stiffness=0.1*k_rep`,
line 87-88) — i.e. a test author already reconstructed exactly the quantity
this section formalizes.

**Design**: a `representative_element_stiffness(mesh, materials, slave_node_ids)`
helper (new function, `dispsolver/constraint3d/` or a small shared module —
not inside `SurfaceContactConstraint3D` itself, since `ContactPair` is the
resolver that already has `mesh`/`model` in scope,
`dispsolver/model/interaction.py:132-149`):

1. For each slave node, find its parent element(s) in `mesh.elements`
   (already buildable from the existing `pid`→element adjacency any mesh
   iteration already walks).
2. Per parent element: `E_elem` from `materials[pid]` — every material
   object already exposes `.E`/`.mu`/`.K` in a consistent form (confirmed:
   `dispsolver/material3d/plastic3d_jax.py:17-25` computes `mu`, `K`, `lam`
   from `E`, `nu` uniformly; hyperelastic materials expose `K`/`mu`
   equivalently — use `E_eff = 3*K*(1-2*nu)`-style back-solve, or more
   simply `E_eff = 9*K*mu/(3*K+mu)` from the standard isotropic relation, so
   the estimator works off `K`/`mu` alone without needing every material
   type to also carry a literal `.E`).
   `L_elem` = element characteristic length, e.g. `cbrt(element_volume)`
   (cheap, already have the element's node coordinates at construction
   time; no need for a proper min-edge-length computation for a first cut).
3. `k_e = E_eff * L_elem` (dimensionally force/length — the same `E*L`
   construction `test_augmented_lagrangian_contact.py` already uses by
   hand; a full continuum stiffness estimate would be `E*A/L`, but node-to-
   surface contact's "representative stiffness" is conventionally taken
   per-node against a facet, not per-element against a bar, so `E*L`
   dimensionally matches while avoiding inventing a facet-area convention
   this codebase doesn't have yet — flagged as the simplest defensible
   choice, not a literal transcription of an Abaqus-internal formula Abaqus
   itself does not publish either).
4. Take the **minimum** `k_e` over all slave-node parent elements (Abaqus's
   own guidance is conservative here — a stiffness default should not be
   dominated by one unusually stiff element in a mixed-material contact
   zone).
5. `k_default = 10 * k_e` (linear penalty), exposed as the new default when
   `ContactPair.build_runtime_constraint(penalty_stiffness=None)` — replacing
   the flat `1.0e8` fallback, which becomes a documented last-resort only
   for constraints built with no `mesh`/`materials` context at all (unit
   tests constructing `SurfaceContactConstraint3D` directly, matching how it
   is used today).
6. Add a `stiffness_scale_factor: float = 1.0` field on `ContactProperty`
   (mirrors Abaqus's `CONTACT CONTROLS, STIFFNESS SCALE FACTOR=factor`,
   `ctc_contactcontrols_std.txt`) multiplying whatever `k_default` (auto or
   user-specified) is computed — a cheap, direct-mapped knob for exactly
   the "increase by an order of magnitude" tuning pattern Abaqus's own docs
   recommend, without inventing a new mechanism.

This is Phase-1-compatible: it only changes what `penalty_stiffness=None`
resolves to, not the constraint's math. No `SurfaceContactConstraint3D`
change needed for this part alone.

### A.2 Nonlinear (4-region) penalty — the concrete, sourced fix for the Hertz-divergence finding (priority #1)

**Source**: `ctc_contactconstraints_std.txt`, "Nonlinear Penalty Method."
Four regions in penetration `h` (h>0 = penetrating), with `c0` = clearance
at zero pressure (default 0):

| region | range | behavior | default |
|---|---|---|---|
| inactive | `h < -c0` | `p = 0` | `c0 = 0` |
| initial stiffness | `-c0 ≤ h ≤ e` | `p` linear, slope `K_i` | `K_i` = 1× rep. element stiffness; `e` = 1% of characteristic facet length `L_char` |
| stiffening | `e ≤ h ≤ d` | `p` quadratic, stiffness ramps `K_i→K_f` linearly | `d` = 3% of `L_char` |
| final stiffness | `h > d` | `p` linear, slope `K_f` | `K_f` = 100× rep. element stiffness |

The doc states the qualitative shape (linear→quadratic→linear, matching
stiffness continuity at `e` and `d`) but not the literal quadratic
coefficient — that's fully determined by requiring `p` and `p'` both
continuous at `e` and `d` (a real derivation, not an invented number):

```
p(h) =
  0                                              h < -c0
  K_i*(h + c0)                                   -c0 ≤ h ≤ e
  K_i*h + (K_f-K_i)/(2*(d-e)) * (h-e)^2           e ≤ h ≤ d      [c0=0 case shown; shift h→h+c0 generally]
  K_i*d + (K_f-K_i)*(d-e)/2 + K_f*(h-d)           h > d
```

with `K(h) = dp/dh` continuous everywhere (`K_i` at `h=e`, ramping linearly
to `K_f` at `h=d`, then flat) — confirmed by construction: `p` is `C¹`
(value and slope continuous) at both breakpoints; only `p''` (i.e. `dK/dh`)
has jumps at `e` and `d`, which is the mildest possible discontinuity a
piecewise law can have there.

**Why this is the priority-#1 robustness fix, precisely**: the doc's own
stated rationale — "The low initial penalty stiffness typically results in
better convergence of the Newton iterations and better robustness, while
the higher final stiffness keeps the overclosure at an acceptable level as
the contact pressure builds up" — targets *exactly* the failure this
codebase already measured. `dev_log/hertz_contact_benchmark_20260913.md`
found Newton reliably diverging **at the instant a new slave node crosses
into penetration**, independent of load-step size (bisected 1/128→1/8192,
same failure point every time) — i.e. the size of the residual jump at
activation, not the step size, is the problem. Today's linear penalty jumps
stiffness from `0` to `k_contact` (10×–1000× element stiffness) the instant
a node activates. Nonlinear penalty instead jumps from `0` to `K_i` (1×
element stiffness — an order of magnitude smaller than even the *linear*
law's already-conservative 10× default) at that same instant, only ramping
up to the stiff `K_f=100×` regime once penetration has grown past `e`
(1% of `L_char`) — i.e. by construction it makes the newly-activated
constraint's very first Newton iteration see a stiffness comparable to the
surrounding structure's own stiffness, not 10-1000× stiffer than it.

**Important, precise correction to the framing in the task brief**: this
does **not** make the activation event itself smooth (see §2.3 below for
why — the `0→K_i` jump at `h=-c0` is still a genuine stiffness
discontinuity, same *order* of non-smoothness as today's hard-linear law).
What it changes is the **magnitude** of that discontinuity, which is
exactly the lever the doc's own rationale describes and exactly what SDI
needs to be cheap to exempt through (SDI already tolerates the
discontinuity structurally; nonlinear penalty makes each individual
SDI-exempted iteration's residual jump smaller, so SDI should need fewer
exempted iterations per activation event, not zero). **Recommendation: this
is the first thing to actually build and re-run against
`benchmark_element/benchmark_3d_contact.py`'s Hertz curved-block case** —
before augmented Lagrangian is trusted further, and before any new Newton-
formulation machinery, because it is the cheapest change (a piecewise
function substituted for one line) most directly targeted at the one
concrete, measured failure this codebase has.

**Implementation note (design only)**: this replaces the one-line
`f_mag = self.k_contact * penetration` in
`SurfaceContactConstraint3D.assemble()` with a call to a shared
pressure-overclosure law object — see §2.4, which is where nonlinear
penalty and the soft laws below turn out to share one seam.

### A.3 Contact stabilization/damping — replace the pseudo-velocity hack with the real Abaqus mechanism, for the mechanism's real documented purpose

**Source**: `ctc_contactdamping.txt` (the physical damping-force law) +
`ctc_contactcontrols_std.txt` ("Automatic Stabilization of Rigid Body
Motions in Contact Problems").

Two important facts, both explicit in the source, that correct the current
code's own framing:

1. `ctc_contactdamping.txt`'s formula is
   `f_vd = μ0 * A * v_rel`, `μ0` in units of pressure/velocity, `A` = nodal
   area, `v_rel` = relative normal (or tangential) velocity between the
   surfaces — a genuine physical viscous term, "not applicable for linear
   perturbation procedures" and explicitly "should generally be used only
   when it is otherwise impossible to obtain a solution."
2. `ctc_contactcontrols_std.txt`'s automatic stabilization is explicitly
   scoped: "It is not meant to simulate general rigid body dynamics; **nor
   is it meant for contact chattering situations** or to resolve initially
   tight clearances between mating surfaces." Its actual purpose is damping
   *rigid-body drift of an as-yet-unconstrained body* before contact
   closure establishes equilibrium (the doc's own example: two rigid
   surfaces flanking a column that hasn't touched either one yet). It is
   ramped down linearly over the step (default), active only while surface
   separation is below "a characteristic surface dimension," and the
   damping coefficient is computed from "the stiffness of the underlying
   elements and the step time" (Abaqus does not publish the exact formula
   for this either — same gap as the exponential pressure-overclosure
   formula in §2.1; not invented here).

**This means the current `stabilization_coefficient`/`c_stab` mechanism in
`SurfaceContactConstraint3D` (`dispsolver/constraint3d/surface_contact3d.py:100-133,296-316`)
is targeting the WRONG failure mode by Abaqus's own documented scoping.**
Its own docstring already says it was aimed at "the residual discontinuity
at first activation" — i.e. exactly the chattering/activation-discontinuity
class Abaqus's docs say automatic stabilization is *not* for. That
explains, retroactively, why it was "never proven to help" (per its own
docstring) and why SDI (a mechanism actually documented for this exact
purpose) is what fixed the measured problem instead. It also uses the
wrong time basis: it measures Δgap **per Newton iteration**, not per real
(pseudo-)time step, so it has no principled physical units at all (velocity
needs a `Δt`, and Newton iterations aren't time).

**Recommendation**:
- **Do not extend `c_stab`** — it is solving a problem SDI already solves,
  with the wrong physical basis. Demote it in documentation to "known
  superseded by SDI, kept only for A/B comparison" rather than continuing
  to develop it.
- **Do** add a *real* contact stabilization mechanism, for its *actual*
  documented purpose (damping rigid-body drift of a not-yet-contacting
  body), reusing this codebase's own existing, correctly-scoped pattern:
  `dispsolver/solver/stabilization.py`'s `AbaqusViscousStabilization`
  (`F_stab = c*M_diag*v`) is architecturally the right template — it
  already uses a genuine per-load-step pseudo-time basis (`v` there is a
  real Δu/Δt over the step, not over a Newton iteration). A per-contact-pair
  version would compute `v_rel = (gap_current - gap_at_step_start) /
  Δt_step` (the *step's* time increment, from `solve_step`'s own `dt`
  argument, not "since the last `assemble()` call"), scaled by a coefficient
  derived from the same `representative_element_stiffness` estimator in
  §1 (dimensionally: `μ0 ~ k_default * Δt_step / A_facet`, giving
  pressure/velocity — a defensible, documented-consistent default the same
  way §1's `k_default` is, not a literal transcription of Abaqus's own
  unpublished internal formula), ramped down linearly over the step per
  the doc's default, and active only while separation is below a
  characteristic length (reuse `L_char`/`L_elem` from §1/§2).
- Gate this behind `normal_behavior="HARD"` only for now — matching
  Abaqus's own default note that automatic stabilization is
  "zero for contact modeled with contact elements" in some configurations,
  and matching this design's own §2.3 finding that soft contact doesn't
  change the *need* for activation-robustness machinery, only its
  magnitude, so there's no reason to couple the two.
- This is a genuinely new, small mechanism (Phase 2.5, not Phase 1-
  compatible without a small `DynamicSolver3D` hook to pass `Δt_step` and
  the step-start gap into the constraint) — not a priority-#1 item like
  §2.2's nonlinear penalty; recommend building it only if a future example
  actually exhibits unconstrained rigid-body drift before contact
  (none of today's fold/Hertz examples do — the display/plate is already
  Dirichlet-driven, never floating).

### A.4 Overclosure/adjustment refinements beyond the current one-shot `strain_free_adjust`

**Source**: `ctc_adjustsurfaces_std.txt` (contact-pair-specific) and
`ctc_genlcontinit_std.txt` (general-contact equivalent, background only —
this codebase's contact pairs map onto the contact-*pair* semantics, not
general contact, per the 2026-09-13 design doc's own §1 scope decision).

Current code (`SurfaceContactConstraint3D.__init__`,
`dispsolver/constraint3d/surface_contact3d.py:176-188`): if
`strain_free_adjust=True`, any node with `gap0 < 0.0` (i.e. any amount of
initial penetration, no matter how large) gets its offset zeroed
unconditionally. Cross-checked against the source: this is **exactly**
Abaqus's `ADJUST = 0.0` case ("Specifying a value of `0.0` for `a` causes
Abaqus/Standard to adjust **only** those secondary nodes that are
penetrating the main surface") — i.e. the current code is already
correctly matching one specific, named Abaqus mode, not an ad hoc
approximation. **No change needed to this behavior itself.**

What's missing, as real (not speculative) extensions:
1. **A nonzero adjustment-zone distance `a`** (`ctc_adjustsurfaces_std.txt`,
   `CONTACT PAIR, ADJUST=a`): nodes within distance `a` of the main
   surface — on either side, i.e. small initial *gaps* too, not just
   overclosures — get snapped exactly onto it. Mechanically this is a small
   change to the existing per-node loop: replace the `gap0 < 0.0` test with
   `gap0 < a` (default `a=0.0`, preserving current behavior exactly) and
   set the offset for any node satisfying it, including small positive
   gaps. Cheap, backward-compatible, directly Abaqus-parity.
2. **Interference-fit resolution** (`ctc_adjustsurfaces_std.txt`'s default
   behavior for contact pairs is actually the *opposite* of strain-free
   adjustment — treat initial overclosure as a real interference fit,
   resolved gradually over the step via a shrink-fit ramp, generating real
   stress). This is a genuinely different, more complex feature (needs a
   per-step-fraction target penetration, not a one-time offset) — **not
   recommended now**: no current example/benchmark has an actual
   interference-fit geometry (every fold/Hertz example starts from clean,
   non-overlapping geometry by construction), so building it would be
   speculative infrastructure this project's own culture argues against.
   Flagged as available in Abaqus and known, not designed further.

### A.5 The Lagrange-multiplier 1000× threshold — a guardrail, not an actionable feature

Table 1 in `ctc_contactconstraints_std.txt`: for penalty or augmented-
Lagrange hard contact, Lagrange multipliers are only introduced if
`k_penalty > 1000 * k_e`. This codebase has no Lagrange-multiplier contact
path (deliberately deferred, per the 2026-09-13 design doc's own §4) and
isn't gaining one here. Its value here is as a **numeric ceiling on how
hard a caller should be allowed to push `penalty_stiffness` or
`stiffness_scale_factor`** (§1) without silently entering a regime Abaqus
itself considers to need a different enforcement method. Recommendation:
have `ContactPair.build_runtime_constraint` emit a `warnings.warn` (not
raise — this project's own "no permanent exceptions" bias suggests a
warning, since a user might have a legitimate reason) when the resolved
`penalty_stiffness > 1000 * k_e`, citing this exact threshold. Cheap,
directly sourced, no new mechanism.

### A.6 Second-order-face / supplementary-constraint concerns — confirmed not yet applicable

`ctc_contactcontrols_std.txt`'s discussion of ambiguous nodal forces on
second-order element faces (ambiguous force distribution on `C3D20`/`C3D10`-
family secondary surfaces) does not apply today: every contact example in
this codebase builds `slave_node_ids` from first-order (`C3D8`-family) hex
meshes. Flagged for revisit only if/when a contact surface is ever built
from a second-order element family — no action needed now.

---

## Part B — Soft contact (pressure-overclosure laws)

### B.1 The full catalogue, per `ctc_normalinteraction.txt`

Six pressure-overclosure relationships exist in Abaqus; three matter here:

| law | applicable here? | why |
|---|---|---|
| **Hard** | yes (exists) | current default |
| **Linear** | yes — design below | simplest soft law |
| **Exponential** | yes — design below | the "smoother than hard" case |
| **Tabular (piecewise-linear)** | yes — design below | user-supplied curve |
| Tabular via scale-factor | **no** | "available only for general contact in Abaqus/Explicit" — this project has neither Explicit nor general contact |
| No-separation | **no** (for now) | changes physics (allows tensile transmission, permanently bonds on first touch) — a different modeling intent than frictionless separable contact; trivial to add later as a fourth law once the others exist, not designed further here |

### B.2 Exact mathematical forms

Let `h` = overclosure = penetration = `-gap` (matching this codebase's
existing sign convention, `SurfaceContactConstraint3D.assemble()`'s
`penetration = -gap`). All three laws below are expressed as `(p(h), dp/dh)`
pairs — this uniform shape is deliberate, see §2.4.

**Linear** (fully specified in `ctc_normalinteraction.txt`'s plain text,
no figure needed):

```
p(h) = k * (h - c0)   for h > c0
p(h) = 0              otherwise
dp/dh = k             for h > c0, else 0
```

`k` is a **user-specified** slope (Abaqus: "you specify the slope of the
pressure-overclosure relationship, k") — unlike hard contact's `k_contact`,
this is not meant to be auto-derived from element stiffness; it represents
a real physical interposed-layer stiffness the user is modeling
deliberately. `c0` (default 0) is the "clearance at which contact pressure
is zero" shift, same role as §2.2's nonlinear-penalty `c0`.

**Important finding**: this is mathematically **identical in form** to the
current hard-contact code (`f_mag = k_contact * penetration`,
`SurfaceContactConstraint3D.assemble()` line ~288) — same one-sided ramp,
same activation kink at `h=c0`. The doc itself says as much: "The linear
pressure-overclosure relationship is identical to a tabular relationship
with two data points, where the first point is located at the origin."
**Implementing `normal_behavior="LINEAR"` needs no new math at all** — it
is exactly today's penalty law, generalized only by (a) making the slope a
directly user-specified physical quantity rather than a numerically-derived
"hard contact" approximation, and (b) exposing `c0`. The distinction
Abaqus draws between "hard" and "linear soft" is conceptual/default-method
(§B.3), not mathematical.

**Exponential** — **source gap, stated honestly**: `ctc_normalinteraction.txt`
gives the two user parameters (`c0` = clearance at which pressure starts
transmitting, `p0` = pressure at zero clearance) and a qualitative
description ("increases exponentially as the clearance continues to
diminish"), but the *page itself* has no printed closed-form equation —
re-fetching the raw (unstripped) HTML confirmed the formula appears only as
an `<img>` schematic figure (`exp_pressure_overclosure_std.png`, saved to
`reference_abaqus_docs/` this session), not as text or MathML. This is a
genuine limit of the source material, not an extraction artifact (the
`README.md`'s own caveat about lost LaTeX/MathML doesn't apply here — there
was no equation markup to lose).

Given only `(c0, p0)`, the minimal exponential curve satisfying Abaqus's
own two stated boundary conditions — `p(c0)=0`, `p(0)=p0`, monotonically
increasing as clearance/`c` decreases, smoothly continuing into
penetration (`c<0`) — is a one-parameter family (shape/rate constant `α`
free):

```
p(c) = p0 * (exp(α*(c0 - c)) - 1) / (exp(α*c0) - 1)     for c < c0
p(c) = 0                                                  for c ≥ c0
```

(`c` = clearance = `-h`; check: `p(c0) = 0` for any `α>0`; `p(0) =
p0*(exp(α c0)-1)/(exp(α c0)-1) = p0` for any `α>0` — both stated boundary
conditions are satisfied identically in `α`, confirming `α` really is an
extra degree of freedom Abaqus's user-facing inputs don't fix.) Abaqus
itself does not expose `α` as a user parameter on this page — it must be
fixed internally by Abaqus in a way this fetched material doesn't reveal.
**§B.7.3 below gives the recommended, fully closed-form way to fix both
`α` and `p0` together** (no root-finding needed — an earlier draft of this
section proposed fixing `α` alone via a numerical bisection against
`k_default`; superseded, since B.7.3's two-condition derivation pins both
constants in one closed form and additionally answers the "how should a
user actually choose `p0`/`c0` without typing an absolute number" question
the auto-derivation requirement raised). **Flag explicitly**: whichever
constant-fixing rule is used is this codebase's own reconstruction, not a
transcription of an Abaqus-internal formula — the `(c0, p0)` boundary
conditions themselves (§B.2, `p(c0)=0`, `p(0)=p0`) are directly sourced and
certain; the specific rule for choosing `α`/`p0` beyond those two
conditions is not. Cross-check against the Abaqus Theory Manual (not on
this machine's local doc mirror under any RefMap path tried this session)
before treating that rule as final; it does not block building the LINEAR
or TABULAR laws, which have no such gap.

```
dp/dc = -p0 * α * exp(α*(c0-c)) / (exp(α*c0) - 1)     for c < c0, else 0
dp/dh = -dp/dc  (since h = -c)
```

**Tabular** (fully specified in text, no figure needed): user supplies
increasing pairs `(h_i, p_i)`, `i=1..n`. "Surfaces transmit contact
pressure when the overclosure ... is greater than `h_1`" (i.e. `h_1` plays
`c0`'s role — may be negative, allowing an initial-gap tolerance, or zero).
For `h_1 ≤ h ≤ h_n`: linear interpolation between bracketing points. For
`h > h_n`: "extrapolated based on the last slope computed from the
user-specified data." For `h < h_1`: `p=0`.

```python
def tabular_p(h, h_pts, p_pts):
    if h <= h_pts[0]:
        return 0.0, 0.0
    if h >= h_pts[-1]:
        slope = (p_pts[-1] - p_pts[-2]) / (h_pts[-1] - h_pts[-2])
        return p_pts[-1] + slope * (h - h_pts[-1]), slope
    # else: locate bracketing segment, linear interpolate + return its slope
```

(`np.interp` handles the middle case directly; the two boundary branches
need to be handled explicitly since `np.interp` clamps rather than
extrapolates.)

### B.3 Composability with SDI — a more precise answer than "smoother, so maybe unnecessary"

The task brief asks whether soft contact's smoothness might make SDI
"easier or unnecessary." **Precise answer, derived from the actual
functional forms above, not assumed**: **no law here removes the
activation discontinuity; every one of them still has a `0 → finite`
stiffness jump exactly at the activation threshold** (`h=c0` for linear/
nonlinear-penalty, `c=c0` for exponential, `h=h_1` for tabular) — because
that jump is a structural consequence of the **complementarity condition
itself** (`gap ≥ c0` XOR `pressure > 0`), not of which curve is used once
the constraint is active. Linear soft contact's law is bit-for-bit the same
shape as hard contact's at that boundary (§2.2 above). Exponential's
derivative at `c=c0⁻` is `-p0*α/(exp(α c0)-1)` — generically nonzero and
finite, jumping up from the zero stiffness of the inactive branch. Tabular
is explicitly piecewise-*linear*, so unless a user deliberately supplies a
first segment with zero slope, it has the same jump.

**What DOES change is the *magnitude* of the jump**, exactly as found for
nonlinear penalty in §2.2: a well-chosen soft law (small `p0`/`k` near
`c0`, or nonlinear penalty's `K_i`) makes the residual step at activation
smaller in absolute terms than jumping straight to a stiff `k_contact`.
This makes each SDI-exempted iteration's job easier (smaller
discontinuity to jump across) but does **not** make SDI structurally
unnecessary — the active-set switch itself is still there. **This
generalizes §2.2's finding**: soft contact and nonlinear penalty are, from
the solver's perspective, the *same kind* of mitigation (shrink the
activation-event residual jump), not different mechanisms. `SDI` remains
required by every law in this catalogue.

The one law that *would* remove the switch entirely is "no separation"
(§B.1's excluded sixth option) — but that changes physics (permanent bond,
tensile-capable), which is a different modeling intent, not a numerical
convenience; out of scope here.

### B.4 Composability with augmented Lagrangian — Abaqus's own answer is "no," and it's the right one

**Direct source finding**: `ctc_contactconstraints_std.txt`, Augmented
Lagrange Method section: **"The augmented Lagrange method ... applies only
to hard pressure-overclosure relationships."** This is explicit, not
inferred. The Direct Method section separately confirms: **"The direct
method is the only method that can be used to enforce 'softened'
pressure-overclosure relationships."** — i.e. Abaqus does not offer
penalty *or* augmented-Lagrange enforcement for soft laws at all; only the
direct method (with Lagrange multipliers switched on above the same 1000×
threshold from §5, off below it).

This answers the task brief's specific question ("does soft contact even
need augmented Lagrangian, given it's already smoother than hard penalty
by construction?") with a sourced "no, and Abaqus's architecture agrees
for a reason that generalizes cleanly": augmented Lagrangian exists to
drive an *approximate* representation of a *deliberately exact* target
(zero penetration) toward that exact target. A soft law's compliance is
the *intended physical answer*, not a numerical artifact standing in for
an exact one — there is nothing to augment toward. Recommendation for this
codebase: **mirror this restriction exactly** — `ContactPair.build_runtime_constraint`
should raise `NotImplementedError` if `constraint_enforcement=
"AUGMENTED_LAGRANGE"` is combined with any `normal_behavior` other than
`"HARD"`, matching the existing raise-on-unsupported-combination pattern
already in that method (`dispsolver/model/interaction.py:150-172`).

### B.5 API shape

Mirroring `dispsolver/model/constraint.py`'s `Tie`/`RigidBody` pattern
(already the established convention this codebase's Abaqus-parity CAE
objects follow) and Abaqus's own `*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=`
keyword structure:

```python
@dataclass
class ContactProperty:
    name: str
    normal_behavior: str = "HARD"          # "HARD" | "SOFT_LINEAR" | "SOFT_EXPONENTIAL" | "SOFT_TABULAR"
    friction: Optional[float] = None

    # --- scale-based auto-derivation, the DEFAULT path (see §B.7) ---
    softness_scale: float = 1.0             # dimensionless; 1.0 = matches auto hard-contact stiffness
    allowable_penetration: Optional[float] = None   # length; None = auto (1% of L_char, §A.2's own `e` default)

    # --- explicit absolute opt-out overrides (existing pattern; take
    #     precedence over softness_scale/allowable_penetration when set) ---
    linear_stiffness: Optional[float] = None       # k, LINEAR only
    clearance_at_zero_pressure: float = 0.0         # c0, shared by LINEAR/EXPONENTIAL/nonlinear-penalty
    exponential_c0: Optional[float] = None
    exponential_p0: Optional[float] = None
    tabular_overclosure: Optional[Sequence[float]] = None   # h_i, increasing
    tabular_pressure: Optional[Sequence[float]] = None       # p_i, increasing
    # penalty hardening (Part A)
    penalty_form: str = "LINEAR"            # "LINEAR" | "NONLINEAR" (A.2's 4-region law)
    stiffness_scale_factor: float = 1.0     # A.1's CONTACT CONTROLS analog, applied after auto/override resolution
```

Each law resolves, inside `ContactPair.build_runtime_constraint`, to one
small law object exposing `p(h) -> (pressure, dp_dh)` (§2.4). `ContactProperty`
stays the single place a caller configures *which* law and its parameters,
matching how it already owns `normal_behavior`/`friction` today — no new
top-level CAE class needed. **§B.7 below specifies exactly how
`softness_scale`/`allowable_penetration` resolve into the absolute fields
above when the caller leaves them at `None`** — this is the primary,
recommended path; the absolute fields remain available for a caller who
already knows the number they want, exactly mirroring how
`ContactPair.build_runtime_constraint(penalty_stiffness=None)` already
treats `None` as "auto-derive" for hard contact (§A.1).

### B.6 The shared seam — one mechanism, not three code paths

Every law above (`HARD`'s current one-liner, `A.2`'s nonlinear penalty,
and all three §2.2 soft laws) reduces to the same interface:

```python
class PressureOverclosureLaw(Protocol):
    def evaluate(self, h: float) -> tuple[float, float]:
        """Returns (pressure p(h) >= 0, tangent dp/dh) — h = penetration = -gap."""
```

`SurfaceContactConstraint3D.assemble()`'s current
`f_mag = self.k_contact * penetration` (one line, hard-law-only) becomes
`f_mag, k_diag_law = self.law.evaluate(penetration)` — everything else in
`assemble()` (the `f_node = -f_mag * self.normal` sign convention verified
correct in the 2026-09-13 sign-bug fix, the 3×3 stiffness scatter, the
stats dict) is **unchanged**, because it was already written generically
enough (it only ever consumed a scalar `f_mag` and a scalar `k_diag`) —
confirmed by reading `assemble()` end to end (lines 228-398): nothing below
the `f_mag =` computation depends on *which* law produced it.

`get_active_set()`'s current branch —

```python
if self.augmented_lagrange:
    p = self._lam[s_nid] + self.k_contact * (-gap)
    if p > 0.0: active.add(s_nid)
elif gap < 0.0:
    active.add(s_nid)
```

— collapses to one call through the same law object once augmented
Lagrangian is restricted to `HARD` only (§B.4): `p, _ = self.law.evaluate(-gap
+ (self._lam[s_nid]/k_law if self.augmented_lagrange else 0.0))`-style
unification is possible but not required by the restriction; the simpler,
equally correct change is just: `p, _ = self.law.evaluate(penetration);
if augmented_lagrange: p += self._lam[s_nid]`, `if p > 0: active.add(...)`
— i.e. the *hard-contact-specific* augmentation offset stays exactly as
specific as Abaqus itself restricts it to be, while the underlying
pressure evaluation still goes through one shared law seam. Either way,
this is the *complete* answer to the task's Part 3 "unifying formulation"
question for the piece Abaqus's own docs say should be unified (the
pressure-overclosure evaluation), while correctly keeping the piece Abaqus
says should NOT be unified (augmented Lagrangian only applying to hard
contact) as an explicit restriction rather than a false generalization.

### B.7 Scale-based auto-derivation — soft contact draws from the SAME auto-stiffness hard contact uses, controlled by two intuitive knobs

**User requirement (added after initial draft, folded in here rather than
reworking §B.2/§B.5 above from scratch)**: soft contact should not require
typing an absolute stiffness/pressure number the user has no principled
basis for choosing. It should auto-derive from the **same** material/
geometry basis §A.1 already gives hard contact (`k_ref`), controlled by two
small, physically-intuitive knobs: a dimensionless `softness_scale` (1.0 =
"behaves like default hard contact", <1.0 = softer, >1.0 = stiffer) and a
length, `allowable_penetration` ("how far can it penetrate before the
reaction approaches hard-contact-like stiffness"). This subsection makes
that concept fully specified and buildable, not just a naming convention.

**B.7.1 — the one shared reference stiffness, `k_ref`**

§A.1 already designs `representative_element_stiffness(mesh, materials,
slave_node_ids)` → `k_e` (minimum, over the slave nodes' parent elements,
of `E_eff * L_elem`). Call this quantity `k_ref` here — it is the single
auto-derived basis **both** hard and soft contact draw from:

- Hard / linear-penalty default (§A.1, unchanged): `k_default_hard = 10 *
  k_ref`, matching Abaqus's own "10× representative underlying element
  stiffness" default.
- Nonlinear penalty (§A.2, unchanged): `K_i = k_ref` (1×), `K_f = 100 *
  k_ref`, matching Abaqus's own nonlinear-penalty defaults.
- **New here**: every soft law's auto-derived parameters (below) are
  expressed as closed-form functions of `(k_ref, softness_scale,
  allowable_penetration)` — no separate stiffness basis is invented for
  soft contact; it is the same `k_ref`, only reached with different
  multipliers, exactly the way hard-contact's own 10×/1000× multipliers
  already work off it. This directly answers the "where should this live"
  question: `k_ref` is computed once per contact pair inside
  `ContactPair.build_runtime_constraint` (it already has `mesh`+`materials`
  in scope, §A.1) and handed to whichever law-parameter resolver runs next
  — one computation, shared by every downstream law, not recomputed per law.

**B.7.2 — LINEAR: `softness_scale` alone is sufficient**

LINEAR's shape has no natural transition length distinct from a slack
tolerance (§B.2: constant slope everywhere past `c0`), so it needs only
one knob:

```
k = softness_scale * k_default_hard = softness_scale * 10 * k_ref
c0 = allowable_penetration if explicitly set, else 0.0   # optional slack tolerance, not required
```

`softness_scale=1.0` reproduces **exactly** today's hard/linear-penalty
slope (satisfies the user's own stated requirement precisely, not
approximately) since `k = 1.0 * 10 * k_ref = k_default_hard`.
`allowable_penetration` is accepted but secondary for this law — it maps to
`c0` (a pre-contact slack zone), not to a stiffening ramp, because LINEAR
has no ramp to control.

**B.7.3 — EXPONENTIAL: both knobs are needed, and here is the exact closed form**

Unlike LINEAR, EXPONENTIAL has a genuine second shape/length degree of
freedom (§B.2's `α`) beyond a single stiffness — this is exactly the knob
`allowable_penetration` should control, and it is possible to pin both
`(p0, α)` in closed form (no bisection, fully specified) from
`(softness_scale, allowable_penetration, k_ref)` by choosing **two**
physically meaningful conditions instead of one:

1. **Touching stiffness matches `softness_scale`'s own definition
   exactly** (the requirement `softness_scale=1.0 ⇒ behaves like default
   hard contact` must hold precisely, not approximately, so this condition
   is not optional): the local contact stiffness at the moment of
   geometric touching (`c=0`, i.e. `-dp/dc` there) must equal
   `K_target := softness_scale * k_default_hard = softness_scale * 10 *
   k_ref`.
2. **`allowable_penetration` (`δ`) sets the length scale over which
   stiffness ramps up by the same 10× factor Abaqus's own nonlinear-penalty
   `K_i→K_f` ratio uses** (§A.2 and B.7.1 both already use exactly a 10×/
   100× = 10× step) — i.e. by the time penetration has reached `δ` past
   touching, the local stiffness should have grown to `10 * K_target`
   (mirroring nonlinear penalty's own `K_f = 100*k_ref = 10*K_i` ratio,
   scaled by `softness_scale` the same way `K_i` itself is): `κ(-δ) = 10 *
   K_target`, where `κ(c) := -dp/dc`.

Setting `c0 = δ` (reusing the single length knob for both Abaqus's own
`c0` and this ramp-length role — deliberately, to keep this a genuinely
single-length-scale knob, not two hidden ones) and solving both conditions
against the exponential form from §B.2 (`κ(c) = p0*α*exp(α(c0-c))/(exp(α
c0)-1)`, so `κ(-δ) = κ(0)*exp(αδ)` directly, since `c0=δ`):

```
κ(0) = K_target                              [condition 1]
κ(0) * exp(α*δ) = 10 * K_target              [condition 2]
  ⇒ exp(α*δ) = 10  ⇒  α = ln(10) / δ         (pure geometry, independent of K_target/softness_scale)

κ(0) = p0*α*exp(α*δ)/(exp(α*δ)-1) = p0*α*10/9 = K_target
  ⇒ p0 = (9/10) * K_target / α = (9/10) * K_target * δ / ln(10)
```

Fully closed-form, no root-finding needed:

```
δ  = allowable_penetration           (or 1% of L_char if left at None — same default §A.2's own `e` uses)
s  = softness_scale
K_target = s * 10 * k_ref
α  = ln(10) / δ
c0 = δ
p0 = 0.9 * K_target * δ / ln(10)     = 0.9 * s * 10 * k_ref * δ / ln(10) ≈ 3.907 * s * k_ref * δ
```

giving the concrete Abaqus-notation law parameters requested:

```
exponential_c0 = δ = allowable_penetration
exponential_p0 = 0.9 * softness_scale * 10 * k_ref * allowable_penetration / ln(10)
```

**Sanity checks** (all verified algebraically, not just asserted): at
`softness_scale=1.0`, touching stiffness `κ(0) = 10*k_ref = k_default_hard`
— exactly matches hard/linear-penalty's own default, satisfying the user's
literal requirement. The curve's shape (`α`, hence `c0`) doesn't depend on
`softness_scale` at all — only its overall vertical scale (`p0`, and hence
every pressure value) does, since `p0 ∝ s` while `α` and `c0` are pure
functions of `δ` — i.e. `softness_scale` is a genuinely proportional
softer/stiffer dial at every point on the curve, matching the user's own
"proportionally softer/stiffer" framing exactly. A smaller
`allowable_penetration` makes `α` larger (steeper), reaching the "10×
stiffer" milestone over a shorter penetration distance — i.e. behaves
closer to hard contact sooner, matching the "allow less penetration ⇒
closer to hard" intuition directly.

**B.7.4 — TABULAR: reuse the exponential curve as an auto-generated starting table**

Tabular contact's entire point is a user-supplied curve, so it has no
independent "auto-derive from two knobs" story of its own — but when a
caller wants to *start* from the auto-scaled behavior and then hand-edit
specific points, the design is: sample the §B.7.3 exponential curve (using
the same `softness_scale`/`allowable_penetration` resolution) at a small
fixed number of points (e.g. 6, geometrically spaced from `c0` down to
`-3*δ`) to auto-populate `tabular_overclosure`/`tabular_pressure`, which
the caller can then edit. This reuses B.7.3's formula rather than inventing
a third independent one — deliberately, to keep exactly one soft-contact
math derivation in this document, not three.

**B.7.5 — resolution order in `ContactPair.build_runtime_constraint`**

1. Always compute `k_ref` (§B.7.1) once per contact pair — needed
   regardless of which law or knobs are used.
2. If the law's own **absolute** field(s) are explicitly set
   (`linear_stiffness` for LINEAR; `exponential_c0` **and**
   `exponential_p0` together for EXPONENTIAL; `tabular_overclosure`/
   `tabular_pressure` for TABULAR) — use them verbatim; `softness_scale`/
   `allowable_penetration` are ignored for that law (warn if both an
   absolute field and a non-default `softness_scale`/`allowable_penetration`
   are given, rather than silently picking one — matching this project's
   existing "explicit warning over silent precedence" preference,
   e.g. §A.5's 1000× warning).
3. Otherwise, auto-derive per §B.7.2/§B.7.3/§B.7.4 from
   `(softness_scale, allowable_penetration, k_ref)`, defaulting
   `allowable_penetration` to 1% of `L_char` (the same characteristic
   length §A.2's nonlinear-penalty `e` parameter already defaults to — one
   consistent default length scale across this whole design, not a second
   invented one) when left at `None`.
4. Apply `stiffness_scale_factor` (§A.1) as a final multiplier on whatever
   was resolved in step 2 or 3 — consistent with Abaqus's own step-level
   `CONTACT CONTROLS, STIFFNESS SCALE FACTOR` being a further multiplier on
   top of whatever base stiffness was already established, not a
   replacement for it.

**API usage this enables, exactly as requested**:

```python
# Auto-scaled soft contact -- no absolute number typed anywhere:
ContactProperty(name="SOFT_LAYER", normal_behavior="SOFT_EXPONENTIAL",
                softness_scale=0.3, allowable_penetration=0.05)

# A caller who already knows the absolute numbers can still opt out:
ContactProperty(name="EXACT_LAYER", normal_behavior="SOFT_EXPONENTIAL",
                exponential_c0=0.02, exponential_p0=5.0)
```

---

## Part C — Is there a single unifying formulation, and should this codebase adopt one?

**Direct answer: no clean single Abaqus-native "dial" exists that
subsumes hard/soft/augmented-Lagrangian — Abaqus's own architecture treats
them as a branch by modeling *intent*, not a continuum — and this
codebase should not manufacture one either. The right unifying move is
narrower and cheaper than a new Newton formulation: unify the
pressure-overclosure *evaluation* (§B.6, already a small, local change),
and keep SDI + a scoped-down augmented-Lagrangian exactly where they
already are, because both are already correctly targeted and independently
verified (SDI fully, augmented-Lagrangian on one case).**

Working through the specific ideas the task brief raised:

1. **"Does a well-chosen EXPONENTIAL law with the right augmentation
   parameter subsume hard contact as a limiting case?"** Mathematically,
   yes in the limit (`α→∞` or `p0`/effective-stiffness-at-`c0`→∞ makes the
   exponential curve approach a vertical wall at `c=0`, i.e. hard
   contact's idealized shape). But this is **not** how Abaqus frames it,
   and for a stated, sourced reason (§B.4): Abaqus explicitly does not
   allow augmented Lagrangian on soft laws, and separately warns that
   pushing a soft law's slope past 1000× the element stiffness makes
   Abaqus silently switch to Lagrange-multiplier enforcement anyway (Table
   1, same threshold as §A.5) — i.e. Abaqus's own system already detects
   "you've made this soft law hard enough that it needs exact treatment"
   and reacts by changing *method*, not by exposing one smooth dial the
   user tunes. Manufacturing a single dial here would mean re-deriving and
   maintaining that same silent-switch logic ourselves for no accuracy
   gain over just implementing `HARD` and `EXPONENTIAL` as the two
   separate, correctly-scoped laws the source already describes.
2. **"Is there a semi-smooth-Newton (Alart-Curnier) or Nitsche formulation
   that naturally generalizes hard/soft in one code path?"** Yes, these
   are real, well-established techniques (the prior research pass already
   flagged Alart-Curnier as the #2 follow-on after SDI, per the teammate's
   own summary) that reformulate the normal complementarity condition as a
   single nonsmooth-but-well-posed equation solved by a semi-smooth Newton
   step, and Nitsche-based contact avoids Lagrange multipliers by
   symmetrizing a consistency term directly in the weak form — either
   would, in principle, handle hard and soft laws (and even the
   augmented-Lagrangian limit) through one residual/tangent pair. **Not
   recommended now**, for a reason specific to this codebase's actual
   state, not a general dismissal: (a) SDI — the cheaper, already-adopted
   fix — has not yet been given its natural companion (§A.2's nonlinear
   penalty) a chance to work on the one concrete failure this codebase has
   measured (Hertz curved-block divergence); (b) augmented Lagrangian has
   only been verified on a trivial single-element case, not the harder
   case it exists for; adopting a wholesale new formulation before either
   of those cheaper, already-partially-built mechanisms has been fully
   exercised would be exactly the "elegant on paper, not simplest that
   reaches parity" mistake this project's own engineering culture (quoted
   in the task brief) warns against. Semi-smooth Newton / Nitsche remain
   the correct Phase 4+ candidates if nonlinear penalty + soft contact +
   the current augmented-Lagrangian scope turn out insufficient once
   actually re-run against the Hertz benchmark — not before.
3. **The actually-cheap unification available right now** is §B.6's shared
   `PressureOverclosureLaw` seam: one `evaluate(h)` interface serving
   `HARD`, `A.2`'s `NONLINEAR` penalty, and all three soft laws, with
   `SurfaceContactConstraint3D.assemble()`'s downstream code (sign
   convention, stiffness scatter, stats) completely unchanged because it
   was already written generically enough. This is the "simplest design
   that reaches Abaqus parity on speed+stability+accuracy" the task brief
   asks for — it costs one small interface and a handful of law
   implementations, not a new solver-level formulation, and it is
   directly and only where Abaqus's own docs say unification is
   appropriate (the pressure-overclosure evaluation) rather than where
   they say it explicitly is not (augmented Lagrangian × soft contact).

---

## 4. Recommended build order (Phase 2.5 → 3, none authorized to start without separate go-ahead, per the existing design doc's own deferral framing)

1. **§A.2 nonlinear (4-region) penalty.** Cheapest, most directly targeted
   at the one concrete measured failure (Hertz curved-block divergence at
   node activation). Re-run `benchmark_element/benchmark_3d_contact.py`'s
   Hertz case with it before anything else — this is the acceptance test
   that actually matters here, not a new unit test in isolation.
2. **§B.6 the shared `PressureOverclosureLaw` seam + §B.2's three soft
   laws (LINEAR/EXPONENTIAL/TABULAR), built together with §B.7's
   `softness_scale`/`allowable_penetration` auto-derivation from the
   start** — not as a follow-on. B.7's resolution logic is what a caller
   actually calls (`ContactProperty(normal_behavior=..., softness_scale=...,
   allowable_penetration=...)`); the raw absolute-parameter law objects
   from §B.2 are what B.7's resolver produces internally. Building B.2
   without B.7 would ship exactly the "type in an absolute number you have
   no basis for" experience the auto-derivation requirement exists to
   avoid — land them as one unit. Land alongside #1 since #1's nonlinear
   penalty is naturally expressed through the same seam — no reason to
   build the seam twice.
3. **§A.1 automatic default penalty stiffness + §A.5's 1000× warning.**
   Independent of #1/#2, low-risk, removes the `1e8` magic constant.
4. **§B.4's augmented-Lagrangian restriction to `HARD` only** (a one-line
   guard) — land whenever #2 lands, since that's the point at which
   `normal_behavior != "HARD"` first becomes constructible.
5. **Re-verify augmented Lagrangian on the Hertz curved-block case**
   (still open from `dev_log/hertz_contact_benchmark_20260913.md`) —
   only meaningful after #1, since the plain-penalty Newton solve on that
   case needs to actually get through activation events first.
6. **§A.3's real contact-stabilization mechanism** — lowest priority;
   build only when a future example genuinely exhibits pre-contact
   rigid-body drift (none does today).
7. **§A.4's nonzero adjustment-zone `a`** — small, low-risk, no urgency;
   land opportunistically.
8. **Semi-smooth Newton / Nitsche (Part C's item 2)** — explicitly Phase
   4+, revisit only if 1-5 above are built and re-measured against Hertz
   and found insufficient.

---

## 5. New source material fetched this session

- `benchmark_element/reference_abaqus_docs/exp_pressure_overclosure_std.png`
  — Figure 4 from `simaitn-c-normalinteraction.htm` (same page
  `ctc_normalinteraction.txt` was already sourced from), fetched to confirm
  the exponential pressure-overclosure law has no printed closed-form
  equation on that page (§B.2's source gap). `README.md` updated with the
  exact source URL for both files.
- No other new pages were needed — `ctc_contactconstraints_std.txt`,
  `ctc_contactcontrols_std.txt`, `ctc_contactdamping.txt`,
  `ctc_normalinteraction.txt`, `ctc_adjustsurfaces_std.txt`,
  `ctc_genlcontinit_std.txt`, and `ctc_contactpairform_std.txt` (all
  already present from 2026-09-13) covered every claim above.
