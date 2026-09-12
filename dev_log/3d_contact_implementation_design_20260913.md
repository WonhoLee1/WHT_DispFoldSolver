# 3D Contact Implementation — Design (deferred, no timeline)

**Status: design only, not scheduled.** Per explicit user direction
(2026-09-13): "contact은 매우 어려운 분야. 설계 철저히. 기능 세분화
(abaqus 수준) 그래서 우선 순위는 후순위." (Contact is a very hard field.
Design thoroughly. Break features down at Abaqus granularity. So
priority is deferred.) This document exists so that WHEN contact is
picked up, it is picked up with a real plan instead of an ad hoc penalty
hack — not as a signal that it should be started now. See
`dev_log/3d_missing_capabilities_summary_20260913.md` item 5 for how this
fits among the other deferred capabilities.

Source material: 16 real Abaqus Interaction/Model Guide pages, fetched
2026-09-13 to `benchmark_element/reference_abaqus_docs/ctc_*.txt` (see
that folder's README for the list). All section references below
(`ctc_*.txt`) are to those files, not to memory or invention.

---

## 0. What already exists in this codebase (read before designing anything new)

`dispsolver/constraint3d/surface_tie3d.py` (`SurfaceTieConstraint3D`) is
a **bonded** node-to-Quad4-face penalty tie — no separation, no
friction, always-on. But it already contains most of the *mechanical*
machinery a first contact implementation needs:

- **Closest-point projection** (`_project_point_to_quad4`): Newton
  iteration in (ξ,η) to project a 3D point onto a bilinear quad face,
  clamped to the face boundary. This IS node-to-surface contact search
  in miniature — the same operation Abaqus's node-to-surface
  discretization performs to find each secondary node's "anchor point"
  (`ctc_contactpairform_std.txt`, §"Small-Sliding, Node-to-Surface
  Contact").
- **Reprojection under deformation** (`reproject_deformed`): re-solves
  the projection every Newton iteration from the current (not reference)
  geometry — exactly what a finite-sliding contact formulation requires
  (small-sliding contact would instead freeze the anchor point/tangent
  plane at its initial value, per `ctc_contactpairform_std.txt`'s anchor-
  point description).
- **Penalty force/stiffness assembly** in the 5-node (1 secondary + 4
  main-vertex) pattern already matches the shape a node-to-surface
  penalty contact constraint would take.
- **A dead scaffold worth knowing about**: `self._lam_dict` (per-slave-
  node "Augmented Lagrange multipliers", 3-vector) is declared in
  `__init__` but **never read or written in `assemble()`** — the tie is
  currently pure penalty despite the docstring/attribute name promising
  more. This is the same "declared but not wired in" pattern this
  project has hit repeatedly in the 2D element work (AGENTS.md
  §4.2/§4.8/§4.16) — not a contact bug (the tie doesn't claim to be
  augmented-Lagrange anywhere user-facing), but worth fixing or removing
  when this file is next touched, and a natural place to actually land
  the augmented-Lagrange loop this design recommends for Phase 2 (§4
  below).

**What's missing, concretely, relative to this**: (a) the gap must be
measured along the surface **normal only**, not the full 3D vector `xs -
xm_proj` the tie uses — contact must stay free to slide tangentially;
(b) the constraint must be **one-sided/activation-based** — force
applied only when the normal gap is negative (penetrating), exactly
zero when separated, which introduces a genuine non-smoothness (a
KKT/complementarity condition, `gap >= 0`, `pressure >= 0`,
`gap * pressure = 0`) that a bonded tie never has to deal with. This
one-sidedness — not the projection/search machinery, which mostly
already exists — is where essentially all of contact's real difficulty
lives (see §8).

`dispsolver/solver3d/dynamic3d.py`'s `solve_step()` (fixed this session
for a false-convergence bug, `dev_log/solve_step_false_convergence_20260913.md`)
is a plain Newton loop with a fixed-direction backtracking line search
and a single generic force-residual convergence check (`rel_r < 5e-3 or
r_norm < 1e-3`) — **no concept of a contact active-set at all**. Any
contact implementation needs this loop to (a) know which constraints are
currently active, (b) re-evaluate that set every iteration (or on a
controlled schedule), and (c) have a stopping criterion that also checks
active-set stability, not just force residual (§8).

`dispsolver/solver/stabilization.py`'s `AbaqusViscousStabilization`
(`F_stab = c*M_diag*v`, this project's existing viscous-damping device
for the 2D solver) is architecturally the same idea as Abaqus's contact
viscous damping (`ctc_contactdamping.txt`: a damping coefficient
proportional to relative normal velocity, active only while in contact,
zero otherwise) — reusable pattern, not reusable code (that one is
global/2D; contact damping needs to be per-contact-pair and normal-
direction-only), but confirms this project already has the right mental
model for this piece.

---

## 1. Scope decision: contact pairs vs. general contact

**Recommendation: build the equivalent of "contact pairs," not "general
contact," and build it as an explicit pairwise construct the user
specifies (mirroring how `surface_tie3d.py` and `RigidBodyPart` already
work) — not an automatic all-surfaces self-contact search.**

Reasoning, specific to this codebase rather than restating Abaqus's own
history: per `ctc_contactoverview.txt`, general contact's main
advantages (automatic surface detection, mixed surface types, self-
contact, better parallel scaling) solve problems this codebase doesn't
have yet — it has a small number of well-understood interfaces (one
display-to-plate tie per fold model, one or two RBE3 couplings), not
dozens of unknown potential contact pairs in a complex assembly. General
contact's whole value proposition is reducing the BURDEN of manually
specifying many interactions; there is no such burden here yet. Building
general contact first would mean building a real contact-pair search
engine (bounding-volume hierarchy, broad+narrow phase) as a prerequisite
to something simpler that would already deliver the needed capability.
Contact pairs, defined the same explicit way `SurfaceTieConstraint3D` is
constructed today (`slave_node_ids`, `master_faces` passed in directly),
require no new "which surfaces might touch" search infrastructure at
all for a first version — see §7.

---

## 2. Surface representation

`Mesh3D` + `surface_tie3d.py`'s existing pattern (a list of node IDs for
the secondary/slave side, a list of `(n1,n2,n3,n4)` face tuples for the
main/master side) is a usable **element-based deformable surface**
(`ctc_deformablesurf.txt`'s category) already. No new surface-definition
machinery is needed for deformable-vs-deformable contact — reuse this
exact convention for a first contact constraint class rather than
inventing a parallel one.

What's genuinely missing: an **analytical rigid surface**
(`ctc_rigidsurf.txt`) — a surface defined by a simple primitive (plane,
cylinder) rather than a mesh, with no discretization error and (per
Abaqus) generally better contact behavior against it than a faceted
rigid body. This codebase's existing `RigidBodyPart`
(`dispsolver/part/rigid_body.py`, 2D) models rigid bodies as ordinary
meshed parts with kinematic (Dirichlet) slave-displacement constraints,
not as analytical primitives — a rigid *contact* surface would be a new,
smaller concept (a plane/cylinder equation plus a rigid-body reference
node for reaction-force reporting), not a reuse of `RigidBodyPart`
directly. Given the Hertz contact acceptance target (§6, §9) is exactly
"deformable cylinder vs. flat rigid plane," a **flat analytical rigid
plane** is the only rigid-surface primitive that needs to exist for
Phase 1 — do not build a general rigid-surface library (spheres,
cylinders, arbitrary quadrics) speculatively.

Node-based surfaces (`ctc_nodebasedsurf.txt`) are not needed for a first
version — they exist in Abaqus mainly for point-cloud-like secondary
surfaces or numerically fragile secondary-side element types; this
codebase's element-based surfaces cover the target use cases.

---

## 3. Contact formulation: discretization and sliding-tracking choice

**Recommendation: small-sliding, node-to-surface, as the Phase 1
target — not finite-sliding, not surface-to-surface, despite both being
Abaqus's own defaults for contact pairs and general contact
respectively.**

- **Node-to-surface vs. surface-to-surface** (`ctc_contactpairform_std.txt`):
  node-to-surface is cheaper per constraint, generates one constraint
  per secondary node against a "flat approximation of the main surface"
  (exactly what `_project_point_to_quad4` already computes), and is what
  `surface_tie3d.py`'s existing machinery already IS, structurally.
  Surface-to-surface needs an averaged/smoothed constraint over multiple
  secondary nodes at once — a real additional formulation to derive and
  code, with no existing scaffold in this codebase. Node-to-surface is
  cheaper to build here specifically, not just cheaper in general.
- **Small-sliding vs. finite-sliding** (same source, "anchor point"
  section): small-sliding freezes each secondary node's projection
  (anchor point + local tangent plane) at its *initial* configuration and
  only updates the *normal gap* each iteration — cheaper, more robust,
  but only valid when relative sliding stays small compared to the
  facet size. Finite-sliding (what `surface_tie3d.py`'s
  `reproject_deformed` already effectively does — re-searching every
  iteration) is needed once bodies slide meaningfully relative to each
  other. **Recommendation: start with small-sliding anyway**, even
  though the reprojection machinery for finite-sliding already exists in
  `surface_tie3d.py` — small-sliding removes an entire class of
  Newton-iteration non-smoothness (the anchor point itself changing,
  which finite-sliding contact must also stabilize against, on top of
  the normal-gap activation non-smoothness every contact formulation
  has). Reintroduce finite-sliding only once small-sliding contact is
  proven correct and convergent on the Phase 1/2 targets (§9) — do not
  build both at once.

---

## 4. Constraint enforcement method: penalty vs. Lagrange multiplier vs. augmented Lagrange

Per `ctc_contactconstraints_std.txt`, Abaqus/Standard's actual defaults
are formulation-dependent: **node-to-surface contact pairs default to
the direct/Lagrange-multiplier method**; **surface-to-surface and
general contact default to penalty**; **3D node-to-surface self-contact
defaults to augmented Lagrange**.

**Recommendation: start with penalty, explicitly planning the
augmented-Lagrange upgrade as Phase 2, not Phase 1** — this is a
deliberate departure from Abaqus's own node-to-surface default (direct/
Lagrange), and the reason is architectural, not doctrinal:

- **The 2D precedent this project already has is a warning about
  penalty, not an endorsement** (AGENTS.md §4.10): `RBE2HingeElement`
  (penalty + augmented Lagrangian) was abandoned for driving a rigid
  plate because its finite penalty stiffness meant the constrained
  motion was never *exactly* rigid — the tie chasing a non-exactly-rigid
  target was the whole failure mode. **This precedent argues FOR
  eventually wanting an exact (Lagrange/augmented) method for contact
  too** — a permanently-approximate penalty contact could show the same
  class of symptom (small, load-dependent penetration that looks like a
  0.5% force-residual convergence artifact, not an obviously-wrong
  answer) that took real diagnostic effort to trace back in the RBE2
  case. Recording this now so whoever implements Phase 2 doesn't have to
  rediscover it.
- **Pure penalty is still the right STARTING point despite that
  warning**, for a concrete, narrower reason: a Lagrange-multiplier
  contact constraint adds a new solved unknown (the contact pressure)
  per active constraint, changes system size/sparsity pattern every time
  the active set changes, and produces an indefinite (not positive-
  definite) system — none of which this codebase's current linear-
  solve path (`pardiso_spsolve` in `dynamic3d.py`, expecting a fixed-size
  SPD-ish system) is set up to handle today. Penalty adds a normal-
  direction stiffness term directly into the existing displacement-only
  `K`/`f_int` assembly — no new DOFs, no solver-interface change,
  reusing exactly the pattern `surface_tie3d.py.assemble()` already
  demonstrates (return `f_tie`, `(rows, cols, data)` for the shared
  tail — see AGENTS.md §4.8 for why that shared-tail assembly point
  matters and must not be bypassed).
- **How it plugs into `dynamic3d.py` concretely**: a new
  `SurfaceContactConstraint3D` (parallel to `SurfaceTieConstraint3D`)
  added to `self.constraints` (the same list `assemble_system`'s shared
  tail already iterates per AGENTS.md §4.8's lesson — verify this
  session's contact addition follows that exact pattern, not a new
  bypassable branch), returning `(f_contact, (rows, cols, data))`
  exactly like the tie does, but computing gap and force only for
  constraints whose current normal gap is negative (the activation
  check, evaluated inside `assemble()`, fresh every call — this is the
  cheapest possible active-set update: recompute status every Newton
  iteration rather than freezing it, since `solve_step` re-assembles
  every iteration anyway).
- **Phase 2 (augmented Lagrange)**: reuses the same penalty assembly,
  wrapped in an outer augmentation loop around `solve_step` (per
  `ctc_contactconstraints_std.txt`'s own description: converge with
  penalty, check penetration against a tolerance, "augment" the
  effective contact pressure and re-solve, repeat) — this is exactly
  what `surface_tie3d.py`'s unused `_lam_dict` was evidently scaffolded
  for. Land it there when this phase is reached, rather than duplicating
  a second penalty-tie-like class.
- **Direct/Lagrange multiplier**: deliberately deferred past Phase 2,
  possibly indefinitely — it requires a genuine solver-architecture
  change (mixed/indefinite system, active-set-dependent sparsity) that
  is a bigger lift than either phase above and is not obviously worth it
  once augmented Lagrange already gets penetration small.

---

## 5. Normal contact behavior (pressure-overclosure)

**Recommendation: "hard" contact only for Phase 1** (`ctc_normalinteraction.txt`:
zero pressure for positive clearance, unbounded pressure for zero
clearance, no penetration in the idealized/direct-method limit). Softened
(linear/exponential/tabular) contact exists in Abaqus mainly for
numerical conditioning tricks and for physically-soft interfaces
(gaskets, rubber seals) — neither motivation applies to this codebase's
current example problems (rigid plate / display panel folding, and the
Hertz/thick-cylinder acceptance targets, all of which are "hard" contact
in the physical sense). Do not build the softened variants speculatively;
add them only when a specific problem needs one.

---

## 6. Friction: explicitly deferred past Phase 1 and Phase 2

**Recommendation: frictionless normal contact only through at least
Phase 2.** This is a real, defensible scoping choice, not corner-cutting:

- Every acceptance target currently in reach (`bmk_hertzcontact` —
  "smooth contact (no friction) is assumed," verified directly in
  `reference_abaqus_docs/bmk_hertzcontact.txt`; `bmk_pipecrushing` — also
  explicitly frictionless per that same file, though NOT-APPLICABLE here
  for unrelated reasons, §9) is frictionless in the original Abaqus
  benchmark itself. There is no benchmark currently motivating friction.
- Friction (`ctc_friction.txt`) adds its own enforcement-method choice
  (penalty/"stick-slip" vs. Lagrange, mirroring the normal-direction
  choice in §4 but for the TANGENT plane) and its own activation
  non-smoothness (sticking vs. slipping is itself a complementarity
  condition, layered on top of the normal contact/separation one) — a
  second, mostly-independent source of the same class of difficulty
  §8 describes, not a small add-on to normal contact.
- Once frictionless normal contact is verified (Phase 1/2), isotropic
  Coulomb friction with penalty (stick-modeled-as-stiff-elastic-slip)
  enforcement is the natural Phase 3 addition — anisotropic friction,
  friction that depends on temperature/field variables, and user-defined
  interfacial behavior (`ctc_userinteraction.txt`) are not worth
  planning in detail until Phase 3 is reached.

---

## 7. Contact initialization / search

**Recommendation: brute-force, exhaustive candidate search is
acceptable for Phase 1 — say so explicitly rather than building a real
broad-phase search speculatively.** `surface_tie3d.py`'s
`_build_tie_pairs`/`reproject_deformed` already do exactly this (loop
every secondary node over every master face, O(n_secondary ×
n_master_faces)) and it's adequate for this codebase's current model
sizes (tens to low hundreds of contact-candidate nodes/faces in the
existing fold examples). This stops scaling once a model has thousands
of candidate faces — a bounding-volume hierarchy / spatial hash would
be needed then, but building one now, before any model needs it, would
be exactly the kind of speculative infrastructure this project's own
engineering culture (AGENTS.md throughout — measure the actual problem
before generalizing) argues against.

**Initial overclosure/gap handling** (`ctc_genlcontinit_std.txt`,
`ctc_adjustsurfaces_std.txt`): Abaqus's two conventions — general
contact eliminates initial overclosures with a "strain-free adjustment"
(silently move the mesh so surfaces start exactly touching, no
resulting stress), contact pairs by default treat initial overclosure as
a real interference fit to be resolved with real stress in the first
increment. **Recommendation: strain-free adjustment for Phase 1** — it's
simpler to implement (a one-time geometry nudge before the first
assembly, not a physical loading step) and matches this codebase's
existing examples, which are built to start from an already-correct,
non-interfering initial geometry (no interference-fit problem currently
motivates the other convention).

---

## 8. Convergence/robustness concerns specific to contact — why `solve_step` needs more than it has today

This is the section that most directly explains why contact is hard
here specifically, not just in general:

1. **Active-set instability ("chattering").** A contact constraint
   flipping between active (penetrating) and inactive (separated)
   between consecutive Newton iterations is contact's classic source of
   non-convergence — the residual can oscillate indefinitely rather than
   decrease, because the *set of equations being solved* changes each
   iteration, not just their values. `solve_step`'s current convergence
   check (`rel_r < 5e-3 or r_norm < 1e-3`, no other state tracked) has no
   way to detect this — it would either accept a false convergence at a
   fortunate moment mid-oscillation, or run out `max_iters` and (since
   this session's fix) correctly report `DIVERGED`, but with no
   diagnostic telling a user *why* (contact chattering vs. any other
   Newton failure look identical to this loop today). **A real contact
   implementation needs the active-set itself checked for stability
   across iterations** (e.g., track whether the active set is unchanged
   for N consecutive iterations as part of the convergence criterion,
   the same spirit as Abaqus's own contact-specific controls,
   `ctc_contactcontrols_std.txt`) — this is new logic `solve_step`
   doesn't have any version of today, not a parameter tweak.
2. **Contact stabilization/damping** (`ctc_contactdamping.txt`): a
   normal-direction viscous term, active only while in contact, damps
   exactly the chattering mode in (1). This project already has the
   right *concept* (`AbaqusViscousStabilization`, §0) but would need a
   per-contact-pair, normal-direction-only version, evaluated inside the
   same constraint's `assemble()`, not the global displacement-based
   stabilization currently used for the unrelated purpose of damping
   rigid-body drift in quasi-static 2D problems.
3. **Overconstraint.** Abaqus's own documentation
   (`ctc_contactconstraints_std.txt`) repeatedly warns about
   overconstraint from combining contact with other constraint types at
   the same node (ties, MPCs, boundary conditions). This codebase
   already has ties and RBE3 in 3D — a contact constraint sharing a node
   with an existing tie or RBE3 is a real, checkable failure mode to
   design a guard against before it's hit silently (matching this
   project's own AGENTS.md §4.8 lesson: an unchecked interaction between
   two constraint types produced a silent, wrong-physics failure before,
   for ties specifically).
4. **Line search interacting badly with a discontinuous residual.**
   `solve_step`'s backtracking line search (`s *= 0.5` until
   `r_t_norm <= max(r_norm, ...)`) assumes the residual varies smoothly
   along the Newton direction. A contact activation flip partway along
   that direction (an element newly penetrating, or newly separating, at
   some fractional step `s`) makes the residual's dependence on `s`
   piecewise, not smooth — the existing line search may accept a step
   that's on the "wrong side" of an activation event without knowing it.
   Not necessarily fatal (penalty contact's force is still continuous,
   just with a slope discontinuity at zero gap, not a jump), but worth
   flagging as untested territory for the existing line-search code.

---

## 9. Phased plan

Every phase below is a **candidate**, not a schedule — none of this is
authorized to start without a separate, explicit go-ahead per the
user's deferral direction.

### Phase 0 — Prerequisite cleanup (small, could ride along with unrelated work)
- Resolve `surface_tie3d.py`'s unused `_lam_dict` scaffold (either wire
  it into an augmented-Lagrange tie option, matching Phase 2's method
  below, or remove it if it's confirmed genuinely dead) — not contact
  work per se, but the natural low-risk warm-up since it touches the
  exact file Phase 1 extends.

### Phase 1 — Frictionless, hard, small-sliding, node-to-surface, penalty contact; one deformable body vs. one analytical rigid plane
- New `dispsolver/constraint3d/surface_contact3d.py`,
  `SurfaceContactConstraint3D`, built by extracting/adapting
  `surface_tie3d.py`'s projection machinery: normal-only gap (not the
  tie's full 3D gap vector), one-sided force (zero for positive gap),
  frozen anchor point per node (small-sliding), initial strain-free
  overclosure adjustment (§7).
- A minimal analytical rigid plane primitive (a point + normal vector;
  no meshing) as the main surface, reporting reaction force through a
  single reference DOF (reusing `RigidBodyPart`'s existing reaction-
  force pattern where possible, §2).
- **Acceptance test: `bmk_hertzcontact`** (`reference_abaqus_docs/bmk_hertzcontact.txt`) —
  **reviewed against the source a second time (2026-09-13) and confirmed,
  not merely assumed**: the original problem is two IDENTICAL cylinders
  pressed together, but the source states outright, "Because of symmetry,
  the contact problem can be modeled as a deformable cylinder being
  pressed against a flat, rigid surface" — i.e. Phase 1's "one deformable
  body vs. one analytical rigid plane" target is not an approximation
  invented for this design, it is the SAME reduction the real Abaqus
  benchmark itself uses. Also confirmed directly against the source on
  this pass: "makes this problem a candidate for the small-sliding
  contact formulation" (matches §3's recommendation) and "the default
  contact pair formulation in the normal direction ... is hard contact"
  (matches §5). Parameters: cylinder radius 254mm, E=206GPa, nu=0.3,
  frictionless, geometric nonlinearity explicitly ignored, load applied
  as a 10.16mm prescribed displacement on the diametric cut (read the
  resulting line load off the rigid reference node's reaction force, same
  pattern this codebase's `RigidBodyPart` already reports reactions
  through). **Still not done, correctly flagged as a gap**: the source
  page itself gives only qualitative/figure-based agreement ("within 1%
  of the classical solution"), not a printed closed-form number — the
  actual Hertz cylindrical-contact formulas (contact half-width `b`, peak
  pressure `p0`, both functions of the line load, `R`, and the combined
  modulus `E* ` from `(1-nu^2)/E` for the deformable side only, since the
  rigid plane contributes no compliance term) still need to be derived
  fresh and carefully from Timoshenko & Goodier (1951) when Phase 1 is
  actually started — do not lift a half-remembered version of this
  formula from general knowledge without re-deriving/cross-checking it
  then, given this project's own standing rule about not trusting an
  unverified number.
- Explicitly NOT in Phase 1: friction, finite-sliding, deformable-vs-
  deformable contact, augmented Lagrange, softened contact.

### Phase 2 — Augmented Lagrange + deformable-vs-deformable contact
- Wire the augmentation loop into the same constraint class (or a
  subclass), penetration-tolerance-driven re-solve per
  `ctc_contactconstraints_std.txt`'s described algorithm.
- Extend from body-vs.-rigid-plane to body-vs.-body (both element-based
  deformable surfaces) — mechanically similar to Phase 1's projection,
  but now BOTH sides move, so main/secondary role assignment
  (`ctc_contactoverview.txt`'s "pure main-secondary" note) and which
  side projects onto which need an explicit, documented choice, not an
  implicit one.
- Add the active-set-stability convergence check and contact damping
  from §8 — Phase 1's simpler geometry (rigid plane, single well-
  separated contact patch) is much less likely to expose chattering than
  a real deformable-vs-deformable case, so this is a reasonable point to
  add real robustness machinery rather than before it's needed.
- Acceptance target: TBD when reached — no currently-reviewed benchmark
  from the static-analysis list (`dev_log/static_analysis_benchmark_design_20260913.md`)
  is a clean deformable-vs-deformable frictionless case; would need a
  fresh literature check at that time, or an in-house-designed test
  (two elastic blocks compressed together, checked against a simpler
  closed-form or a mesh-refinement convergence study rather than an
  external reference).

### Phase 3 — Frictionless-to-friction
- Isotropic Coulomb friction, penalty enforcement in the tangent plane
  (mirrors §4's normal-direction choice).
- No acceptance target currently identified from the reviewed benchmark
  list — `bmk_pipecrushing` is frictionless too and NOT-APPLICABLE for
  unrelated (2D/shell/no-clean-number) reasons anyway.

### Phase 4+ — Finite-sliding, surface-to-surface, general contact, softened contact, thermal/other physics
- Not designed in any detail here — each is a large, separable
  extension of Phase 1-3's foundation, and none has a currently-
  identified motivating problem in this codebase. Revisit if/when one
  does.

---

## 10. Why this is hard, honestly

This section is deliberately blunt, per the user's own framing
("contact은 매우 어려운 분야"):

- **Contact is not a material model or an element formulation — it's a
  changing SET of equations.** Every other nonlinearity this codebase
  handles (large deformation, plasticity, hyperelasticity) is a smooth
  (or piecewise-smooth-but-fixed-structure) function of displacement.
  Contact's active set is discovered DURING the solve and can flip
  every iteration — Newton's method's entire convergence theory assumes
  a fixed, sufficiently smooth residual, which contact violates by
  construction. Every numerical control Abaqus exposes for contact
  (stabilization, augmentation, controlled main-secondary weighting,
  penetration tolerances) exists to manage this one structural fact, not
  a collection of unrelated tuning knobs.
- **Every formulation choice in §§3-5 trades robustness for accuracy (or
  vice versa), with no universally-correct answer.** Node-to-surface is
  cheap and robust but discretization-accuracy-limited (per
  `ctc_contactpairform_std.txt`'s own pressure-accuracy comparison
  figure); surface-to-surface is more accurate but costs more per
  iteration; penalty is easy to implement and converges more readily but
  is never exact; Lagrange multiplier is exact but changes the whole
  system's structure and conditioning. Abaqus's own defaults differ by
  context (contact pairs vs. general contact, 2D vs. 3D self-contact)
  BECAUSE there is no single best choice — this design's own
  recommendations (§§3-4) are informed defaults for THIS codebase's
  current scale and examples, not universal truths, and should be
  revisited if the actual use case changes materially before
  implementation starts.
- **This codebase's own history already contains one real lesson about
  underestimating this class of problem** (AGENTS.md §4.10's RBE2/penalty
  failure) — the honest takeaway is not "contact will definitely fail
  the same way," but "assume it needs the same level of skepticism and
  the same willingness to abandon an approach that's structurally wrong,
  not just under-tuned, before committing to a specific enforcement
  method for real production use."
