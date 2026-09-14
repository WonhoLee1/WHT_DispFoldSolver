# Contact beyond Abaqus: is semi-smooth Newton / Nitsche worth adopting now for deformable-vs-deformable contact? (2026-09-15)

**Status: research only, per task scope — no `dispsolver/` code changed in this
session.** This continues the lineage of `dev_log/3d_contact_implementation_design_20260913.md`
(Phase 1 plan), `dev_log/hertz_contact_benchmark_20260913.md` (measured
rigid-plane results), and `dev_log/contact_abaqus_grade_design_20260915.md`
("Part C", §4 item 8) — that document already ranked semi-smooth Newton /
Nitsche as "Phase 4+, revisit only if [SDI + nonlinear penalty + augmented
Lagrangian] turn out insufficient once actually re-run against the Hertz
benchmark." Per the task brief, that gate has now been treated as cleared
(nonlinear penalty alone fixed the one concrete measured Hertz-divergence
failure), so this document evaluates the Phase-4+ candidates on their own
merits for the *harder*, currently-being-built deformable-vs-deformable case,
rather than re-deriving the rigid-plane conclusions.

Every claim below is either cited to a specific real, identifiable source
(title/authors/year/venue, URL where fetched from the web) or to a specific
piece of code/measurement already in this repository. Where the source
material itself has a gap (an unpublished internal parameter, an unresolved
comparison), that is stated as a gap, not filled with an invented number —
matching the discipline the two prior 2026-09-15 documents already used.

---

## 1. What already exists in this codebase, precisely stated (baseline for the "is it worth it" comparison)

- `dispsolver/constraint3d/surface_contact3d.py`'s `SurfaceContactConstraint3D`:
  node-to-analytical-rigid-plane, frictionless, penalty **or** augmented
  Lagrangian. Read end to end for this document (lines 1-478).
- The augmented-Lagrangian branch (`assemble()`, ~line 296) already computes
  `f_mag = self._lam[s_nid] + self.k_contact * (-gap)`, with the comment
  correctly identifying this as "the standard augmented Lagrangian /
  Alart-Curnier projection for a unilateral constraint." **This is worth
  stating precisely, because it changes the size of the gap between what
  exists and what semi-smooth Newton would add**: the codebase already
  evaluates the exact Alart-Curnier normal NCP function,
  `p = max(0, λ + k(-gap))`. What it does *not* do is treat `λ` as an
  unknown solved *simultaneously* with `u` inside one Newton system — `λ`
  is updated only in an **outer Uzawa loop**, after each inner Newton solve
  fully converges (`solve_step_augmented()`, `dispsolver/solver3d/dynamic3d.py:788-` —
  `max_augment_iters`, `augment_tol`, a separate loop wrapped around
  `solve_step()`). That distinction — projection function already present,
  outer-loop vs. simultaneous-Newton treatment of `λ` — is the exact axis
  semi-smooth Newton would move along, not a wholesale reformulation.
- `dispsolver/solver3d/dynamic3d.py`'s `solve_step()` SDI handling
  (`max_sdi_iters`, lines ~595-793 per this session's own grep): exempts an
  iteration where `_contact_active_set()` just changed from the normal
  convergence check, giving it its own iteration budget — this is the
  mechanism that already absorbs the residual discontinuity at activation.
- `dispsolver/constraint3d/pressure_overclosure.py`'s `PressureOverclosureLaw`
  seam (`HardLaw`, `NonlinearPenaltyLaw`, soft laws) and
  `dispsolver/constraint3d/contact_stiffness.py`'s
  `representative_element_stiffness` — an auto-derived reference stiffness,
  already usable as the `k_ref`/`k_e` basis for whatever comes next.
- `dispsolver/constraint3d/surface_tie3d.py`'s `_project_point_to_quad4`:
  Newton-iterated parametric projection of a point onto a Quad4 face
  (2×2 local linear solve on `[J11,J12;J12,J22]`, clipped `ξ,η∈[-2,2]`),
  frozen small-sliding pairing — this is the piece the deformable-vs-
  deformable contact work in progress reuses for a moving master surface,
  and the piece any mortar-style upgrade would also reuse (§4).
- Measured, sourced fact this session already has (not re-derived here):
  switching the Hertz rigid-plane benchmark from linear to `NonlinearPenaltyLaw`
  took it from diverging at ~45 active contact nodes to converging through
  the full load ramp, zero cutbacks, with contact half-width ratio improving
  from 1.40-1.61 to 0.78 vs. the closed-form Hertz reference
  (per the task brief; this document does not re-run that benchmark).

---

## 2. Literature: Alart-Curnier semi-smooth Newton, specifically for deformable-vs-deformable contact

### 2.1 Foundational formulation

- **P. Alart, A. Curnier (1991)**, "A mixed formulation for frictional contact
  problems prone to Newton like solution methods," *Computer Methods in
  Applied Mechanics and Engineering*, 92(3), 353-375. Reformulates the
  normal (and frictional) complementarity condition as a single nonsmooth
  equation (`p = max(0, λ - k·g)`-type projection), so contact + equilibrium
  becomes one system of nonlinear equations solvable by a generalized/
  semi-smooth Newton method.
- **C. Curnier, P. Alart**, "A Generalized Newton Method for Contact
  Problems with Friction," *Journal de Mécanique Théorique et Appliquée*
  (special issue on numerical methods in mechanics of contact with
  friction) — companion paper establishing the semi-smooth Newton
  convergence machinery for the same projection.

### 2.2 Genuinely deformable-vs-deformable: the equivalence that matters

- **B. Wohlmuth and collaborators / S. Hüeber, B. Wohlmuth (2005)**,
  "A primal-dual active set strategy for non-linear multibody contact
  problems," *Computer Methods in Applied Mechanics and Engineering*, 194,
  3147-3166. This is the paper that matters most for this task's question
  #3(a): it proves the **primal-dual active-set strategy (PDASS) is
  equivalent to a semi-smooth Newton method on the Alart-Curnier NCP
  function**. PDASS is exactly what makes the mixed (Lagrange-multiplier)
  formulation tractable for deformable-vs-deformable contact without
  paying for a full saddle-point solve: at each Newton step, nodes are
  partitioned into an active/inactive set; for **inactive** nodes the
  multiplier is fixed at `λ=0` and the row degenerates to the ordinary
  free-node equilibrium equation; for **active** nodes the row becomes the
  exact geometric constraint (`g(u)=0` in the frictionless case) and `λ`
  is *read off directly* from that node's own equilibrium equation — i.e.
  `λ` is eliminated **locally, node-by-node**, not via a global
  saddle-point solve. This local eliminability is the concrete answer to
  "does the natural formulation stay displacement-only": yes, for the
  penalty-regularized / PDASS form specifically, not for the pure exact-
  Lagrange mixed form (matching the task brief's own framing of the
  question).
- **A. Popp, B. I. Wohlmuth, M. W. Gee, W. A. Wall (2009 and 2012)**,
  "A Primal-Dual Active Set Strategy for Finite Deformation Dual Mortar
  Contact" / related dual-mortar PDASS papers, and **M. Gitterle, A. Popp,
  M. W. Gee, W. A. Wall (2010)**, "Finite deformation frictional mortar
  contact using a semi-smooth Newton method with consistent linearization,"
  *International Journal for Numerical Methods in Engineering*, 84(5),
  543-571. This is the production-grade generalization of §2.2's PDASS
  equivalence to genuinely **deformable-vs-deformable, finite-deformation,
  segment-to-segment (mortar)** contact — the exact regime this session's
  new work is targeting, not the rigid-plane case. Key facts, all directly
  from the search summaries of these papers (title/venue/claims
  cross-checked across the Wiley IJNME listing and the Springer chapter
  listing found this session; full text was paywalled, so the specific
  numerical convergence-rate figures in these papers are **not** re-quoted
  here — flagged as a genuine access gap, not filled with an invented
  number):
  1. Mortar contact naturally yields a **mixed formulation** with nodal
     Lagrange-multiplier DOFs as extra primary unknowns on the contact
     interface.
  2. Using **dual** (biorthogonal) shape functions for the multiplier
     interpolation — rather than the same shape functions as the
     displacement field — makes the mortar mass-matrix-like coupling
     matrix diagonal, which is exactly what allows the multipliers to be
     **statically condensed out of the global system before solving**, so
     the assembled linear system's size and (to first order) sparsity
     pattern match a displacement-only formulation, not a saddle-point
     system.
  3. All nonlinearities — finite deformation, nonlinear material
     behavior, **and** the contact active-set search itself — are handled
     inside **one single Newton iteration**, not a separate outer loop
     around a converged inner solve (the direct contrast with this
     codebase's current `solve_step_augmented()` Uzawa wrapper, §1).
  4. This is not a niche academic formulation: dual-mortar + PDASS contact
     is the production contact algorithm in the BACI/4C research finite-
     element code (Wall's group, TU Munich) and the same underlying
     mathematics is cited by Abaqus's own documentation discussion of
     mortar-style surface-to-surface enforcement (a separate search this
     session found: "a mortar method combined with penalty enforcement
     leads to unphysical penetrations; a Lagrange-multiplier mortar method
     needs extra unknowns, resolved via dual shape functions" — same
     conclusion, independently corroborated). This is the "established in
     production FEA codes or well-cited research codes" bar the task set,
     not just an arXiv preprint.

### 2.3 A genuinely graphics/robotics-adjacent 2019 alternative — cited, but flagged as the wrong infrastructure fit

- **M. Macklin, K. Erleben, M. Müller, N. Chentanez, S. Jeschke,
  V. Makoviychuk (2019)**, "Non-Smooth Newton Methods for Deformable
  Multi-Body Dynamics," *ACM Transactions on Graphics*, 38(5), Article 140
  (arXiv:1907.04587). Solves the contact/friction NCP directly via a
  non-smooth Newton iteration with a purpose-built "complementarity
  preconditioner," using a GPU-based **iterative** (conjugate-residual)
  linear solver for interactive-rate hyperelastic deformable bodies plus
  articulated rigid mechanisms. This is real, current (2019), and
  genuinely NCP-direct (not penalty), so it answers part of the task's
  question #2 ("is there something even newer"). **Explicitly not
  recommended as an infrastructure match for this project**: it is built
  around an iterative Krylov solve with a custom preconditioner tuned for
  interactive/graphics use, not a direct sparse (PARDISO) factorization —
  adopting it would mean replacing this project's entire linear-solve
  backend, which is exactly the "no full architecture change" constraint
  the task ruled out. Its actual contribution (a non-smooth-Newton
  treatment of the same class of NCP function as Alart-Curnier, plus a
  preconditioner) is conceptually adjacent to §2.2 but solves a different
  engineering problem (interactive rigid+soft-body simulation, not
  quasi-static implicit FEA with a direct solver).

---

## 3. Literature: Nitsche-based contact, specifically for deformable-vs-deformable

### 3.1 Foundational and directly-on-topic sources

- **F. Chouly, P. Hild (2013)**, "A Nitsche-based method for unilateral
  contact problems: numerical analysis," *SIAM Journal on Numerical
  Analysis*, 51(2), 1295-1307 — single-body (rigid-obstacle) foundation.
- **F. Chouly, P. Hild, Y. Renard (2015)**, "Symmetric and non-symmetric
  variants of Nitsche's method for contact problems in elasticity: theory
  and numerical experiments," *Mathematics of Computation*, 84(293),
  1089-1112. Introduces a one-parameter family of variants (`θ`): `θ=1`
  symmetric, `θ=0` non-symmetric, `θ=-1` skew-symmetric. Per this session's
  search of the paper's own abstract/summary: **the method is primal — no
  Lagrange multiplier, no extra unknowns, no discrete inf-sup condition to
  satisfy** (direct contrast with the mixed mortar formulation of §2.2 —
  Nitsche buys displacement-only-ness *without* needing PDASS's active-set
  condensation trick at all, because there is no multiplier to condense in
  the first place). The **skew-symmetric (`θ=-1`) variant is reported as
  "much more robust regarding Nitsche's parameter"** than the symmetric
  one, and the **non-symmetric (`θ=0`) variant needs fewer Newton
  iterations to converge over a wider range of the Nitsche parameter**
  than the symmetric (`θ=1`) variant, and is also cited as easier to
  extend to nonlinear elasticity (fewer terms in the weak form). This is
  the most directly relevant sourced finding for the task's activation-
  robustness question on the *parameter-sensitivity* axis specifically —
  see §3.3 for why it does not touch the *activation-event* axis.
- **F. Chouly, R. Mlika, Y. Renard (2018)**, "An unbiased Nitsche's
  approximation of the frictional contact between two elastic structures,"
  *Numerische Mathematik*, 139, 593-631. **This is the paper that directly
  answers the task's "specifically for deformable-vs-deformable" ask**:
  most contact methods (including this codebase's current node-to-plane
  and node-to-Quad4-face projection) use a master/slave split, which the
  paper's own stated motivation identifies as a source of detection
  difficulty for self-contact and genuinely symmetric multi-body contact.
  The unbiased Nitsche formulation treats both bodies identically — same
  contact condition evaluated on each surface, no nominated master — and
  the paper proves well-posedness and optimal `H¹`-norm convergence for
  the small-deformation case.
- **R. Mlika, Y. Renard, F. Chouly (2017)**, "An unbiased Nitsche's
  formulation of large deformation frictional contact and self-contact,"
  *Computer Methods in Applied Mechanics and Engineering*, 325, 265-288.
  Extends the unbiased Nitsche idea to finite/large deformation and true
  self-contact within the same formalism (searched and corrected this
  session — an earlier draft of this document mis-attributed this paper's
  year/author order via a search snippet; verified as Mlika/Renard/Chouly,
  CMAME vol. 325, 2017, not a 2018 Seitz/Wall/Popp paper as an initial
  search phrasing implied).
- **Y. Renard (2016/2018)**, "An overview of recent results on Nitsche's
  method for contact problems," in *Geometrically Unfitted Finite Element
  Methods and Applications* (Springer Lecture Notes in Computational
  Science and Engineering, vol. 121), also posted as a preprint at
  `math.univ-lyon1.fr/~renard/papers/2016_overview_Nitsche_contact.pdf`.
  **Access gap, stated honestly**: the direct PDF fetch of this preprint
  returned HTTP 404 this session (the file may have moved or been
  retracted from that path) — its content is not independently re-verified
  here beyond what the search-result summaries of the sibling papers above
  already corroborate (primal formulation, consistency vs. penalty). Not
  treated as load-bearing beyond that corroboration.

### 3.2 What "consistent" buys, precisely, vs. this codebase's current penalty law

A genuinely useful, sourced distinction (from the Chouly/Hild/Renard line
above, corroborated independently by the search summary's own explicit
statement): **Nitsche's method is consistent; classical penalty is not.**
Concretely, a penalty method (linear or nonlinear, §A.2/§B of the prior
design doc) converges to the exact contact solution only in the limit
`k→∞` — at any finite stiffness there is a systematic (mesh- and
stiffness-dependent) penetration bias, which is exactly why this
codebase's whole `representative_element_stiffness`/`softness_scale`
apparatus exists (to choose a *good enough* finite `k`, not an exact one).
Nitsche's method instead adds a *consistent* term to the weak form (built
from the true traction, not an artificial spring), so it converges to the
exact solution at the discretization's own optimal rate, independent of
how the Nitsche parameter `γ` is chosen within a stability range — the
role `γ` plays is closer to a *stabilization* parameter (bounded below by
a trace-inequality constant tied to element size/material stiffness, the
same kind of quantity `representative_element_stiffness` already
estimates) than to `k_contact`'s role of trading accuracy for robustness.

### 3.3 What Nitsche does *not* change — directly relevant correction to a plausible-sounding claim

**This is the load-bearing finding for the task's specific "does this fix
the measured activation-discontinuity problem" question, and the honest
answer is no, for the same structural reason found in
`contact_abaqus_grade_design_20260915.md`'s §B.3 for soft-contact laws.**
Nitsche's contact term is still built around a one-sided projection of the
normal traction (`[·]_-`/`max(0,·)`-shaped, evaluated at the *current*
weak-form traction rather than at a separately-solved Lagrange multiplier
— that substitution is precisely what makes it primal) — the
complementarity switch (active vs. inactive) is still there, still
piecewise-defined, still not `C¹` at the switch. What Nitsche's own
literature reports improving is **parameter-choice robustness within the
active regime** (fewer iterations, wider stable range of `γ`, per §3.1) —
a different axis from **the activation-event residual jump** SDI already
targets. This generalizes the same conclusion §B.3 of the prior design doc
already reached about soft pressure-overclosure laws: **every formulation
surveyed across both this document and the prior one still needs something
like SDI's iteration-exemption at the exact instant a node's status
flips**, because that discontinuity is a structural consequence of the
complementarity condition itself, not of which method (penalty, soft law,
Alart-Curnier, or Nitsche) evaluates the active branch.

---

## 4. Direct answers to the four numbered questions in the task brief

### 4.1 Genuinely newer refinements targeting activation-robustness specifically (2018-2026)?

**Partial, not a clean yes.** The two candidate lines (§2.2's PDASS/dual-
mortar semi-smooth Newton, §3.1's Nitsche `θ`-family) both post-date and
build on Alart-Curnier, but neither paper reviewed here claims to have
*removed* the activation-event discontinuity itself — Nitsche's own
reported gains (§3.1/§3.3) are about `γ`-sensitivity and iteration count
*given* an already-active or already-inactive state, not about softening
the switch event. The one 2019 paper found that treats the NCP function's
non-smoothness most directly (Macklin et al., §2.3) targets an iterative-
solver infrastructure this project isn't adopting. **No source found in
this session's search claims a method that eliminates the switch itself**
— the honest, sourced conclusion is that the switch is structural (§3.3),
and every method in the literature surveyed here still needs an SDI-class
mechanism around it, exactly as `contact_abaqus_grade_design_20260915.md`
already concluded for soft-contact laws.

### 4.2 Anything newer/better than both Alart-Curnier and Nitsche?

The most relevant recent (2025) work found, **I. Sahu, N. Petrinic**,
"Unbiased higher-order frictional contact using midplane and patch based
segment-to-segment penalty method" (arXiv:2506.21767, physics.comp-ph, not
yet peer-reviewed — flagged as a preprint, per the task's own "not just
arXiv preprints" caution), targets **deformable-vs-deformable**
segment-to-segment contact with **no added DOFs** — but it is still a
**penalty** method (the paper's own description: "normal traction depends
upon the penalisation of true interpenetration"), refined for higher-order
element geometry accuracy (a midplane/patch construction correcting for
surface curvature), not a new complementarity treatment. This is useful
evidence for a different point than "what's newest": **the plain
projection-based penalty approach is still an active, credible research
direction in 2025 for exactly this problem class (deformable-vs-
deformable)**, not a technique the field has moved past. No source found
in this search that is both (a) more recent than semi-smooth Newton/
Nitsche and (b) a genuinely different complementarity treatment for
deformable-vs-deformable contact, beyond the smoothing-function variants
(Fischer-Burmeister, §4.3) already known from the earlier 2000s literature.

### 4.3 Fischer-Burmeister NCP function as an Alart-Curnier alternative

Found and worth recording, but with an honest gap: Fischer-Burmeister is a
well-known NCP function, `C¹` everywhere except at the origin (vs.
Alart-Curnier's max-projection, which is Lipschitz but has a kink along a
manifold, not just at a point) — in principle a smoother complementarity
function for a semismooth Newton scheme. A specific comparative study,
**"A comparative study between two smoothing strategies for the simulation
of contact with large sliding,"** *Computational Mechanics* (2012), exists
and is directly on this question, but its full text was paywalled this
session (Springer redirect to an authentication wall) — **its actual
verdict on which formulation is more robust is not independently
confirmed here and should not be treated as settled by this document.**
General knowledge of this literature area (not from a fetched source, so
flagged as such rather than cited) is that production mortar/PDASS codes
(§2.2) predominantly use the Alart-Curnier/max-projection form, not
Fischer-Burmeister, but this document does not have a directly-fetched
source confirming *why* — treated as an open gap, not resolved here.

### 4.4 Implementability, concretely, against this project's actual constraints

| Candidate | Needs indefinite/mixed system? | Size of change | Composes with SDI? |
|---|---|---|---|
| Nonlinear penalty (already designed, `contact_abaqus_grade_design_20260915.md` §A.2) | No — pure displacement-only, already the current architecture | One piecewise function replacing one line | Yes, by construction (§B.3 of that doc) |
| Penalty-regularized Alart-Curnier / PDASS, deformable-vs-deformable (§2.2) | **No** for the penalty-regularized/PDASS form specifically — `λ` eliminated **locally, per contact node**, via the same active/inactive branch this codebase's `get_active_set()` already computes every iteration | Moderate: promote `λ` from "outer-Uzawa-frozen" to "solved inside the same Newton iterate via the existing per-node active-set branch," generalize the point-projection in `surface_tie3d.py` to a genuine segment-to-segment (mortar) weighted-gap integral for two moving surfaces instead of one fixed rigid plane | **Composes directly** — PDASS's own active-set update *is* the mechanism SDI's iteration-exemption already protects; this is not a competing mechanism, it's SDI's natural counterpart on the constraint-equation side |
| Full mixed (exact-Lagrange) dual-mortar semi-smooth Newton (§2.2, without PDASS's local condensation) | Yes — a genuine saddle-point/KKT system, one extra scalar DOF per active contact node, needing either a Schur-complement wrapper around the existing PARDISO call or a different solver entirely | Large: new global DOF bookkeeping, new sparsity pattern, careful indefinite-system handling | Needs re-deriving how SDI's iteration-exemption interacts with a system that now has multiplier DOFs — not free |
| Unbiased Nitsche, deformable-vs-deformable (§3.1/§3.2) | No — primal by construction, no multiplier DOF of any kind | Large in a different way: requires assembling a *consistency* term built from the true traction/normal derivative of the stress field at the contact surface (a genuinely new integral, not a reuse of the existing penalty-force line) plus choosing/validating a stability-bounded `γ`; the payoff is exact consistency, not activation-robustness (§3.3) | Still needs SDI-class handling at the switch (§3.3) — same conclusion as the penalty-regularized row |
| Macklin et al. NCP-direct + iterative solver (§2.3) | No (displacement-only, if the multiplier is folded into a projected residual the way that paper does), but **requires replacing the direct PARDISO solve with an iterative Krylov method + custom preconditioner** | Very large — a different linear-algebra backend, explicitly ruled out by the task's own stated constraints | Not evaluated — infrastructure mismatch makes this moot |

The **penalty-regularized Alart-Curnier / PDASS row is the standout**
because it is the only Phase-4+ candidate that (a) stays displacement-only
by a real, sourced mathematical mechanism (local per-node condensation,
not an approximation), (b) is not a wholesale reformulation but a targeted
promotion of machinery this codebase **already has** (the exact
Alart-Curnier projection is already coded in `assemble()`'s augmented-
Lagrangian branch; `get_active_set()` already recomputes the active set
every iteration; SDI already exists to protect exactly the iteration where
that set changes), and (c) directly extends to genuine deformable-vs-
deformable contact via the mortar/segment-to-segment generalization of the
already-built, already-proven `_project_point_to_quad4` node-to-face
projection — the same projection machinery the in-progress work is already
reusing from `surface_tie3d.py`.

---

## 5. Recommendation for the deformable-vs-deformable contact currently being implemented

**Build (c): the projection-based penalty/nonlinear-penalty approach now,
as the working baseline, with the penalty-regularized Alart-Curnier/PDASS
promotion documented here as the concrete, well-scoped follow-on — not
semi-smooth Newton or Nitsche adopted wholesale from the start.**

Reasoning, weighing the same engineering-culture bar the task brief itself
invoked ("cheapest fix that's actually targeted at the measured failure"
beating "elegant formulation adopted speculatively"):

1. **Nothing in this literature survey found evidence that deformable-vs-
   deformable specifically breaks the projection-based approach** in a way
   rigid-plane contact didn't (the task's own explicit bar for overriding
   the "keep it simple" default, per its question #4). The complementarity
   switch is structural regardless of formulation (§3.3) — deformable-vs-
   deformable adds genuine new *engineering* complexity (both sides' own
   tangent stiffness coupling into the contact stiffness block, master/
   secondary role assignment for a moving second surface), but that
   complexity is orthogonal to which NCP-treatment is used, and the
   already-designed nonlinear-penalty fix (§1, and the prior design doc's
   §A.2) is exactly targeted at it.
2. **The single biggest reason semi-smooth Newton / Nitsche were flagged
   as worth researching at all — activation-event robustness — is not
   actually what either technique's own literature claims to fix** (§3.3,
   §4.1). Adopting either now would be paying real implementation cost
   (§4.4's "moderate" to "large" rows) for a property (consistency, or
   parameter-range robustness) that is real and valuable, but is not the
   property this project's own measured failure (Hertz divergence at
   activation) was actually about — that failure was fixed by nonlinear
   penalty, a much cheaper change, already measured to work.
3. **The follow-on path costs less than it looks like it should**,
   because this codebase's augmented-Lagrangian branch already contains
   the exact Alart-Curnier projection function (§1) — the PDASS upgrade is
   "stop freezing `λ` across the whole outer Uzawa cycle and instead let
   the existing per-iteration active-set computation drive `λ`'s local
   elimination inside the same Newton system," not "implement a new
   contact mechanics formulation from a blank page." This should be
   scoped as its own follow-on task once (a) is built, generalized to
   moving surfaces, and re-verified with `sanity_report()`-class physical
   checks (per `AGENTS.md` §4.9's standing rule that Newton convergence is
   not proof of physical correctness) — not before.
4. **Do build (a) with the mortar/segment-to-segment generalization in
   mind from the start**, even though PDASS itself is deferred: reusing
   `_project_point_to_quad4` for a moving master surface (rather than a
   fixed analytical plane) is exactly the geometric machinery §2.2's
   dual-mortar papers also need, so no work done now is wasted if PDASS is
   picked up later.

**Do not adopt Nitsche now.** Its actual, sourced advantage (consistency —
no systematic penetration bias at finite stiffness, §3.2) is real, but
this project has no current measured evidence that penalty-law
inconsistency (as opposed to the already-fixed activation-divergence
problem) is limiting accuracy on any live benchmark — and building it
means a genuinely new consistency-term integral, not a reuse of existing
force/stiffness assembly, which is a larger and less-targeted change than
the PDASS follow-on.

---

## 6. Honest gaps in this research pass

1. Popp/Gitterle/Wohlmuth/Wall's specific numerical convergence-rate and
   accuracy figures (§2.2) were not independently re-verified — the
   papers' full text was paywalled; only their stated formulation
   properties (dual shape functions ⇒ local condensation; one unified
   Newton iteration) were confirmed, from search-result summaries of the
   papers' own abstracts/publisher listings, not from reading the full PDF.
2. Y. Renard's 2016 Nitsche-overview preprint 404'd on direct fetch this
   session — not independently re-verified beyond corroboration from the
   sibling Chouly/Hild/Renard papers' own search summaries.
3. The Fischer-Burmeister vs. Alart-Curnier comparative-study paper
   (§4.3) is cited but not read — its actual verdict is an open question,
   not resolved by this document.
4. No source found makes a *direct, load-bearing* numerical comparison
   between Nitsche's method and PDASS/dual-mortar specifically on an
   activation-robustness metric (both are compared here against this
   project's own SDI-based baseline understanding, not against each
   other on a shared benchmark from the literature) — if this distinction
   ever becomes decision-relevant, that comparison would need a dedicated
   search pass, not an inference from this document's sources.
