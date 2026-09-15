# Precise design pass: Dual Mortar shape functions for 3D contact (2026-09-15)

**Status: design only. No `dispsolver/` code changed.** Reviewed against
`dev_log/contact_pdass_precise_design_20260915.md` (this session's PDASS
design, read in full) and `dev_log/contact_beyond_abaqus_research_20260915.md`
(read in full). Grounded in the actual code: `_quad4_shape_functions`,
`_quad4_normal`, `_project_point_to_quad4` (`dispsolver/constraint3d/
surface_tie3d.py`, `surface_contact3d_deformable.py`), read directly this
session; formulas below use this codebase's exact conventions
(`xi`, `eta`, `N1..N4`, `q1..q4`).

---

## 0. Sources

Wohlmuth (2000, SIAM J. Numer. Anal., "A mortar finite element method
using dual spaces for the Lagrange multiplier") and Popp, Gee & Wall
(2010, IJNME) full texts were **not accessible this session** (paywalled,
consistent with the prior research pass's finding). What follows is a
from-scratch re-derivation of the bilinear-quad dual basis from the
well-established, textbook-level structure of dual Lagrange multiplier
spaces (biorthogonality + partition of unity), cross-checked against the
one closed-form result that is public/reproducible in secondary
literature (the 1D two-node dual basis `ψ1=(1-3ξ)/2, ψ2=(1+3ξ)/2`) rather
than copied from an inaccessible source. This is an honest gap, stated
per this project's own citation discipline.

---

## 1. Exact dual basis for Quad4

### 1.1 The generic construction (any element, any basis)

Let `M_ij = ∫_face N_i N_j dA` (the face's own primal "mass matrix",
using the SAME measure `dA` — the true, geometry-dependent physical area
element, not a fixed reference-parametric one). Seek `Ψ_i = Σ_j A_ij N_j`
(dual basis is always a linear recombination of the primal basis on the
same element) satisfying biorthogonality `∫ Ψ_i N_j dA = D_i δ_ij`
(D diagonal) **and** partition of unity `Σ_i Ψ_i ≡ 1` (required so a
constant traction field is representable and so rigid-body/patch
consistency of the mortar coupling holds).

Biorthogonality: `A M = D` ⟹ `A = D M^{-1}`.
Partition of unity: `1ᵗ A = 1ᵗ` (since `Σ_i N_i ≡ 1` already) ⟹
`1ᵗ D M^{-1} = 1ᵗ` ⟹ `dᵗ = 1ᵗ M` (d = diag(D) as a vector) ⟹, since M is
symmetric, `d_i = Σ_j M_ij = ∫ N_i (Σ_j N_j) dA = ∫ N_i dA`.

**Result, true for any face shape, any basis:**

    D_i = ∫_face N_i dA         (diagonal entries — the "lumped mass" of node i)
    A   = diag(D) · M⁻¹
    Ψ_i = Σ_j A_ij N_j

This is **exact and diagonal by construction for every element**,
including a warped/non-planar/trapezoidal Quad4 — diagonality is pure
linear algebra (any `A = D M⁻¹` with `D` diagonal is biorthogonal to `M`
by definition), it does **not** require the element to be regular. What
*does* depend on element shape is whether the `D_i` are all **equal** to
each other, and whether `Ψ_i` has a simple closed form independent of
geometry.

### 1.2 Are the D_i equal? Only for a parallelogram (constant Jacobian)

For a general Quad4, `dA = |J(ξ,η)| dξ dη` and `|J|` is a genuine
function of `(ξ,η)` unless the face is a **planar parallelogram**
(`q2-q1 = q3-q4`, i.e. `t_ξ`, `t_η` from `_quad4_normal` constant), in
which case `|J|` is constant = `A_e/4` (`A_e` = face area). Then, since
each 1D/bilinear shape function integrates to `1` over the reference
square in `(ξ,η)` (shown below), `D_i = ∫ N_i |J| dξdη = |J| = A_e/4` for
**all four** `i` — the diagonal entries are equal, and the closed form
below (§1.3) is *exact*.

For a general (trapezoidal, warped, or non-planar) Quad4, `D_i` differ
from node to node (nodes bounding a locally denser/larger sub-area of
the face carry more "mass"), and the exact `Ψ_i` requires computing the
actual `M` (2×2 Gauss quadrature of `N_i N_j |J(ξ,η)|`, since `|J|` is no
longer constant, the integrand is no longer a simple polynomial in the
tensor-product sense) and inverting the resulting 4×4 `M` (or solving
`AM = D` directly) **per element**. This is a small, cheap, one-time
(per contact pair, per configuration) 4×4 linear solve — not expensive —
but it is a real per-element computation, not a universal formula.

### 1.3 Closed form (exact for parallelograms, first-order-accurate approximation otherwise)

Bilinear `N_i(ξ,η)` factor as `N_i = l_a(ξ) l_b(η)` with 1D linear
shape functions `l_-(ξ)=(1-ξ)/2, l_+(ξ)=(1+ξ)/2`. The 1D dual basis
(standard, reproduces the known literature result): with 1D
`M = [[2/3,1/3],[1/3,2/3]]` (reference-measure `∫_{-1}^{1}`), row sums
`d = [1,1]`, `A = D M⁻¹ = M⁻¹ = [[2,-1],[-1,2]]`, giving

    ψ_-(ξ) = 2 l_-(ξ) - l_+(ξ) = (1 - 3ξ)/2
    ψ_+(ξ) = -l_-(ξ) + 2 l_+(ξ) = (1 + 3ξ)/2

When `|J|` is constant (parallelogram, §1.2), the 2D mass matrix is the
Kronecker product of two 1D mass matrices scaled by `|J|`, and the dual
basis is **exactly** the tensor product of the two 1D duals:

    Ψ1(xi, eta) = 0.25 * (1 - 3*xi) * (1 - 3*eta)
    Ψ2(xi, eta) = 0.25 * (1 + 3*xi) * (1 - 3*eta)
    Ψ3(xi, eta) = 0.25 * (1 + 3*xi) * (1 + 3*eta)
    Ψ4(xi, eta) = 0.25 * (1 - 3*xi) * (1 + 3*eta)

(numbering matches `_quad4_shape_functions`'s `N1..N4`, `q1..q4`
ordering exactly — bottom-left/right, top-right/left). Sanity checks:
`Σ Ψ_i ≡ 1` (partition of unity, verify termwise: coefficients of
`xi*eta`, `xi`, `eta` cancel, constant sums to 1); `∫∫ Ψ_i N_j dξdη = δ_ij`
(verified by the 1D factorization, since the 2D integral of a product of
two separable functions factors into the product of two 1D integrals,
each already `= δ` by the 1D dual-basis construction above).

**This closed form is exact only for a parallelogram face** (planar,
opposite edges parallel/equal — includes squares/rectangles). For a
general trapezoidal or warped/non-planar Quad4 (the general case for a
finite-strain-deformed contact face — exactly what `_quad4_normal`
already has to handle, since it does NOT assume constant `|J|`), this
formula is only a **first-order-accurate approximation**: it is exactly
biorthogonal with respect to the *reference-parametric* measure
`∫dξdη` (no Jacobian) but only approximately biorthogonal with respect
to the *true physical* measure `∫|J|dξdη` once `|J|` varies over the
face — the discrepancy is `O(distortion)`, vanishing as the face
approaches a parallelogram, but not exactly zero for an arbitrarily
warped face. **If exact diagonality is required for a distorted face,
use §1.2's per-element `M`/4×4-solve construction, not this formula.**

---

## 2. Does the diagonal-D benefit require full segment-to-segment integration? — the central practical question

**Short, precise answer: yes, in the sense that matters. The diagonal-D
property is a property of an INTEGRATED (weak, Galerkin-tested) mortar
constraint — a slave-side Lagrange-multiplier FIELD `λ_h(ξ,η) =
Σ_i λ_i Ψ_i(ξ,η)` tested against the gap over the interface. The current
`DeformableSurfaceContactConstraint3D` (and `SurfaceTieConstraint3D`) do
not build any such field or any such integral at all — they are
node-to-segment (NTS) point-collocation schemes: one scalar multiplier
`λ_i` (`self._lam[s_nid]` in the PDASS design) is evaluated and enforced
by a pointwise complementarity condition `C_i(u,λ)=0` at slave node `i`'s
own projected point, with no test-function integration. There is
literally no dense coupling matrix `D`/`M_seg` being formed in that
scheme for dual shape functions to diagonalize — you cannot condense a
matrix that was never assembled.**

Concretely, working through what "layering `Ψ_i` reweighting onto the
existing 5-node stencil" would mean:

- Current force distribution: slave node's penalty force `f_mag·n` is
  applied to the slave DOF with weight `1`, and reacted onto the four
  master vertices with weights `[-N1,-N2,-N3,-N4]` (Newton's-third-law
  consistent lumping of a *single point load* at `(ξ,η)`, exactly
  standard node-to-segment force transfer — this is a load-distribution
  choice, not an integral).
- Swapping `N_i → Ψ_i` in that stencil would NOT diagonalize anything,
  because there is no `D` matrix in this formulation to begin with — it
  would simply change which (now possibly *negative*, since `Ψ_i` is not
  ≥0 the way `N_i` is — e.g. `Ψ1(0.9,0.9) = 0.25(1-2.7)(1-2.7) ≈ 0.72`
  but `Ψ1(-0.9,-0.9) ≈ 0.25(3.7)(3.7) ≈ 3.42`, and `Ψ1` goes negative for
  `ξ>1/3` or `η>1/3`) fractions of the point load get distributed to
  each master vertex, with **no compensating benefit** — it would be an
  ad hoc, non-standard, and arguably *worse* (weights can exceed 1 or go
  negative, unlike partition-of-unity-bounded `N_i` in `[0,1]`) load
  redistribution rule, not "dual mortar, node-to-segment variant." There
  is no meaningful degenerate form of dual-mortar condensation that
  attaches to a single-point collocation scheme; the entire mechanism
  (test against a multiplier basis function integrated over an area)
  requires the area integral to exist in the first place.

**(2a) answered directly: no, the diagonal-D property cannot be realized
in any useful form on top of the existing node-to-single-face pairing
as currently coded. It requires promoting the constraint from
point-collocation to a genuinely integrated (segment) mortar coupling —
a slave element's gap field tested against `Ψ_i` over an actual area,
producing a real `D` (slave self-mass) and `M_seg` (slave-master
cross-mass) that the diagonal `D` then lets you invert for free.**

**(2b) — how much new machinery, precisely, and where it can be
minimized:** the general mortar case (non-matching slave/master
discretization, or master coverage split across multiple faces) needs:

1. **Neighbor search / overlap detection**: for each slave face, find
   all master faces whose projection overlaps it (not just the single
   nearest one `_project_point_to_quad4`'s frozen-pairing scheme picks).
2. **Polygon clipping**: intersect the slave face's projected image
   with each candidate master face in a common (e.g. slave-face
   parametric) plane — Sutherland-Hodgman or a similar convex-polygon
   clipper; each slave-master overlap pair yields a (possibly
   non-quadrilateral, e.g. pentagon) clip polygon.
3. **Sub-triangulation + quadrature**: each clip polygon must be
   triangulated (fan triangulation from its centroid is standard) and
   integrated with its own Gauss rule per sub-triangle, since a single
   `2×2` rule on the *original* face quadrature no longer coincides with
   the true integration domain once slave and master don't line up.
4. Assembling `D` (slave self-mass, still block-diagonal/exactly
   diagonal by §1's construction — this is real and this part of the
   benefit is genuine) and `M_seg` (a real, generally non-square,
   non-diagonal slave-master cross-coupling matrix spanning however many
   master nodes the overlap touches) and forming the actual mortar
   projection operator `D⁻¹ M_seg` used to map/condense the multiplier.

**This is a real, nontrivial computational-geometry subsystem** — not a
formula change — and it is honest to say so rather than downplay it, per
the task's own instruction. It is comparable in scope to (roughly) the
gap between the existing `_project_point_to_quad4` (a Newton-iterated
scalar/2-parameter root-find, already built) and a full 3D contact
search-and-clip library.

**However — the important qualification that changes the staging
recommendation (§5): full polygon clipping is only needed for
NON-MATCHING or SLIDING interfaces.** For a **conforming** mesh pair
(matching node/face topology across the interface, no or negligible
relative sliding — exactly the two-stacked-cubes test already built and
verified this session) the "segment" degenerates to the whole face
itself: each slave face's projection coincides exactly with one master
face, no clipping polygon is needed, and the "mortar" cross-mass matrix
`M_seg` is *exactly* the same-shape 4×4 (or `N_slave × N_master`, both
4 here) matrix `_quad4_shape_functions` already computes pointwise for
the tie/contact projection — just now integrated over the face (2×2
Gauss on the existing quadrilateral) instead of evaluated at one
projected point. **This conforming-mesh special case is a real, much
smaller step that genuinely exercises §1's diagonal-D construction
without needing any new geometric machinery (§3 below gives the staged
plan built on this observation).**

---

## 3. Interaction with PDASS (this session's own prior design)

Read `dev_log/contact_pdass_precise_design_20260915.md` §1-3 in full
before this section (its equations (1)/(2), the `f_mag`/`k_diag`
formulas, and its own frozen-normal orthogonality argument in its §3 are
referenced directly below).

**Structurally similar, but NOT a drop-in "same formula, different gap"
swap — three distinct effects, checked one at a time (mirroring that
document's own discipline of checking orthogonality rather than
assuming it):**

1. **What `λ_i` represents changes.** In PDASS today, `λ_i =
   self._lam[s_nid]` is a genuine point value of contact pressure/reaction
   at slave node `i`'s own projected point — the NCP function `C_i(u,λ)`
   (PDASS doc eq. 1) is evaluated pointwise. Under a true mortar
   discretization, `λ_i` becomes a **coefficient** in the FE expansion
   `λ_h(ξ,η) = Σ_i λ_i Ψ_i(ξ,η)` approximating a continuous traction
   field — it is still one scalar unknown per slave node (the dual basis
   is nodal, `Ψ_i` associated 1:1 with node `i`, exactly like `N_i`), so
   the *bookkeeping* (one `self._lam[nid]` dict entry per slave node)
   is unchanged, but its physical meaning shifts from "the pressure at
   this point" to "the weight of this node's basis function in the
   traction field," a different (though closely related, since `Ψ_i≈N_i`
   in an averaged sense) quantity.
2. **The gap fed into the NCP function must become the weak/mortar gap,
   not the point gap.** PDASS's `f_mag_i = λ_i + c·penetration_i(u)`
   (eq. in its §1.3) needs `penetration_i(u)` replaced by a
   `D`-normalized weak gap `g̃_i(u) = (1/D_i) ∫ Ψ_i · penetration_h(u) dA`
   — i.e. `g̃ = D⁻¹ M_seg · (slave/master point gaps)`, not a single
   point evaluation. Because `D` is diagonal (§1), this normalization is
   still a **cheap, explicit, per-node scalar division** — no linear
   solve needed at this step, which is exactly the computational payoff
   dual bases are for. **This part composes cleanly**: PDASS's per-node
   active-set/complementarity structure (same case-split, same
   `max(0,·)` NCP function, same one-multiplier-per-slave-node
   bookkeeping) carries over with `g̃_i` substituted for `penetration_i`,
   *provided* `g̃_i` is computed from a real mortar integral (§2) — it
   does **not** carry over if `g̃_i` is faked as a point evaluation
   weighted by `Ψ_i(ξ_proj,η_proj)` (§2's rejected "just reweight the
   stencil" idea), because that quantity is not actually `D`-normalized
   against anything and the diagonal-condensation argument does not
   apply to it.
3. **The force/stiffness ASSEMBLY (not the active-set logic) changes
   substantially.** PDASS's current stencil reacts onto exactly the 4
   vertices of ONE projected master face (`[1,-N1,-N2,-N3,-N4]`, PDASS
   doc §3 point 3 / this file's `k_ab` pattern). A genuine mortar
   coupling reacts a slave node's multiplier onto **every master node
   whose face overlaps that slave node's dual-function support** — for
   the conforming-mesh degenerate case (§2) this is still just the same
   4 master vertices (no growth), but for the general non-matching case
   it can span more nodes across several master faces, and the
   stiffness contribution is no longer a simple rank structure built
   from one `(1,-N1,-N2,-N3,-N4)` vector — it is a `D⁻¹ M_seg`-weighted
   sum over all touched master DOFs. **This is the piece that is a real
   implementation increase, not orthogonal to PDASS's own frozen-normal
   simplification but genuinely a bigger rewrite of the assembly loop
   than PDASS's own "the residual/tangent formula in `assemble()`'s
   active branch does not change" headline finding — dual mortar's
   `assemble()` DOES change, structurally, in exactly the place PDASS's
   didn't.**

**Frozen-normal orthogonality (mirroring PDASS §3's own check):** PDASS's
frozen (undifferentiated) normal is a *linearization-only* device — it
doesn't change what physical quantity the residual represents, only how
its Jacobian is approximated. Dual mortar's diagonal-`D` condensation
concerns a *spatial discretization* choice (a different axis) — a mortar
scheme can equally use a frozen or exact normal in its tangent, so the
two are orthogonal **in principle**. One real coupling exists: mortar
normal fields are conventionally NODE-AVERAGED (a weighted average of
the normals of all faces touching a slave/master node, itself often
built via the SAME shape-function weights) rather than the single
per-projected-face normal `_quad4_normal` returns today — so "freeze the
normal" under mortar would mean freezing this averaged nodal field, a
modest but real change to *what* gets frozen, not a change to *whether*
freezing composes with dual bases.

---

## 4. Honest assessment: present need or speculative infrastructure?

**This is forward-looking infrastructure, not a fix for any currently
measured problem, and that should be stated plainly rather than
glossed over.** The only existing test for
`DeformableSurfaceContactConstraint3D` (per the task brief: two stacked
deformable cubes, node-for-node conforming, no sliding) has:

- No non-matching mesh resolution across the interface.
- No large sliding (the class's own module docstring restricts it to
  small-sliding, frozen pairing — `freeze_projection=True`-style).
- No measured accuracy gap, no measured excess iteration count, no
  measured conditioning problem attributable to the current NTS scheme
  on that test — unlike nonlinear penalty (motivated by a **measured**
  Hertz-divergence) or PDASS (motivated by a **measured** outer-loop
  iteration count, per that design doc's own Test 2). There is no
  analogous number to point to here.

Dual mortar's actual, real-world payoff (diagonal-D condensation making
sliding contact between non-matching meshes as cheap as a matching one)
is a property that only matters once the codebase actually needs
non-matching-mesh or large-sliding deformable-vs-deformable contact —
neither of which is a currently built or currently tested capability.
**Building the general (§2's polygon-clipping) machinery now would be
exactly the kind of "designing for hypothetical future requirements"
this project's own engineering culture (and `AGENTS.md`'s standing
"don't add abstractions beyond what the task requires" instruction)
argues against.**

The one piece that IS cheap and immediately verifiable without
speculative scope (§2's conforming-mesh degenerate mortar, §5 Stage 1) is
defensible as a **verification/characterization exercise** — it
concretely demonstrates the diagonal-D property on hardware that already
exists (the two-cubes test) — but even that should be understood as
laying groundwork/building confidence in the formulation, not as fixing
a defect in the current NTS scheme, which is not shown to have one on
its own test.

---

## 5. Staged implementation plan (if pursued)

**Stage 0 (no new code, already true):** document explicitly (this file)
that the current NTS schemes have no dense coupling matrix and therefore
no dual-mortar condensation opportunity exists to "add" to them without
first promoting to a real mortar (integral) formulation.

**Stage 1 — conforming-mesh mortar demonstrator (smallest real,
non-speculative step):**
- Scope: matching slave/master face topology only (exactly today's
  two-stacked-cubes fixture) — no polygon clipping, no new neighbor
  search.
- Build: per contact-pair face, compute the real `M_ij = ∫∫ N_i N_j |J|
  dξdη` via 2×2 Gauss (already have `_quad4_shape_functions`/`_quad4_normal`
  machinery to build `|J|` from `t_xi × t_eta`'s norm), `D_i = Σ_j M_ij`,
  `A = diag(D) M⁻¹`, `Ψ_i = Σ A_ij N_j` (§1's exact-for-parallelogram
  closed form as a fast path when the face is checked-parallelogram,
  full per-element solve otherwise).
- Replace the point NCP/gap evaluation with the weak `g̃_i` (§3 item 2),
  keep PDASS's per-node active-set/multiplier update structure otherwise
  unchanged.
- **Verification (concrete, falsifiable, not assumed):** re-run the
  existing two-stacked-cubes test; print the assembled `M`/`D` matrices
  and confirm `D` is numerically diagonal (off-diagonal terms ~ machine
  epsilon relative to diagonal, since the faces are conforming/close to
  parallelogram in that fixture) — this is the "measured, not assumed"
  check the task brief explicitly asks for; compare converged
  `max_penetration`/contact-force distribution against the current NTS
  point-scheme's result on the same fixture to quantify any difference
  (expected small, since both are consistent discretizations of the
  same continuous problem on a conforming mesh); re-run
  `sanity_report()`/region-tracking checks (`AGENTS.md` §4.9) to confirm
  Newton convergence isn't masking a coupling defect.

**Stage 2 — general (non-matching mesh / large sliding) segment-to-segment
mortar — only if a real future task needs non-matching contact
interfaces:**
- Scope: full polygon clipping (§2 item 2), sub-triangulated quadrature
  (§2 item 3), extended neighbor search to find all overlapping master
  faces per slave face (not just nearest-one).
- Flagged explicitly as substantial new computational-geometry
  machinery, not a small increment on Stage 1 — budget it as its own
  design-then-review pass (its own `dev_log` precise-design document,
  matching this session's own process), not as a follow-on line item.
- Verification: construct a genuinely non-matching two-block test
  (different mesh resolution across the interface, some relative
  sliding) and confirm force/energy balance (Newton's-third-law
  slave/master reaction sums) and the diagonal-`D` property still hold
  once clipping is involved (the clipped sub-face masses still assemble
  into a diagonal slave self-mass by the same §1 algebra, since
  diagonality only depends on `A = D M⁻¹`, not on how `M` was
  integrated).

---

## 6. Summary answers (for the requester)

1. **Exact dual basis**: `Ψ_i = Σ_j A_ij N_j`, `A = diag(D) M⁻¹`,
   `D_i = ∫_face N_i dA` — always diagonal by construction; `D_i` are
   equal to each other only for a parallelogram face, where the closed
   form `Ψ1..Ψ4 = 0.25(1∓3ξ)(1∓3η)` is exact. For a general warped Quad4,
   `D_i` differ and the exact `Ψ_i` needs a per-element 4×4 solve.
2. **Full segment integration required?** Yes, for the diagonal-D benefit
   to mean anything — it is a property of an integrated mortar
   constraint that the current node-to-single-face collocation scheme
   does not build. But full polygon clipping is only needed for
   non-matching/sliding interfaces; a conforming-mesh (matching
   topology) mortar is a real, much smaller degenerate case using the
   existing per-face quadrature machinery.
3. **Present need or speculative?** Speculative/forward-looking — no
   currently measured defect on the existing (conforming, no-sliding)
   test motivates this, unlike nonlinear penalty (Hertz) or PDASS
   (iteration count). Honest recommendation: do not build the general
   (Stage 2) machinery now; Stage 1's conforming-mesh demonstrator is
   defensible only as characterization/groundwork, not a bug fix.
