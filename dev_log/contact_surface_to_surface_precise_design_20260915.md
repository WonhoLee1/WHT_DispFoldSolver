# Precise design pass: true surface-to-surface (segment-to-segment) 3D contact discretization (2026-09-15)

**Status: design only. No `dispsolver/` code changed.** This is an explicit
feature request, not speculative infrastructure ("우리 surface to surface
지원해야한다" -- confirmed by the requester, overriding
`contact_dual_mortar_precise_design_20260915.md`'s earlier "no currently
measured defect motivates this" recommendation for the *general* case).
Read in full and built on directly, not restated: `contact_dual_mortar_
precise_design_20260915.md` (dual Quad4 basis derivation, §1; its own
Stage 1/Stage 2 split, §5), `contact_pdass_precise_design_20260915.md`
(single-loop augmented Lagrangian, §1-3), and
`benchmark_element/reference_abaqus_docs/ctc_contactpairform_std.txt`
(full text, read this session -- quoted below by section title, not by
line number, since the source file has no numbered sections).
`dispsolver/constraint3d/surface_contact3d_deformable.py` (full file, 295
lines) and the relevant slice of `dispsolver/solver3d/dynamic3d.py`
(`_contact_active_set` L573-587, `solve_step` L589-825) were read in full
this session; every variable name below (`gap`, `penetration`, `lam`,
`k_contact`, `n`, `N1..N4`, `xi`, `eta`, `self.pairs`, `self._lam`) is
that file's own.

---

## 0. What Abaqus's own source actually says, precisely, and what it implies for scope

Quoting `ctc_contactpairform_std.txt` directly (not paraphrased, since the
exact wording bounds what "surface-to-surface" is allowed to mean here):

> "The surface-to-surface formulation enforces contact conditions in an
> average sense over regions nearby secondary nodes rather than only at
> individual secondary nodes... The contact direction is based on an
> average normal of the secondary surface in the region surrounding a
> secondary node."

> "With node-to-surface discretization... each contact condition involves
> a single secondary node and a group of nearby main nodes... Normal
> contact constraints due to surface-to-surface discretization produce
> unsymmetric terms in both two- and three-dimensional cases."

> Table 1: node-to-surface 13% (secondary=top) / 31% (secondary=bottom)
> max CPRESS error vs. analytical 100 Pa uniform pressure; surface-to-
> surface ~1% for either secondary-surface choice, on a two-block,
> dissimilar-mesh-density test.

Two scope-defining consequences, stated up front because they shape every
later section:

1. **The averaging is over a region "nearby" (predominantly centered on)
   one secondary node, not the whole secondary surface.** This is exactly
   what the dual-basis Ψ_i's local support gives (§2 below) -- it is
   *not* the same thing as a single global smoothing pass over CPRESS.
2. **The doc explicitly names the accuracy driver as "dissimilar mesh
   refinement"** (Figure 4/Table 1's own setup). This means Stage 1 of
   the predecessor dual-mortar doc (conforming, matching-topology,
   single-segment-per-face) is necessary infrastructure but is **not
   sufficient to demonstrate the claimed benefit** -- see §4's
   falsifiable prediction, which requires the general clipping machinery
   of §1 below to even be tested.

---

## 1. Broad phase, clipping, and quadrature -- exact algorithm

### 1.1 Broad phase (candidate overlap search)

For each slave face (4 deformed node coordinates), build an axis-aligned
bounding box padded by the existing `position_tolerance` constructor
parameter (reused: today it bounds a point-to-face *distance*; here it
bounds an AABB *inflation*, same physical role -- "how far apart can two
things be and still be considered candidates"). Test against every master
face's own padded AABB via 6 interval comparisons
(`lo_a[d] <= hi_b[d] and lo_b[d] <= hi_a[d]` for `d` in x,y,z). This is
the direct AABB generalization of `_build_pairs`'s existing exhaustive
`for s_nid ... for m_face ...` double loop (`surface_contact3d_deformable.py`
L134-156) -- same O(n_slave_faces * n_master_faces) asymptotic cost class,
just keeping **all** AABB-overlapping candidates instead of only the
nearest one. No spatial hash/BVH is proposed now: the existing code
already runs this shape of double loop without complaint at this
project's mesh sizes; flag a spatial-hash upgrade as a later optimization
only if a benchmark shows the O(n*m) broad phase itself as the bottleneck,
not before.

### 1.2 Common-plane projection

For each (slave face, candidate master face) surviving the broad phase,
project onto the **slave face's own local tangent plane at its centroid**
(ξ=η=0) -- this is this codebase's existing convention (the slave side
already owns the frozen normal/pairing decision in `_build_pairs`, and the
dual basis Ψ is defined on the slave face, §2), reusing `_quad4_normal`
verbatim for the plane normal:

```python
n_plane = _quad4_normal(qs1, qs2, qs3, qs4, 0.0, 0.0)   # slave face centroid normal
centroid = 0.25 * (qs1 + qs2 + qs3 + qs4)
e1 = t_xi / np.linalg.norm(t_xi)          # t_xi at (0,0), already computed inside _quad4_normal
e2 = np.cross(n_plane, e1)                # already unit, since n_plane and e1 are unit and orthogonal
```

Project any 3D vertex `v` (slave or master) to 2D plane coordinates:

```python
v_rel = v - centroid
v2d = np.array([np.dot(v_rel, e1), np.dot(v_rel, e2)])   # discards the out-of-plane component (v_rel . n_plane)
```

**Non-planarity, addressed precisely, not hand-waved.** A first-order
hex-derived Quad4 face is planar only if undeformed and undistorted; under
finite strain it is generally warped. Discarding the out-of-plane
component is an **approximation whose error is second-order in the
warping amplitude relative to the face's own in-plane size** (dropping a
component orthogonal to the two retained ones changes the projected area
by `O((δ/L)^2)`, not `O(δ/L)`, where δ is the out-of-plane deviation and L
the face size) -- this is the standard "small-sliding, near-planar
contact" approximation this class's own module docstring already commits
to (frozen pairing, modified-Newton normal), so accepting a second-order
projection error here is consistent with, not additional to, that existing
scope decision. **This is Stage B's default path (§5).** If a future
benchmark shows this error is not negligible for a specific mesh (e.g. a
face wrapped around a tight bend), the exact fallback is: split each
Quad4 into 2 triangles (fixed diagonal split, `q1-q2-q3` / `q1-q3-q4`,
matching this codebase's own triangle-fan convention used elsewhere for
Quad4 post-processing), each triangle now being **exactly planar in 3D by
construction**, and clip triangle-against-triangle (§1.3) in each
triangle's own plane instead of one shared plane -- up to 4 triangle-pair
clips per quad-vs-quad candidate instead of 1 polygon clip. This is a
real, flaggable follow-on (not built now), because it changes the
data-structure fan-out (§2.3) from "1 clip result per candidate pair" to
"up to 4," and should only be added once a concrete warped-face test case
motivates it -- do not add it speculatively.

### 1.3 Clipping: Sutherland-Hodgman, with its precondition checked, not assumed

Clip the **master** face's projected quadrilateral (subject polygon,
Sutherland-Hodgman's subject may be any simple polygon, convex or not)
against the **slave** face's projected quadrilateral (clip window --
Sutherland-Hodgman **requires** the window to be convex; this is the one
precondition that must be checked, not assumed). A first-order Quad4 face
projected from a mildly distorted hex is convex in the overwhelming
majority of cases, but large shear distortion can in principle produce a
slightly non-convex quadrilateral. **Precise handling**: compute the
signed area of each of the slave quad's 4 consecutive edge-triples via the
standard cross-product convexity test; if all 4 signs agree, proceed with
standard Sutherland-Hodgman (4 clip edges, one polygon-vs-half-plane pass
per edge, exactly the textbook algorithm -- no modification needed for a
convex window). If the slave quad fails the convexity check, **do not
silently clip anyway** (this would silently produce wrong, unflagged
areas -- the exact class of defect `AGENTS.md` §4.8/§4.16 spent multiple
sessions cataloguing): fall back to the existing point-collocation
node-to-surface treatment for that specific slave face only (log a
one-time warning), never fabricate a clip result against an invalid
window. This fallback is cheap to implement (it is exactly today's
existing `DeformableSurfaceContactConstraint3D` code path, already
present) and keeps the failure mode a graceful accuracy degradation on a
handful of badly distorted faces rather than a silent global error.

Sutherland-Hodgman against a convex quad window produces at most **8
vertices** (each of the subject's 4 edges can be clipped by each of the 4
window edges, and a convex-vs-convex clip of two quadrilaterals is a
classical result bounded by `4+4=8`). Degenerate outputs (0, 1, or 2
vertices -- edge-touching or no overlap) mean no segment for that
candidate pair; discard it.

### 1.4 Sub-triangulation and quadrature order

Fan-triangulate the clipped polygon (≤8 vertices) from its own centroid
into ≤8 triangles -- standard, and already the codebase's implicit
convention for post-processing arbitrary polygons. For each sub-triangle,
integrate with a fixed symmetric Gauss-triangle rule.

**Precise statement of what "exact" can and cannot mean here, since a
false claim of exactness would repeat the class of error `AGENTS.md`
§4.16/§4.18 already corrected once (the `Q4_CR`/`C3D8H` bit-identical
anomaly's lesson: verify, don't assume).** The integrand needed for `M_ij`,
`D_i`, and the weak-gap numerator (§2) is `N_i(ξ_s,η_s) * N_j_or_1(...) *
w`, where `w` is the physical-area quadrature weight. For a **planar**
slave face this reduces, by the exact 2D-bilinear-quad identity this
project already proved in `AGENTS.md` §4.18 (`J̄ = J₀` for any 2D
bilinear quad, `detJ` linear in ξ,η), to a low-degree polynomial in the
*slave face's own* parametric coordinates -- but the quadrature here is
performed in the **clip polygon's** sub-triangle coordinates, and mapping
a sub-triangle's barycentric coordinates back to the slave face's `(ξ,η)`
is **not** an affine map in general (it is the composition of the
triangle's own affine parametrization with the *inverse* of the slave
face's bilinear map, which is a rational/non-polynomial function once the
face is a genuine trapezoid rather than a parallelogram). And for a
**warped** (non-planar) face, the physical area element itself is
`|t_ξ × t_η|`, a norm (square root of a quadratic form) -- not a
polynomial in `(ξ,η)` at all, regardless of parametrization. **Exact
polynomial quadrature is therefore unattainable in the general case**, and
no fixed order should be described as "exact" for this integral outside
the parallelogram special case. The precise, honest choice: use a fixed
6-point, degree-4-exact symmetric triangle rule per sub-triangle as the
practical default (enough headroom above the leading bilinear-times-
bilinear ≈ biquadratic content to make the *polynomial-truncation* part of
the error small), and **verify by refinement, not by assumption**: on
this project's own conforming-mesh test fixture, re-run with a
higher-order (e.g. 12-point, degree-6) rule and confirm `M_ij`/`D_i`/the
assembled weak gap change by less than a stated tolerance (propose
`<0.1%`, matching this project's own established "measured, not assumed"
discipline) before trusting the 6-point default in the general
(non-parallelogram) case.

---

## 2. Mortar matrices, weak gap, and the `assemble()` residual/tangent -- exact formulas, in this codebase's variables

### 2.1 The dual basis, reused exactly, computed once per slave face at construction

Per slave face `f` (4 nodes), compute the **full-face** self-mass and row
sums once, at construction, on the **reference** (undeformed) coordinates
-- consistent with this class's existing frozen-pairing convention
(`xi, eta, sign` are already frozen per pair at construction; extending
that same freeze to the face's own `M`/`D`/`Ψ` is not a new scope
decision, it is the same one applied to a new quantity):

```python
M_f = 2x2_gauss( N_i(xi,eta) * N_j(xi,eta) * |t_xi x t_eta|(xi,eta) )   # 4x4, over the WHOLE face, no clipping
D_f = M_f.sum(axis=1)                          # (4,) -- dev_log/contact_dual_mortar_precise_design_20260915.md sec1.1's d_i = sum_j M_ij
A_f = np.diag(D_f) @ np.linalg.inv(M_f)         # (4,4)
# Psi_f(xi, eta) = A_f @ N(xi, eta)             # dual basis values, 4 numbers, at any (xi,eta) on face f
```

This is a **4x4 solve per slave face, once**, exactly the "small, cheap,
one-time... 4x4 linear solve" the dual-mortar doc's §1.2 already
characterized -- not a new cost class.

### 2.2 Global aggregation of `D` -- the answer to "does a shared slave node need one value or many"

A slave face's 4 local nodes are, in general, **shared with other slave
faces** (any interior slave-surface node belongs to 4 quad faces in a
regular mesh). `D_f` above is **face-local** by construction (§2.1 never
looked at neighboring faces). The precise, standard-FEM-assembly answer
(same scatter-add pattern already used for internal-force assembly,
nothing new): build one **global** array `D_global`, sized by the number
of distinct slave node ids, by summing every face's local `D_f` entries
into their global node's slot:

```python
D_global[nid] = 0.0  # init once, over ALL slave node ids appearing in ANY slave face
for f in slave_faces:
    D_f = ...   # sec2.1, computed once per face
    for a, nid in enumerate(f.node_ids):
        D_global[nid] += D_f[a]
```

`D_global` is **still diagonal** (it is a per-node scalar, not a matrix --
diagonality was never about the assembly pattern, only about `A = D M^-1`
having no off-diagonal `D` entries, which summing scalars cannot break).
Note explicitly, honestly: the resulting **effective** global dual
function for a shared node (implicitly `D_global[nid]^-1 * sum_f(D_f[a] *
Psi_f)`) is not a single globally-continuous FE-consistent dual basis in
the full Wohlmuth sense (that would require assembling and inverting a
*global*, patch-connected `M`, a materially bigger and more expensive
step than this document is proposing) -- it is a **sum of face-local dual
contributions, normalized once by a globally-assembled `D`**. This is a
deliberate, stated scope decision (matching the predecessor dual-mortar
doc's own explicit "per-face" derivation, §1.1 there), not an oversight:
it keeps the O(1) 4x4-solve-per-face cost structure exactly as that
document already characterized as cheap, while still producing a single,
well-defined scalar per global slave node -- which is exactly what §2.3's
active-set question needs and is sufficient for.

### 2.3 Segment data structure (replaces the fixed 5-tuple `self.pairs`)

```python
# Built once at construction (frozen small-sliding pairing, sec1):
self.slave_faces: List[Tuple[int,int,int,int]]      # NOTE: constructor signature change,
                                                       # see sec2.5 -- slave side becomes
                                                       # face-based, matching master_faces,
                                                       # not node-based as today
self.D_global: Dict[int, float]                      # sec2.2, keyed by global slave node id
self.A_f: Dict[face_idx, np.ndarray]                 # 4x4 per slave face, sec2.1

# Segment = one (slave face, master face) pair with nonzero clip overlap:
Segment = namedtuple("Segment", ["slave_face_idx", "master_face_idx", "quad_points"])
QuadPoint = namedtuple("QuadPoint", ["w", "xi_s", "eta_s", "xi_m", "eta_m", "n"])
# w: scalar quadrature weight, ALREADY including the sub-triangle's own physical-plane
#    area scaling (i.e. w is the final dA weight -- no separate detJ multiply at assembly time)
# xi_s, eta_s: pulled-back slave-face parametric coords of this quad point (small linear/
#    bilinear inversion of a point already known to lie in the projected plane -- NOT a
#    full _project_point_to_quad4 Newton search, since the point's plane membership is
#    already guaranteed by construction from the clip)
# xi_m, eta_m: pulled-back master-face parametric coords, same inversion against the master face
# n: this SEGMENT's own frozen normal (sec2.4) -- evaluated once at construction from the
#    MASTER face's centroid normal, sign-oriented by the same gap0>=0 convention _build_pairs
#    already uses. Per-segment, not per-slave-face, because two different master faces
#    touching one slave face (e.g. across a coarse/fine mesh transition) are not guaranteed
#    to share a normal direction.

self.segments: List[Segment]
```

**A slave face can now appear in multiple `Segment`s** (one per
overlapping master face) -- this is the exact data-structure change the
task named as required, and is precisely why `self.pairs`'s fixed
`(slave_nid, ONE master_face, ...)` 5-tuple cannot be extended in place;
it must become a list indexed by segment, with slave/master face identity
carried per segment rather than assumed unique per slave node.

### 2.4 Weak gap, per global slave node -- two-pass aggregation, mirroring `D_global`

For a given displacement `u` (this is `get_active_set(u)` / `assemble(u)`
/ `update_augmented_multipliers(u, ...)`'s shared first pass, computed
once and reused by all three):

```python
gap_numerator = {nid: 0.0 for nid in D_global}   # reset every call -- NOT cached across calls
for seg in self.segments:
    f = self.slave_faces[seg.slave_face_idx]
    m = self.master_faces[seg.master_face_idx]
    A_f = self.A_f[seg.slave_face_idx]
    for qp in seg.quad_points:
        Ns = _quad4_shape_functions(qp.xi_s, qp.eta_s)          # (N1..N4) of slave face f
        Nm = _quad4_shape_functions(qp.xi_m, qp.eta_m)          # (N1..N4) of master face m
        Psi = A_f @ np.array(Ns)                                # dual basis values at this quad point
        xs = sum(Ns[a] * (coords[f[a]] + u[3*f[a]:3*f[a]+3]) for a in range(4))
        xm = sum(Nm[b] * (coords[m[b]] + u[3*m[b]:3*m[b]+3]) for b in range(4))
        penetration_q = -np.dot(xs - xm, qp.n)                  # SAME sign convention as
                                                                 # _current_gap_and_normal's `gap`
        for a in range(4):
            gap_numerator[f[a]] += Psi[a] * penetration_q * qp.w

g_tilde = {nid: gap_numerator[nid] / D_global[nid] for nid in D_global}   # weak gap, one scalar per slave node
```

`g_tilde[nid]` now plugs directly into the **unmodified** PDASS formulas
(`contact_pdass_precise_design_20260915.md` eq. 1/2), substituting for the
old point `penetration_i(u)`:

```python
# get_active_set(u):  p = self._lam[nid] + self.k_contact * g_tilde[nid];  active if p > 0
# assemble(u), active branch:  f_mag = self._lam[nid] + self.k_contact * g_tilde[nid];  k_diag = self.k_contact
# update_augmented_multipliers(u):  p_target = max(0, lam_old + self.k_contact * g_tilde[nid])
```

identical to today's code, verbatim, with `g_tilde[nid]` where
`(-gap)`/`penetration` appears today.

### 2.5 Residual and tangent -- the real assembly change, derived precisely

Define, once `g_tilde`/`p_i` (`= self._lam[nid] + k_contact * g_tilde[nid]`,
or `self.law.evaluate(g_tilde[nid])[0]` in plain-penalty mode) are known
for every slave node, the **interpolated multiplier field value at a
quadrature point** (this is the same `λ_h(ξ,η) = Σ λ_i Ψ_i(ξ,η)` concept
the dual-mortar doc's §3 item 1 already introduced, now given a concrete
per-quad-point evaluation):

```python
lam_h(qp, f) = sum(p[f[a]] * Psi_a(qp)  for a in range(4) if p[f[a]] > 0.0)   # only active nodes contribute
```

**Force** (virtual-work derivation: slave virtual displacement enters via
its own *primal* `N_a`, not `Ψ_a` -- `Ψ` is the multiplier's test
function, not the slave's displacement interpolation; this is the one
place a naive "just reweight the old 5-node stencil by Ψ" idea, already
rejected by the dual-mortar doc §2 for the point-collocation scheme, would
also be wrong here for a different reason):

```python
# per segment, per quad point, accumulate into GLOBAL f_contact (same sign convention as
# today's f_contact[...] += -w_a * f_mag * n):
for a in range(4):                      # slave face f's own 4 local nodes
    f_contact[3*f[a] : 3*f[a]+3] += -Ns[a] * qp.w * lam_h(qp, f) * qp.n
for b in range(4):                      # master face m's own 4 local nodes
    f_contact[3*m[b] : 3*m[b]+3] += +Nm[b] * qp.w * lam_h(qp, f) * qp.n
```

(sign flip between slave/master is the same Newton's-third-law
push/reaction pattern `assemble()` already documents; `qp.n` is the
segment's own frozen normal, §2.3.)

**Tangent -- a genuine, precisely-derived structural change, not a
formula tweak.** Differentiating `f_contact` above w.r.t. `u` while
holding `Ψ`, `N`, and `n` frozen (modified-Newton, matching this file's
existing established convention) still leaves a real `du`-dependence
through `p_i` itself (`p_i = lam_i + k_contact * g_tilde_i(u)`, active
branch), and `g_tilde_i` depends on the penetration at **every** quad
point across **every** segment touching slave node `i` (§2.4), not just
one. Chaining this through gives, for each **active** slave node `i` on
face `f`:

```python
b_i = zeros(3 * n_touched_dofs)          # touched dofs = f's own 4 slave nodes UNION every
                                          # master node of every segment (f, ·) that has a
                                          # nonzero Psi_i somewhere in its quad points
for seg in segments_of_face(f):
    for qp in seg.quad_points:
        Psi_i_qp = A_f[i, :] @ N(qp.xi_s, qp.eta_s)      # scalar, dual basis value at this qp
        for a in range(4):
            b_i[dof_index(f[a])]        += Psi_i_qp * Ns_a(qp) * qp.w * qp.n     # (3,) each
        for b in range(4):
            b_i[dof_index(seg.master_face[b])] -= Psi_i_qp * Nm_b(qp) * qp.w * qp.n
K_block += (self.k_contact / D_global[i]) * outer(b_i, b_i)
```

i.e. **the tangent contribution of each active multiplier node `i` is a
single rank-1 outer product `(k_contact / D_global[i]) * b_i (x) b_i`**
over that node's own local dof patch (its slave face's 4 nodes, plus every
master node any of that face's segments touch) -- not a dense coupled
solve, and not the old scheme's fixed `k_ab = k_diag * w_a * w_b` computed
independently per pair. **This rank-1-per-active-node structure is the
concrete computational payoff `D_global` being diagonal buys**: each
active node contributes one cheap outer product sized by however many
dofs its own segments touch, never a coupling between two *different*
multiplier nodes' dof patches. This is the precise mechanism the
dual-mortar doc's §3 item 3 flagged as "a real implementation increase...
genuinely a bigger rewrite of the assembly loop" without deriving it to
this level -- it is now fully derived and directly transcribable.

**Consistency check, stated because it is concretely falsifiable and
should be run once implemented**: if a slave face's segments fully and
exactly tile its own domain with no gaps or overlaps (guaranteed true when
master coverage is complete, e.g. the conforming-mesh Stage 1 case, or any
case where the master surface fully covers the slave surface), then
`sum(qp.w over ALL quad points of ALL segments of f) == the face's own
total physical area to quadrature precision`, and `sum_over_segments(
Ψ_i(qp)*N_j(qp)*w summed over that face's own quad points) -> M_f[i,j]`
(§2.1's own full-face self-mass) in the limit of complete coverage. This
is a discrete partition-of-unity / patch-test-style check, directly
analogous to this project's own established "measured, not assumed"
verification discipline (`AGENTS.md` §4.9, §4.16), and should be printed
and checked on the Stage 1 conforming fixture before trusting the general
clipped case.

### 2.6 Constructor signature change, stated explicitly

`DeformableSurfaceContactConstraint3D.__init__`'s `slave_node_ids: List[int]`
parameter must become `slave_faces: List[Tuple[int,int,int,int]]`
(mirroring `master_faces`'s own existing signature) -- the dual basis
`Ψ_i` (§2.1) is defined **per slave face**, and there is no well-defined
"a slave node's own face" in a node list (an interior node belongs to up
to 4 faces with no canonical choice among them). This is a real,
user-facing API change from today's class, not an additive one -- any
existing caller passing a flat node list would need to be updated to pass
the slave surface's own face connectivity instead. Flagging this now so
it isn't discovered as a surprise mid-implementation.

---

## 3. Active-set aggregation -- resolved precisely (not left open)

**Direct answer to the task's own question**: yes, `get_active_set()` can
still return a well-defined, single active/inactive boolean per slave
NODE, and it does **not** need a per-(node, contributing-segment) pair
representation. §2.2's `D_global` and §2.4's two-pass `gap_numerator ->
g_tilde` aggregation together produce **exactly one** `g_tilde[nid]`
scalar per global slave node id, by summing every segment/quad-point
contribution into that node's slot *before* dividing -- the aggregation
happens once, at the `gap_numerator`/`D_global` level, not later. Given
that single `g_tilde[nid]`, `get_active_set()`'s decision is the same
one-line formula PDASS already uses (§2.4's block), and
`_contact_active_set()` in `dynamic3d.py` (L573-587) needs **zero
changes** -- it already just unions `get_active_set()`'s returned
`frozenset` of node ids across constraints, and this class's
`get_active_set()` still returns a `frozenset` of node ids, only now
computed via the two-pass aggregation internally instead of a single
point evaluation. SDI (`is_sdi = current_active_set != prev_active_set`,
L705) is likewise unaffected: it compares two `frozenset`s of node ids by
value, and does not care how each set was computed internally.

The one thing that is **not** well-defined per-node, and should not be
forced to be: the **per-segment** contribution to a node's gap
(`Ψ_i(qp) * penetration_q * qp.w`, summed inside `gap_numerator[f[a]] +=
...`) has no independent physical meaning on its own -- only the fully
summed `g_tilde[nid]` does. This matches the physical intuition stated in
the task brief directly: a node's total contact status is well-defined
even though force/gap information arrives from multiple segments, and the
resolution is a sum-then-decide (not a per-segment vote or an average),
exactly mirroring how ordinary FE nodal quantities (mass, internal force)
are always resolved when multiple elements share a node.

---

## 4. Verification plan -- targeting Abaqus's own measured comparison directly

**Setup** (as close to `ctc_contactpairform_std.txt`'s own Figure 4/Table
1 as this codebase's existing example infrastructure allows): two
stacked, deformable cubes (reusing this session's already-built two-cube
fixture as the geometry/material base), contact interface meshed at
**different densities** on the two blocks (e.g. top block's contact face
2x2 Quad4, bottom block's contact face 4x4 Quad4 -- deliberately
non-matching topology, not just non-matching node positions, since that
is what triggers segment clipping at all). Apply a uniform pressure load
on the top block's far face; bottom block's far face fixed. Analytically,
uniform 100-unit (or this codebase's own consistent unit choice) pressure
should be transmitted uniformly across the interface.

**Comparison**: run the same geometry/mesh/load twice, swapping only the
contact constraint class -- (a) today's `DeformableSurfaceContactConstraint3D`
(node-to-surface, point collocation), (b) this design's segment-to-segment
class. For each, record the per-slave-node contact pressure (`p_i` /
`self._lam` at convergence, the direct analog of CPRESS) across the
interface and compute `max(|p_i - p_uniform_expected|) / p_uniform_expected`.

**Falsifiable prediction, stated precisely as the task requires**: (b)'s
maximum relative CPRESS-analog error on this dissimilar-mesh interface
should be **measurably lower** than (a)'s, and should show visibly less
node-to-node spread (fewer/smaller spikes concentrated at individual
secondary nodes, matching the doc's own "forces tend to concentrate at
these secondary nodes" node-to-surface characterization vs.
surface-to-surface's stated "smoothing effect"). **This codebase's own
absolute error numbers should NOT be claimed to match Abaqus's 13-31% vs.
~1%** -- different mesh, material, and solver internals will give
different absolute numbers; only the **direction and qualitative
relative-improvement** claim is being made. Additionally re-run
`sanity_report()`/`check_region_tracking()`-class checks (`AGENTS.md`
§4.9) to confirm the two bodies remain genuinely coupled through the new
constraint (not just independently converging), and print/verify §2.5's
patch-test-style coverage identity as a build-correctness gate before
trusting the accuracy comparison at all.

**A negative-result sign to watch for, stated honestly**: if (b) shows no
improvement over (a) on this test, the specific, checkable things to
suspect first are (in order of likelihood): the master-coverage identity
in §2.5 failing (segments not actually tiling the slave face, meaning
`D_global`/`g_tilde` are being computed against an incomplete integral) or
the quadrature order being insufficient for the dissimilar-mesh clip
polygons' actual size distribution (§1.4) -- check both before concluding
the formulation itself is at fault.

---

## 5. Staged plan -- honest about what's buildable now vs. deferred

**Stage A (smallest real step, reuses Stage 1 from the dual-mortar doc):**
matching-topology slave/master faces only, single segment per slave face
(no clipping, no AABB broad phase needed -- each slave face pairs with
exactly one master face, whole-face). Purpose: exercise and verify §2's
`D_global` aggregation, the rank-1-per-active-node tangent (§2.5), and the
constructor signature change (§2.6) end-to-end on a fixture where the
segment/quad-point machinery is trivial. **This stage alone cannot
demonstrate the accuracy benefit** (§0's point 2: the benefit is
specifically about dissimilar mesh refinement) -- treat it as a
correctness/regression gate before Stage B, not as the deliverable.

**Stage B (the actual feature, required before §4's benchmark can run at
all):** full AABB broad phase (§1.1) + Sutherland-Hodgman clipping with
the convexity-check fallback (§1.3) + fan-triangulated quadrature with the
refinement-verified order (§1.4), on genuinely non-matching, dissimilar-
density meshes. This is where the real new computational-geometry
machinery lives, matching the dual-mortar doc's own honest sizing of this
work ("comparable in scope to... a full 3D contact search-and-clip
library") -- budget it as such, not as a small increment on Stage A.

**Deferred, explicitly, not built now (flag but do not implement without
a concrete future need, per this project's own standing discipline):**

- The triangle-pair decomposition for genuinely warped (non-planar) Quad4
  faces (§1.2's fallback) -- only needed once a test case with measurable
  warping-induced clipping error actually exists.
- Node-based (non-Quad4) secondary surfaces -- Abaqus's own doc states
  node-based surfaces are always node-to-surface only ("Surface-to-surface
  discretization is not applicable if a node-based surface is used"), so
  this is not a gap in this design, it is Abaqus's own stated limitation
  and this design should match it, not exceed it without reason.
- Friction. `DeformableSurfaceContactConstraint3D`'s own scope is already
  frictionless (module docstring); this design stays frictionless too.
- A full Wohlmuth global (patch-connected, not face-local) dual mortar
  space (§2.2's honestly-stated approximation) -- only worth the added
  global-solve cost if Stage B's face-local-then-globally-normalized
  approximation is shown, by measurement, to be insufficiently accurate
  on a real dissimilar-mesh case.
- Finite-sliding tracking (re-pairing segments as the bodies slide) --
  this design, like today's class, stays within this project's established
  small-sliding, frozen-pairing scope (`ctc_contactpairform_std.txt`'s own
  "small-sliding... groups of nodes involved with individual contact
  constraints are fixed throughout the analysis" is the exact formulation
  being matched here, not finite-sliding's continuously-changing
  connectivity).

---

## 6. Summary answers (for the requester)

1. **Clipping + quadrature**: AABB broad phase over deformed face
   coordinates (§1.1); project onto the slave face's own centroid tangent
   plane (§1.2, error `O((warp/L)^2)`, exact fallback via 2-triangle
   decomposition flagged but deferred); Sutherland-Hodgman with an
   explicit convexity check on the slave (window) polygon, falling back
   to point-collocation on the rare non-convex face rather than clipping
   against an invalid window (§1.3); fan-triangulate the ≤8-vertex clip
   polygon and integrate with a 6-point degree-4 rule as a practical
   default, verified by order-refinement rather than assumed exact, since
   the true integrand is not polynomial for a non-parallelogram or warped
   face (§1.4).
2. **Per-node active-set aggregation**: resolved by summing every
   segment's dual-weighted gap contribution into a global
   `gap_numerator[nid]` and every segment's mass contribution into a
   global `D_global[nid]` (both plain FE-style scatter-adds, no new
   linear solve), then dividing once per node to get a single `g_tilde[nid]`
   -- after which PDASS's existing per-node formulas apply completely
   unchanged, and a node's contact status is a genuine, single,
   well-defined boolean (§2.4, §3). `_contact_active_set()`/SDI in
   `dynamic3d.py` need no changes at all.
3. **Force/tangent assembly**: force distributes to a slave face's own 4
   primal-weighted nodes and every touched master face's own 4
   primal-weighted nodes via the interpolated multiplier field `lam_h(qp)
   = Σ p_i Ψ_i(qp)` (§2.5); the tangent is a sum of rank-1 outer products,
   one per active multiplier node, each sized only by that node's own
   local dof patch -- the concrete demonstration of the diagonal-`D`
   payoff the dual-mortar doc predicted but did not derive to this depth.
4. **Falsifiable accuracy prediction**: on a two-block, dissimilar-mesh-
   density interface, this design's segment-to-segment discretization
   should show measurably lower max relative CPRESS-analog error and less
   node-to-node spike/spread than today's node-to-surface class, mirroring
   Abaqus's own Table 1 finding in direction (not in absolute magnitude,
   which is explicitly not claimed to transfer) (§4).
5. **File**: `dev_log/contact_surface_to_surface_precise_design_20260915.md`
   (this document). No `dispsolver/` code was written or modified.
