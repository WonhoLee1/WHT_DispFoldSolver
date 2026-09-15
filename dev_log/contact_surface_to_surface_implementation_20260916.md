# Surface-to-surface (dual-mortar) contact -- Stage A implementation (2026-09-16)

Implements `dev_log/contact_surface_to_surface_precise_design_20260915.md`'s
Stage A (matching-topology, single segment per slave face -- no AABB
broad phase, no Sutherland-Hodgman clipping yet, that is Stage B).
New file: `dispsolver/constraint3d/surface_contact3d_s2s.py`
(`SurfaceToSurfaceContactConstraint3D`). Existing
`DeformableSurfaceContactConstraint3D` (node-to-surface) is untouched --
this is an additive alternative, not a replacement, matching the design
doc sec2.6's own flag that `slave_node_ids` -> `slave_faces` is a real,
breaking API change.

Priority context: this was picked up as priority item #1 of
`dev_log/contact_development_report_20260915.md`'s "다음 단계" list,
following user instruction "우선순위로 진행" after Stage 1 contact
chattering stabilization was completed
(`dev_log/contact_stabilization_precise_design_20260915.md`).

## What was built

Exact port of the design doc's math (sec2.1-2.5), in the class's own
variables:

- `_build_reference_mortar_matrices()`: per slave face, `M_f` (2x2
  Gauss, reference config), `D_f = M_f.sum(axis=1)`, `A_f =
  diag(D_f) @ inv(M_f)`, and the global `D_global[nid]` scatter-add.
- `_build_pairs()`: one segment per slave face (Stage A), frozen
  centroid-based pairing + per-quad-point frozen `(xi_m, eta_m)`
  projection, sign-oriented normal (small-sliding, matching every
  sibling class's convention).
- `_compute_g_tilde(u)`: the two-pass weak-gap aggregation
  (`gap_numerator` scatter-add -> divide by `D_global` once).
- `get_active_set(u)`: plain `frozenset` of slave node ids -- confirmed
  (not just claimed) that `DynamicSolver3D._contact_active_set()`/SDI
  need zero changes to consume this class.
- `assemble(u)`: force via the interpolated multiplier field `lam_h(qp)
  = sum p[face[a]]*Psi_a(qp)`, distributed to slave/master PRIMAL nodes;
  tangent as one rank-1 outer product per ACTIVE node, summed over every
  segment touching that node (handles a node shared by up to 4 slave
  faces).
- `update_augmented_multipliers(u, omega)`: PDASS-compatible, `g_tilde`
  substituting for point penetration in the unmodified Alart-Curnier
  formula -- confirmed `DynamicSolver3D.solve_step()` already calls this
  automatically (via `getattr(c, "augmented_lagrange", False)`) at its
  existing commit points, no solver-side change needed.

## Verification (`tests/test_surface_to_surface_contact.py`, 8 tests, all passing)

1. **Dual-basis bi-orthogonality** (`test_dual_basis_biorthogonality_flat_face`):
   `integral(Psi_i * N_j dA) == D_f[i]*delta_ij`, checked with an
   INDEPENDENT 4x4 Gauss rule (not the class's own 2x2) -- the defining
   Wohlmuth dual-basis property, re-derived, not assumed.
2. **D_f row-sum identity** (`test_dual_basis_D_f_sums_to_face_area`):
   `sum(D_f) == face area` exactly, from `sum_i N_i == 1`.
3. **Closed-form uniform-pressure patch test**
   (`test_single_face_pair_uniform_pressure_patch_test`): a single
   coincident (gap0=0) slave/master face pair under uniform penetration
   must split force `p*Area/4` per corner (the textbook bilinear-quad
   uniform-load split) -- checked against the CLOSED FORM, plus a
   whole-vector Newton's-third-law balance check.
4. **Multi-face shared-node aggregation**
   (`test_grid_2x2_shared_node_aggregation_and_force_balance`): a 2x2
   grid of matching faces (9 nodes, interior nodes shared by up to 4
   faces) under uniform penetration -- `g_tilde` must stay EXACTLY equal
   to the imposed penetration at every node (the discrete
   partition-of-unity check design doc sec2.5 names) and the whole force
   vector still balances.
5. **Real Newton solve** (`test_full_solve_two_block_matching_topology_converges`):
   two stacked C3D8 blocks, each genuinely subdivided 2x2 in-plane at
   the interface (not a bare geometric face -- real FE nodes, real
   elemental stiffness), converges cleanly through
   `DynamicSolver3D.solve_step()` with zero solver-side changes.
6. **PDASS/AL contraction** (`test_full_solve_two_block_augmented_lagrangian_converges`):
   repeated (solve_step + update_augmented_multipliers) cycles shrink
   max_penetration to the floating-point floor within a handful of
   cycles (faster than the point-collocation sibling's own ~0.77/cycle,
   because solve_step()'s own PDASS is also active automatically on top
   of the explicit outer-loop call).

## A real finding, documented rather than papered over: `penalty_stiffness` units differ from the point-collocation class

An initial cross-check test assumed this class's `penalty_stiffness`
should be numerically interchangeable with
`DeformableSurfaceContactConstraint3D`'s own `penalty_stiffness` on the
same mesh. It is not, and the discrepancy (~3.8x on the two-block
fixture) is not a bug -- it is a genuine units difference, confirmed by
dimensional analysis of the design doc's own formulas:

- `g_tilde` carries units of LENGTH (Psi is dimensionless; `penetration`
  is length; `qp.w` is a physical AREA; dividing by `D_global`, also an
  area, leaves length).
- The force-distribution formula multiplies `p = lam +
  k_contact*g_tilde` by a physical area weight `w` to produce a force
  -- so `k_contact` here must already be a PRESSURE-like penalty
  modulus (force/length^3), not the point-collocation class's own
  per-node lumped spring constant (force/length, already scaled for one
  node's tributary area, confirmed by that class's own docstring).

Documented in both the class's own `penalty_stiffness` docstring and
`tests/test_surface_to_surface_contact.py::test_penalty_stiffness_units_differ_from_point_collocation_by_design`,
which turns the (initially surprising) discrepancy into a positive,
checked assertion instead of leaving it as a silent trap. **Any future
`k_ref`-style auto-derivation for this class must NOT reuse
`DeformableSurfaceContactConstraint3D`'s own formula unchanged.**

## What is deliberately NOT done here (Stage B, per design doc sec5.4)

- No AABB broad phase, no Sutherland-Hodgman clipping -- Stage A is
  matching-topology only. The accuracy benefit over node-to-surface
  (Abaqus's own measured 13-31% vs ~1% on a DISSIMILAR-mesh-density
  interface) cannot be demonstrated by Stage A alone (design doc sec0's
  own point 2) -- this implementation is a correctness/regression gate
  for the mortar machinery (D_global aggregation, rank-1-per-node
  tangent, PDASS compatibility), not the accuracy-benefit deliverable
  itself.
- No CAE-level (`ContactPair`/`interaction.py`) API exposure yet -- the
  class is usable directly (as these tests do), but
  `ContactPair.build_runtime_constraint()` does not yet route to it (it
  would need a `slave_faces`-capable surface resolver on the slave side,
  mirroring `_resolve_master_faces()`'s existing pattern on the master
  side, plus a `discretization="NODE_TO_SURFACE"|"SURFACE_TO_SURFACE"`
  selector, likely on `ContactPair` mirroring Abaqus's own
  `*CONTACT PAIR` discretization option). Flagged as the natural next
  step, not built speculatively ahead of a concrete need.
- Dual Mortar's own "what does the user get" framing
  (`dev_log/contact_dual_mortar_precise_design_20260915.md`) is now
  concretely answered by this same class: the diagonal-`D` payoff
  (local, rank-1-per-node tangent, no dense/indefinite coupled solve) is
  implemented and verified above, not just derived on paper.
