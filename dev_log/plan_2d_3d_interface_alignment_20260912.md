# Pre-design: 2D ↔ 3D interface alignment (2026-09-12)

**Status: study + pre-design only. Nothing here is authorized for execution.**
The 2D element refactor (`.omc/plans/2d_element_open_items_20260912.md`,
Tiers 1–6) must close first — see §4 for the trigger. This document exists so
that when alignment work *does* start, it starts from measured facts rather
than from a fresh reading of both trees.

**Method / scope.** Everything in §1 was read directly from the working tree on
2026-09-12 (branch `feat/solver-element-abaqus-surpass-20260830`) and is cited
by file:line. Nothing under `element3d/`, `solver3d/`, `material3d/`, `mesh3d/`,
`constraint3d/`, or `model/` was modified. Claims I could not settle by reading
are marked **UNVERIFIED** and are not used to support any recommendation.

---

## 0. Executive summary

Three findings drive everything below.

1. **The CAE object model is already dimension-parametric and already has a 2D
   door — which nothing opens.** `Part(name, dim=3)` takes `dim`
   (`dispsolver/model/part.py:22-24`), `FlattenedSolverSystem` carries
   `dim: int  # 2 or 3` (`dispsolver/model/assembly.py:17-20`), and
   `to_legacy_mesh2d()` already builds a 2D `Mesh` for `DynamicSolver`
   (`assembly.py:65-83`). It has **zero callers** repo-wide. The 2D side does
   not need a new object model built for it; it needs to be plugged into the one
   that exists.

2. **On element↔material separation, 3D is AHEAD of 2D, not behind.** 3D element
   kernels take `(mat_type: int, props, sdvs, dt)` and never name a material
   (`c3d8_eas_tl_numba.py:139-146`); the constitutive law lives behind one
   `material_dispatch_3d(...)` UMAT-style call
   (`material3d/numba_materials.py:35-64`). That is precisely the separation 2D's
   Phase A `response_interface.py` describes as an unmet goal — 2D still encodes
   the material in the element *name* (`CPE4I` means "incompatible modes **+
   viscoelastic**", per `plan_abaqus_spirit_element_refactor_20260911.md` §3.7),
   and zero 2D element kernels call through `response_interface.py` yet.

3. **On frame correctness and dispatch integrity, 2D is far ahead — and the two
   are related.** 3D is *purely Total Lagrangian*: there is no `F_n`, no
   `REF_N`, no push-forward, no `ul_mode` anywhere under `element3d/` or
   `solver3d/` (grep: zero hits), and `assemble_system` always passes the
   original `self.mesh.nodes_array()` (`dynamic3d.py:256`). So the F4/B1 defect
   class **cannot currently exist in 3D** — but only because 3D has not yet
   attempted the formulation where it lives. 3D's verification is the exact
   pattern AGENTS.md §4.14 identifies as insufficient: rigid-rotation canary
   plus axis-aligned FD tangent (`tests/test_3d_tl_element.py:16,57`,
   `tests/test_3d_corotational.py:10,31`), with no rotated-reference sweep and
   no global-axis isotropy check.

**The single biggest conflict** is not naming and not architecture — it is that
the two sides made **opposite backend bets**, and the material interface is
where that bites. 2D's contract is a *JAX closure* over an opaque params blob
(`response_fn(F, state, dt, params)`, params may hold a bound Python string);
3D's is a *Numba `@njit` integer-tag switch* over a flat `props: float64[36]`
array. A closure cannot cross an `njit` boundary and a `float64[36]` cannot hold
a Prony tuple of arbitrary length. These cannot both be "the" interface without
one side changing its lowering strategy. §3.2 proposes which, and why.

---

## 1. What the 3D side actually built

### 1.1 Element kernel interface

There are **two** 3D element interfaces in the tree, and they are not the same
shape.

**(a) The legacy `C_mat` form** — small-strain, one constant elasticity matrix:

```
compute_c3d8_element_numba(coords, u_elem, C_mat)          c3d8_numba.py:104
compute_c3d8_eas_element_numba(coords, u_elem, C_mat)      c3d8_eas_numba.py:61
compute_c3d8_fbar_element_numba(coords, u_elem, C_mat)     c3d8_fbar_numba.py:25
compute_c3d4_anp_element_numba(coords, u_elem, C_mat)      c3d4_anp_numba.py:22
compute_c3d10m_element_numba(coords, u_elem, C_mat)        c3d10m_numba.py:82
```

**(b) The UMAT form** — the newer one, material-agnostic:

```
_compute_c3d8_eas_tl_element_umat_numba(
    coords_init, u_elem, mat_type: int, props, sdvs_elem, dt
) -> (K(24,24), f(24), err)                        c3d8_eas_tl_numba.py:139-146

compute_c3d8_corotational_element_umat_numba(
    coords, u_elem, mat_type, props, sdvs, dt, controls
) -> (f_global(24), K_global(24,24), error_flag)   c3d8_corotational_numba.py:190-214
```

Note the **return order differs between the two UMAT kernels** —
`(K, f, err)` vs `(f, K, err)`. Both are wrapped by mesh-level assemblers that
normalize to `(f_elems, K_elems, has_error)`
(`c3d8_eas_tl_numba.py:326-366`), so this is contained, but it is an interface
inconsistency in the same generation of code.

Form (b) additionally carries an **8-slot `controls` array** — Abaqus
`*SECTION CONTROLS` as a flat float vector:
`[distortion_control, length_ratio/j_crit, viscous_damping,
anti_inversion_barrier, min_det_f, …]` (`c3d8_corotational_numba.py:203-208`,
allocated per element at `dynamic3d.py:203-218`). **2D has no equivalent
concept at all.** This is a genuine 3D-side design contribution.

The JAX side is an OO class hierarchy instead: `SolidElement3D(ABC)` with
`shape_functions`, `shape_derivatives_natural`, `jacobian`, `b_matrix`
(`element3d/base3d.py:41-141`), and concrete `Hexa8EASElement` /
`Hexa8FbarElement` / `Tetra4ANPElement` / `Tetra10Element`. Their entry point is
`compute_element_stiffness_and_force(coords, u_elem, C_mat, states)` —
i.e. the **legacy C_mat form**, taking a list of `QuadraturePointState3D`.
So the JAX and Numba 3D paths are not two lowerings of one contract; they are
two different contracts. 2D has the same split (`*_jax.py` / `*_numba.py`) but
holds them to a stated agreement requirement (contract C10,
`tests/element_contract/`).

### 1.2 Is there anything like the config ledger?

**No.** There is a state record — `QuadraturePointState3D`
(`base3d.py:14-38`), a dataclass holding `F`, `F_elastic`, `plastic_strain`,
`eq_plastic_strain`, `backstress`, `visco_overstress`, `stress_cauchy`,
`stress_pk2`. It is a *history* record, not a *configuration* record: no field
names which configuration it is referred to, and there is no analogue of
`Config{REF0, REF_N, CURRENT}` or of `gp_internal_force()` as the sole
permitted stress↔B contraction (`element/kinematics/frame.py:102-269`).

It does not currently need one, because 3D is Total Lagrangian throughout
(§0 finding 3). The ledger's whole content is the `F_n` push-forward, which is
the identity when `F_n = I`. **This is the cleanest possible moment to give 3D
the ledger — before it has a UL path to retrofit.**

Also worth recording: `QuadraturePointState3D` is only used on the JAX/OO path
(`dynamic3d.py:110, 347, 370-372`). The production Numba path uses a flat
`elem_sdvs: (n_elems, 8, max_sdvs) float64` array (`dynamic3d.py:198-201`)
instead. Two state representations, one solver.

### 1.3 Material interface

One function, `material_dispatch_3d`
(`material3d/numba_materials.py:35-64`):

```
material_dispatch_3d(mat_type: int, props: f8[:], sdv_prev: f8[:],
                     E_voigt: f8[6], F: f8[3,3], detF: float, dt: float)
    -> (S_voigt(6), C_mat(6,6), sdv_new, error_flag)
```

Its docstring explicitly names the goal — *"Unified Material Routine (UMAT
style) … decoupling kinematics from constitutive laws"* — and it delivers it:
`MAT_LINEAR_ELASTIC=0`, `MAT_HYPERELASTIC_NEOHOOKEAN=1`, `MAT_J2_PLASTICITY=2`,
`MAT_VISCOELASTIC_PRONY=3`, `MAT_CUSTOM_ELASTIC=99` (`:11-17`), with
`get_default_sdv_count(mat_type)` declaring each one's state size (`:19-32`).

Three observations, all from reading:

- **`MAT_VISCOELASTIC_PRONY` is declared but has no dispatch branch.** It has an
  entry in the constants (`:16`) and in `get_default_sdv_count` (`:28-29`), but
  `material_dispatch_3d`'s if/elif chain goes `0 → 1 → 2 → 99 → else:
  error_flag = 2` (`:74-204`). A model tagged `mat_type=3` returns zero stress,
  zero tangent and `error_flag=2`. **This is exactly the class of defect
  2D's `_assert_dispatch_integrity()` was written to make impossible**
  (`solver/dynamic.py:481-502`) — a capability table and a dispatch chain
  disagreeing, with the table being the one you read. Symmetrically,
  `elem_mat_types` is never checked against the set of branches that exist.
- **`MAT_HYPERELASTIC_NEOHOOKEAN` is not Neo-Hookean.** The branch builds a
  constant isotropic `C_mat` and returns `S = C : E` (`:107-116`) — St.
  Venant-Kirchhoff. It also *sniffs the magnitudes of `props[0..1]`* to guess
  whether they are `(C10, D1)`, `(E, nu)` or `(mu, K)` (`:90-105`). A material
  parameterization decided by numeric magnitude is a silent-misinterpretation
  hazard of the same family as AGENTS.md §4.2.
- **`props` is fixed-width `float64[36]`** (`dynamic3d.py:133`), sized for a
  flattened 6×6 `C_mat`. A Prony series with N terms plus a WLF shift does not
  fit a fixed 36 without a packing convention, and no packing convention is
  defined. This is the concrete reason `MAT_VISCOELASTIC_PRONY` has no branch
  — **UNVERIFIED** as the actual reason, but it is the binding constraint
  either way.

### 1.4 Dispatch / assembly structure

`DynamicSolver3D.assemble_system` (`solver3d/dynamic3d.py:254-396`) has two
layers.

Element types are classified **twice, by two different tables that disagree**:

| Site | Recognized strings | Fallback |
|---|---|---|
| `_initialize_elements` (`:96-110`) | `C3D8I`/`C3D8_EAS` → `Hexa8EASElement`; `C3D8_FBAR`/`FBAR` → `Hexa8FbarElement`; `C3D8H`/`C3D8_HYBRID`/`HYBRID` → `Hexa8FbarElement` | `else:` → `Hexa8EASElement` |
| `_setup_numba_topology` (`:137-149`) | `C3D8I`/`C3D8_EAS` → k=0; `C3D8_CR`/`C3D8_COROTATIONAL`/`C3D8_FBAR_CR` → k=1; `C3D8H`/`C3D8_HYBRID`/`HYBRID` → k=2 | `else:` → k=3 |

and k is resolved to a kernel at `:273-292`: k=0 → `c3d8_eas_tl`, k=1 →
`c3d8_corotational`, k=2 → `c3d8_hybrid`, **k=3 (everything else) →
`c3d8_fbar_tl`**.

Consequences, all read directly:

- `"C3D8_FBAR"` is recognized in table 1 but **not** in table 2, so it lands in
  k=3 — which happens to be the F-bar kernel, so it works by coincidence.
- `"C3D8_CR"` is recognized in table 2 but **not** table 1.
- **`C3D4`, `C3D4_ANP`, `C3D10M` and plain `C3D8` all silently route to the
  8-node hex F-bar TL kernel.** `assemble_mesh_c3d4_anp_numba` and
  `assemble_mesh_c3d10m_numba` are exported from `element3d/__init__.py:14-15`
  but have **zero callers outside their own files** (`assemble_mesh_c3d8_numba`
  has exactly one, in `tests/test_3d_numba.py:111`). A tetrahedral mesh fed to
  `DynamicSolver3D` would index 8 nodes from a 4-node connectivity. This is
  AGENTS.md §4.2 verbatim ("`mesh.add_element(..., "Q4_COROTATIONAL", ...)`
  does nothing") reproduced on the 3D side. **UNVERIFIED** whether any caller
  reaches the tet path by another route; I found none.
- The entire fast path is wrapped in `try: … except Exception: traceback;
  print("Fastpath failed! Falling back to python loop.")` (`:333-336`), and the
  fallback is the **legacy C_mat OO path** (`:338-396`) — a different
  formulation with a different material treatment. A mid-run exception silently
  changes the physics and prints to stdout. This is the §4.8 signature
  (converges cleanly, wrong answer) with a traceback as its only warning.

There is **no `_assert_dispatch_integrity()` analogue, no `_ELEMENT_LARGE_DEF`
analogue, and no rejection of an unknown `elem_type`** anywhere in
`solver3d/`.

**Constraint assembly** is the part of 3D's dispatch I would keep verbatim:

```python
for constraint in self.constraints:
    if hasattr(constraint, "reproject_deformed"):
        constraint.reproject_deformed(u_vec)
    f_c, (rows_c, cols_c, data_c), _ = constraint.assemble(u_vec)
    ...                                              dynamic3d.py:317-324
```

declared as an ABC at `model/constraint.py:23-33` (`AssemblableConstraint`).
A constraint returns a force vector and COO triplets and mutates nothing. It is
applied in **both** the fast and fallback paths (`:317` and `:383`) — i.e. the
§4.8 early-return trap is structurally absent here. Compare 2D's
`SurfaceTieConstraint.apply_penalty(u, f_int, K_eff)`
(`constraint/surface_tie.py:184`), which takes the assembled system and mutates
it in place; that signature is what made §4.8 possible (a branch can `return`
before reaching it).

### 1.5 State (SDV) handling — two observations for the 3D session

These are not alignment matters, but they were found while reading the
interface and are material to any decision to adopt 3D's interface wholesale.

- **SDVs are committed on every trial assembly, not on convergence.**
  `sdvs_e = elem_sdvs[e]` is a *view* into `self.elem_sdvs`
  (`c3d8_eas_tl_numba.py:357`), and the kernel writes through it at
  `sdvs_elem[gp] = sdv_gp_new` (`:255-256`). `solve_step` calls
  `assemble_system` once per Newton iteration (`dynamic3d.py:432`) **plus up to
  five more times inside the Armijo line search** (`:473-487`), with no
  save/restore anywhere. So the J2 return map re-applies against already-updated
  plastic strain within a single increment, including on **rejected** trial
  displacements. This is structurally identical to AGENTS.md §4.14's **F5**
  (`_ul_F_n` compounded within a Newton step) — and F5 was masked in 2D only
  because the viscoelastic path converged in ~1 iteration. `DynamicSolver3D`
  runs 2+ iterations and up to 6 assemblies per iteration.
- **`detF` is passed the reference Jacobian ratio, not `det(F)`.**
  `material_dispatch_3d(mat_type, props, sdv_gp, E_voigt, F, detJ / detJ0, dt)`
  (`c3d8_eas_tl_numba.py:252-253`) — `detJ/detJ0` is a ratio of *reference*
  configuration Jacobians, i.e. mesh grading, and is constant in `u`. Since
  `detF` is used only for the `if detF <= 0.0: error_flag = 1` inversion guard
  (`numba_materials.py:70-72`), the guard fires on reference-mesh distortion and
  **never on actual deformation inversion**.
- Minor, same area: `solve_step` returns `True, max_iters` on loop exhaustion
  (`dynamic3d.py:491-492`) — non-convergence is reported as convergence.

### 1.6 The CAE object model

This is the 3D session's strongest deliverable and the natural convergence
point.

```
Model(name, dim=3)                                   model/model.py:34-56
├── Material / Section / SectionControls             model.py:57-127
├── Part(name, dim)                                  part.py:19-55
│   ├── GeneralSet  (nodes+elements+faces in ONE object, set algebra,
│   │                associative node resolution, interior-face
│   │                cancellation, box/sphere/cylinder/plane/normal/
│   │                predicate factories, findAt proximity)   set.py (1046 ln)
│   └── SectionAssignment(region: GeneralSet|str -> Section)  part.py:11-17
├── Assembly(dim) → Instance(Transform3D) → FlattenedSolverSystem
│                                                    assembly.py:85-304
├── Step / InitialStep, entity lifecycle
│   {CREATED, PROPAGATED, MODIFIED, DEACTIVATED, REACTIVATED}
│                                                    step.py:10-217
├── Constraint: RigidBody / Tie / KinematicCoupling /
│               DistributingCoupling / MPC          constraint.py:17-144
└── Load: ConcentratedForce / Pressure / Gravity / BodyForce
                                                     load.py
```

The dimension-parametric facts, restated because they carry §3:
`Part.__init__(self, name, dim: int = 3)` slices coordinates to `dim`
(`part.py:22-41`); `FlattenedSolverSystem` declares `dim: int  # 2 or 3` and
`num_dofs = N_nodes * dim` (`assembly.py:17-39`); and it exposes **both**
`to_mesh3d()` (`:44-63`) and `to_legacy_mesh2d()` (`:65-83`).

But: `Model.create_solver3d()` exists (`model.py:556-562`) and there is no
`create_solver2d()`. `to_legacy_mesh2d()` has **no callers**. And
`MultiStepExecutor` — which lives in `dispsolver/solver/step_executor.py`, the
**2D** package — hardcodes `self.model.create_solver3d()`
(`step_executor.py:52`) and its `StateCheckpoint` snapshots
`solver.elem_sdvs` (`:27, 34-36, 60`), an attribute only `DynamicSolver3D` has
(2D's equivalent is `self.state`, `solver/dynamic.py:1234`).

So the multi-step/branching engine is *filed under 2D* and *only drives 3D*.

---

## 2. Side-by-side

| Axis | 2D (`solver/`, `element/`, `material/`) | 3D (`solver3d/`, `element3d/`, `material3d/`) | Who is ahead |
|---|---|---|---|
| **Element↔material separation** | Material baked into element NAME (`CPE4I` ⇒ viscoelastic; `Q4_COROTATIONAL_EAS` ⇒ J2). `response_interface.py` defines the fix; **0 kernels call it** (Phase C not started) | Achieved: kernels take `(mat_type, props, sdvs, dt)`, never name a material (`c3d8_eas_tl_numba.py:139`) | **3D** |
| **Material contract shape** | `response_fn(F, state, dt, params)` — JAX closure, params an opaque NamedTuple, `make_visco_response(base)` binds the static string | `material_dispatch_3d(mat_type:int, props:f8[36], sdv, E_voigt, F, detF, dt)` — `@njit` int switch | Different bets; see §3.2 |
| **Material coverage vs declaration** | Every declared material has a path | `MAT_VISCOELASTIC_PRONY` declared, **no branch** → `error_flag=2` (`numba_materials.py:16` vs `:201-204`) | **2D** |
| **Dispatch integrity** | `_ELEMENT_LARGE_DEF` + `_DISPATCHED_ELEMENT_TYPES` + `_assert_dispatch_integrity()` at import; `__init__` rejects unknown types (`dynamic.py:382-502`) | Two disagreeing classification tables, both with silent `else:` fallbacks; C3D4/C3D10M dead; broad `except` → different formulation (`dynamic3d.py:96-154, 273-336`) | **2D, decisively** |
| **Frame / large deformation** | NLGEOM per-pid via `_use_ul_for`; config ledger `frame.py` with `Config{REF0,REF_N,CURRENT}` and a single `gp_internal_force()`; push-forward unconditional | Pure TL. No `F_n`, no `REF_N`, no push-forward (grep: 0 hits). Ledger not needed *yet* | **2D** (3D has no exposure yet) |
| **Element verification** | 11 contracts C1–C11 over every dispatched type, rotated-reference AR sweep, global-axis isotropy, FD-Jacobian-on-rotated-reference, condensation symmetry, JAX≡Numba, mesh-invariant FD step; `_XFAIL` is a *characterization baseline with measured numbers* (`tests/element_contract/`) | Rigid-rotation canary + axis-aligned FD tangent (`test_3d_tl_element.py:16,57`; `test_3d_corotational.py:10,31`). Exactly the pattern §4.14 proves insufficient — it passed for all three of F4/F5/F6 | **2D, decisively** |
| **Constraint contract** | `apply_penalty(u, f_int, K_eff)` — mutates the assembled system; skippable by an early `return` (this **is** §4.8) | `assemble(u) -> (f, (rows, cols, data), flag)` ABC, pure, applied in both assembly paths (`constraint.py:23-33`, `dynamic3d.py:317, 383`) | **3D** |
| **State (SDV) discipline** | `self.state` written only on commit (F5 fix, §4.14) | Written in place on every trial assembly incl. rejected line-search steps (§1.5) | **2D** |
| **Section controls** | none | `SectionControls` → 8-slot per-element float array, distortion control + anti-inversion barrier (`dynamic3d.py:203-218`) | **3D** |
| **Object model** | none; meshes come from `.inp` (`io/model_builder.py`) or ad-hoc builders (`mesh/plate_builder.py`) | Full `Model/Part/Set/Step/Constraint/Load` + smart `GeneralSet` + branching; `dim`-parametric; 2D exit path exists but unused | **3D** |
| **Multi-step / branching** | none | `Step` lifecycle + `StateCheckpoint` + `MultiStepExecutor` — filed in `solver/` but 3D-only (`step_executor.py:52`) | **3D** |
| **Naming** | Target is one axis: Abaqus name = kinematics only, material passed separately (plan §3.7); non-Abaqus names (`CPE4S`) to be **removed, not aliased** | Abaqus names used (`C3D8I`, `C3D8_FBAR`, `C3D10M`) but with invented aliases (`FBAR`, `HYBRID`, `C3D8_CR`) accepted alongside | 2D's *policy*; 3D's *practice* is close enough |

**Where they already agree, without coordination:**

- Both key material properties to an opaque, material-defined blob the element
  never inspects. 2D calls it `params`; 3D calls it `props`. Same idea.
- Both split JAX and Numba lowerings of the same element.
- Both treat the constraint list as a post-element additive contribution.
- Both use `nlgeom` as the user-facing switch name (`dynamic3d.py:41`,
  `dynamic.py` §4.14) — arrived at independently.
- Both use `pypardiso` with a scipy fallback.

---

## 3. Pre-design for a converged interface

Not an execution plan. A sketch, with the cost stated on **each** side.

### 3.1 A shared kinematics module

Proposal: `dispsolver/kinematics/` (dimension-agnostic), with
`dispsolver/element/kinematics/frame.py` becoming a 2D specialization.

What generalizes cleanly (the mathematics is dimension-free):

- `Config{REF0, REF_N, CURRENT}` — verbatim.
- `push_forward_stress(S, F_n) = F_n S F_nᵀ / det F_n` — verbatim; only
  `_det2` → `_det3` changes.
- `push_forward_tangent(C, F_n) = Tᵀ C T / det F_n` — the Voigt pull-back
  operator `T` is 3×3 in 2D and 6×6 in 3D. Same construction rule, different
  size. This is a build-from-`F_n` helper parameterized by the Voigt
  convention, not two unrelated functions.
- `gp_internal_force(kin, S_ref0, C_ref0)` and the **invariant that it is the
  only permitted stress↔B contraction** — verbatim, and this is the part with
  all the value.

What is genuinely dimension-specific and must **not** be forced to converge:

- `B_L` shape ((3,8) vs (6,24)) and the Voigt ordering. 2D uses
  `[E11, E22, γ12]`; 3D uses `[xx, yy, zz, xy, yz, zx]`
  (`base3d.py:121`). Do not unify the *arrays*; unify the *contract* — "B_L is
  built from `F_inc` and is therefore conjugate on REF_N", which is a statement
  about provenance, not about shape.
- Plane-strain's implicit out-of-plane condition has no 3D counterpart.
- The 2D `thickness` factor in `weight` has no 3D counterpart.

Cost to 3D: adopt `GPKinematics` in the UMAT kernels (currently they pass
`coords_init` and compute `F` inline). Under `@njit`, a frozen dataclass does
not lower — 3D would need either a `numba.experimental.jitclass`, a NamedTuple
(which njit does support), or to accept the ledger as a **JAX-side-only** device
with the Numba mirror held to it by the contract suite instead. **This is a real
constraint, not a detail** — 2D's ledger is written in `jnp` and explicitly
notes it is "built and consumed inside one traced function"
(`frame.py:76-80`). The njit answer is probably a NamedTuple; **UNVERIFIED**
that `njit` handles a NamedTuple of arrays with the zero-cost unrolling the JAX
version gets.

Cost to 2D: near zero — move the file, parameterize `_det2`/`voigt_pullback` by
dimension.

**The strongest argument for doing this at all:** 3D will eventually want UL
(large rotation with a rotated reference — the fold problem in 3D is the same
fold problem). The moment it does, F4/B1/B2/B3 become writable there. Giving 3D
the ledger *before* the UL path exists costs one refactor; giving it after costs
one refactor plus the six-defect discovery sequence 2D already paid for.

### 3.2 One material contract — and the backend problem

The two contracts are the same *function* with different *dispatch mechanics*:

```
2D:  response_fn(F, state, dt, params)          -> (S_voigt, state_new)
     tangent_response_fn(F, state, dt, params)  -> (S_voigt, C_voigt, state_new)

3D:  material_dispatch_3d(mat_type, props, sdv, E_voigt, F, detF, dt)
                                                -> (S_voigt, C, sdv_new, err)
```

Differences that are cosmetic and should converge:
- `E_voigt` and `detF` are redundant — both derivable from `F`. Drop them.
  (Doing so also removes the `detJ/detJ0` mis-pass of §1.5 by construction.)
- Argument order. Pick 2D's `(F, state, dt, params)`.
- `error_flag` vs exception. Keep 3D's flag — it is the only one that works
  under `njit`, and 2D's `_check_mesh_quality` wants the same information.

The difference that is **not** cosmetic:

| | 2D | 3D |
|---|---|---|
| How a material is selected | Python closure bound at trace time | `int` tag compared at runtime inside `njit` |
| How params travel | Opaque NamedTuple; may contain arrays of any shape; a static string can be *bound into the closure* (`make_visco_response(base)`) | Flat `float64[36]`; no room for a variable-length Prony series; no way to carry a string |

A single contract requires choosing. The two coherent options:

**Option A — int-tag everywhere (3D's bet wins).** 2D materials get integer
tags and a flat props array. Cost to 2D: severe. It gives up
`make_visco_response(base)`, gives up the "params is opaque, element never
inspects it" property that `response_interface.py` is built on, and forces a
packing convention for Prony/WLF that does not exist even in 3D. Cost to 3D:
zero. **Not recommended** — it trades 2D's stronger property for uniformity.

**Option B — closure contract is canonical; the int tag becomes a *lowering
detail* (2D's bet wins, 3D keeps its kernels).** The contract is
`response_fn(F, state, dt, params) -> (S, state_new)`. The JAX lowering binds
the closure directly. The **Numba lowering** keeps exactly what
`material_dispatch_3d` already does — an int switch — but the tag and the props
packing are *generated from* the material object by a registry, not hand-written
per call site. Cost to 3D: define a `MaterialSpec` (tag, props-packing fn,
sdv-count) per material and stop hand-coercing in
`_setup_numba_topology`'s ladder (`dynamic3d.py:156-196` — six `hasattr`/`dict`
branches with magnitude-sniffing, the exact thing a registry removes). Cost to
2D: write the Numba lowering side, which Tier 2 of the current plan is already
touching. Replace `props: float64[36]` with a ragged layout
(`props_flat` + `props_offsets`) so Prony fits — this is what unblocks
`MAT_VISCOELASTIC_PRONY`.

**Recommendation: Option B.** It preserves the property 2D's interface exists
to defend, and it does not ask 3D to rewrite a single kernel — only to build the
registry that fills `elem_mat_types`/`elem_props`, which is *already* a
hand-written coercion ladder that should not be hand-written.

### 3.3 Dispatch integrity, shared

`_assert_dispatch_integrity()` (`dynamic.py:481-502`) is ~20 lines and is
dimension-independent: it asserts that a capability table's keys and a
dispatch-branch set are equal, at import. Proposal:
`dispsolver/dispatch_integrity.py` exposing
`assert_dispatch_integrity(declared: set, dispatched: set, context: str)`, with
each solver calling it on its own two sets.

Cost to 3D: build the two sets it does not currently have —

- a `_DISPATCHED_ELEMENT_TYPES3D` (the strings k=0/1/2 actually recognize), and
- a declaration table whose keys are the element types the solver claims to
  support.

The assert then fails at import, correctly, because `C3D4_ANP`/`C3D10M`/`C3D8`
are declared-and-exported but not dispatched (§1.4). **That failure is the
point** — it is the same failure that found 2D's `"CPE4"` and `Q4_VISCO_EAS`
defects on the day it was written. It must be run in the 3D session's own time,
not sprung on them.

Cost to 2D: extract the function, keep behavior identical.

Same treatment for the material side: `assert_dispatch_integrity` over
`{MAT_* constants}` vs `{branches in material_dispatch_3d}` catches
`MAT_VISCOELASTIC_PRONY` immediately.

### 3.4 The contract suite

`tests/element_contract/` (C1–C11) is the 2D side's most transferable asset.
The contracts are dimension-independent *statements*:

- C6 rotated-reference AR sweep, C7 global-axis isotropy, C8 "K is the FD
  Jacobian of the element's own residual, on a rotated reference", C9
  condensation symmetry, C10 JAX≡Numba, C11 mesh-invariant FD step — every one
  of these is a claim about an element, not about a dimension.
- The harness's key design choice — *drive through `Solver._assemble()`, not
  through kernel functions, because a kernel-level probe tests code the solver
  may not be running* (`tests/element_contract/harness.py`) — applies with
  **more** force in 3D, where `assemble_system` has a silent `else:` → F-bar and
  a broad `except:` → a different formulation.
- The `_XFAIL` discipline (every entry carries a *measured number* and a
  `BY DESIGN` / `DEFECT` classification; a passing entry shows as `xpass` and is
  deleted, never widened) is a process asset, not a code asset, and transfers
  for free.

Cost to 3D: parameterize the harness over `(solver_class, dim, n_dofs_per_node,
probe_geometry)`. C10 becomes immediately valuable there because 3D has two
genuinely different element interfaces (§1.1), not two lowerings of one.

### 3.5 Should the CAE object model own 2D meshes?

**Yes, and it is closer than either side realizes** — `Part(dim=2)` and
`to_legacy_mesh2d()` already exist and already work (`assembly.py:65-83`);
nothing calls them. The concrete shape:

- `Model.create_solver2d()` mirroring `create_solver3d()`
  (`model.py:556-562`), routing through `to_legacy_mesh2d()`.
- `MultiStepExecutor` (`solver/step_executor.py`) takes the solver factory
  instead of hardcoding `create_solver3d()` at `:52`, and `StateCheckpoint`
  snapshots a named state attribute rather than `elem_sdvs` specifically
  (2D's is `self.state`, `dynamic.py:1234`). Then multi-step branching works for
  the fold problem in 2D — which is where the fold problem actually is.
- `gen_ex12_inp.py`'s graded mesh becomes a `Part` + `GeneralSet`s. The
  `_graded_display_x()` transition clustering (AGENTS.md §4.12) is exactly what
  `create_set_from_box` / `find_at` are for.

**What must not converge:** `dispsolver/io/model_builder.py`'s `.inp` reader is
2D's production entry point and `ex12_abaqus_inp_plate_fold.py` is the reference
result (AGENTS.md §1.0). Any object-model adoption is **additive alongside** it,
never a replacement, until the 90°/side result reproduces through the new path.

### 3.6 Naming

2D's policy (`plan_abaqus_spirit_element_refactor_20260911.md` §3.7) should
win — it is a *policy* and 3D has no competing one:

- Abaqus element name = **kinematics only**; material passed separately.
- Invented names get **removed, not aliased**, when they claim an Abaqus element
  that does not exist. In 3D that means auditing `FBAR`, `HYBRID`, `C3D8_CR`,
  `C3D8_FBAR_CR` (`dynamic3d.py:139-146`): `C3D8I`, `C3D8H`, `C3D8R`, `C3D4`,
  `C3D10M` are real; `C3D8_CR` is not. 3D is already closer to compliant than
  2D is.
- The `supports_nlgeom` rename (plan §3.6) should land on both at once.

---

## 4. Sequencing

**Do not start any of this now.** The 2D refactor is mid-flight (Tiers 1–6,
`.omc/plans/2d_element_open_items_20260912.md`), and §3's proposals all assume
2D's element layer has stabilized. Starting earlier means aligning against a
moving target and re-doing it.

**Trigger: 2D Tier 3 closes** (`CPE4` wired end-to-end through the config
ledger and verified against `cook_membrane.py` / `nlgeo_cantilever.py` plus the
contract suite). Rationale: Tier 3 is the first point at which the ledger has
been *proven on a real element through the real solver*, rather than existing as
infrastructure. Aligning 3D onto an unproven abstraction would propagate a
design error into a second codebase. Tiers 4–6 (ΔF reformulation, hybrids,
Stage-4 gate) change *which elements exist*, not *what the interface is*, so
they need not block alignment.

**Sequence after the trigger, cheapest-and-most-informative first:**

| # | Step | Touches | Risk |
|---|---|---|---|
| 1 | Extract `assert_dispatch_integrity()` to a shared module; 2D calls it unchanged | 2D only | ~zero |
| 2 | 3D builds its two sets and calls it. Expect an import failure naming `C3D4_ANP`/`C3D10M`/`C3D8`/`MAT_VISCOELASTIC_PRONY` | 3D only | Low — fails loudly at import, never silently at physics time |
| 3 | Generalize the contract harness over `(solver, dim)`; run C1/C2/C8 against the 3D UMAT kernels as a **characterization baseline** (`_XFAIL` with measured numbers), fixing nothing | tests only | ~zero; produces the number that decides everything after |
| 4 | `MaterialSpec` registry (§3.2 Option B), replacing `dynamic3d.py:156-196`'s coercion ladder; ragged props layout; wire `MAT_VISCOELASTIC_PRONY` | 3D + shared | Medium |
| 5 | `dispsolver/kinematics/` with `Config` + push-forwards; 2D's `frame.py` re-exports from it | 2D + shared | Low (pure move) |
| 6 | 3D adopts `GPKinematics` — **only if** 3D is about to gain a UL path | 3D | Medium; defer until UL is actually wanted |
| 7 | `Model.create_solver2d()`; `MultiStepExecutor` parameterized over the solver factory | `model/` + `solver/` | Medium — needs the 3D session's agreement, it owns `model/` |

**First low-risk step: #1 + #3.** Neither changes any formulation, both are
additive, and together they replace opinion with numbers before any interface is
committed to. #3 in particular tells us whether 3D's TL kernels satisfy the
contracts that cost 2D six defects to learn — and that answer should be known
*before* anyone argues about whose interface wins.

**Prerequisite, non-negotiable:** steps 2, 4, 6, 7 modify files the concurrent
session owns. They need that session's explicit handoff, not a merge. The
git-stash collision on `part.py`/`set.py` already flagged in the current 2D plan
(Risks table, last row) is the same hazard.

---

## 5. Open questions for the user

1. **Which material contract wins?** §3.2 recommends Option B (2D's closure
   contract canonical; 3D's int-tag demoted to a Numba lowering detail generated
   from a registry). The alternative (Option A) is cheaper for 3D and costs 2D
   the opaque-params property `response_interface.py` was built to provide.
   This is the one decision everything else hangs from.

2. **Should the CAE object model own 2D meshes?** The machinery exists and is
   unused (`Part(dim=2)`, `to_legacy_mesh2d()`). Wiring it means
   `ex12_abaqus_inp_plate_fold.py`'s `.inp` path and an object-model path
   coexist for some time. Is that acceptable, or should 2D stay on `.inp` until
   the object model can reproduce the 90°/side reference result end-to-end?

3. **Where does `MultiStepExecutor` belong?** It is in `dispsolver/solver/`
   (2D's package) and drives only 3D (`step_executor.py:52`). Should it move to
   a dimension-neutral location, or should `solver/` become the shared solver
   package with `solver3d/` folded in eventually?

4. **Does 3D want Updated Lagrangian?** This changes the cost of §3.1 by an
   order of magnitude in both directions. If 3D stays TL forever, the config
   ledger is optional there. If 3D will fold a display in 3D
   (`ex14_3d_display_fold.py` already drives 180°), it will want UL, and the
   ledger should land *before* that path exists.

5. **Who fixes what I found in §1.5?** The SDV-commit-on-trial-assembly issue
   and the `detJ/detJ0`-as-`detF` mis-pass are 3D-side correctness matters, not
   alignment matters. This session did not touch them (read-only mandate). They
   should be relayed to the session that owns `solver3d/` — the SDV one in
   particular is the same defect class as AGENTS.md §4.14's F5 and is likely
   active today on any 3D model with `MAT_J2_PLASTICITY`.

6. **Naming authority.** §3.6 proposes 2D's policy wins because 3D has none.
   That implies auditing `FBAR`/`HYBRID`/`C3D8_CR` on the 3D side. Confirm
   before anyone deletes an alias someone's script depends on.
