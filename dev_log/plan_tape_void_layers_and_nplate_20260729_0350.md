# PLAN — Tape layers with void regions + N-plate part-to-part contact

**Status**: approved 2026-07-29, **not implemented**. Intended to be
executed by a separate agent session. Companion documents:
`dev_log/plan_s2s_contact_stabilization_REVIEWED_20260729_0314.md`
(step S4 depends on it) and
`dev_log/plan_verification_suite_extension_20260728_2248.md`.

---

## Context

Two connected modeling capabilities are needed, planned together because
they share the same mesh-partitioning machinery:

1. **Tape layers with void regions.** A display layer (a "tape") must
   support x-intervals where **no elements are created**. Real tape
   layers are mostly filled — voids are specific cutouts (typically at
   hinge regions so the stack can bend), not "everything between
   plates." For mesh conformity with the layers above/below, every void
   boundary must become a partition line in the **global** display
   x-coordinate list, so every surviving element stays a complete,
   conforming quad.
2. **N plates with part-to-part contact.** The model currently
   hardcodes exactly two plates (left/right). It must generalize to N,
   with contact defined per part-pair: display ↔ plate A, display ↔
   plate B, … This is the prerequisite for multi-hinge models, and it
   is the modeling side of the S2S contact work reviewed in
   `dev_log/plan_s2s_contact_stabilization_REVIEWED_20260729_0314.md`.

**User decisions already made (settled, do not revisit):**
- Voids are keyed by **physical layer number** (1-based, matching the
  existing `DISP_LAYER{n:02d}` numbering), not by `LayerSpec` — because
  `layer_pattern` is a *repeating* unit, so a field on `LayerSpec` would
  apply to all 7 repeats of that material.
- Void intervals are **absolute x coordinates**. No mirroring, no
  centerline-relative mode — one rule, works for asymmetric and
  multi-hinge layouts.
- Void regions are defined **independently of plate positions** (not
  auto-derived from plate gaps).

**Nothing here exists today.** A repo-wide search for void / hole /
cutout / element-mask / element-deactivation machinery returns nothing;
both mesh generators are unconditional `for j / for i` loops over a full
dense grid, and `Mesh` has no concept of an inactive element.

---

## A. Tape layers with void regions

### A.1 Config shape

`dispsolver/fold_model_config.py`, additive to `GeometryConfig`:

```python
    # Physical-layer index (1-based, matching DISP_LAYER01..NN and the
    # Qt viewer's Part/Layer list) -> list of [x_start, x_end] absolute-x
    # intervals where that layer has NO elements. Layers absent from this
    # dict are fully filled (the default for every layer).
    #
    # Keyed by physical layer rather than by LayerSpec because
    # layer_pattern is a *repeating* unit -- a field on LayerSpec would
    # apply the same cutout to all n_layer_pairs repeats of that
    # material, which is never what a single tape layer means.
    layer_void_regions: Dict[int, List[Tuple[float, float]]] = field(default_factory=dict)
```

Example: `layer_void_regions={3: [(-10.0, -5.0), (5.0, 10.0)]}` puts two
cutouts in physical layer 3.

Validation to add (fail loudly at config-read time, not mid-mesh):
intervals within a layer must be sorted, non-overlapping, `x0 < x1`, and
clipped inside `[-display_half_length, +display_half_length]`; layer
indices must be in `1..n_layer_pairs*len(layer_pattern)`.

### A.2 The x-partition problem — the core of this feature

`examples/gen_ex12_inp.py::_graded_display_x()` currently builds the
x-array from a **hardcoded 8-entry segment list** with no mechanism for
injecting extra coordinates.

**Do not merge void edges with `np.union1d`.** It dedups on exact float
equality, but segment interiors come from `np.linspace`, so a void edge
at `x=5.0` inside the `hinge_span_dx` segment is generated as
`-8.0 + k*0.25` — close to but generally *not* bit-identical to `5.0`.
The result is a sliver element ~1e-15 wide. The existing guard
(`gen_ex12_inp.py:70`) is `assert np.all(np.diff(xs) > 0)` — strictly
`> 0`, **not** `> tol` — so it **passes**, and the degenerate element
reaches the solver as a near-zero-`det(J)` element.

**Rewrite as a breakpoint-driven build** (~25 lines, one function):

```python
def _zone_dx(x_mid: float, config) -> float:
    """Grading dx for the zone containing x_mid (by |x|)."""
    g, geo = config.grading, config.geometry
    ax = abs(x_mid)
    if ax >= geo.display_half_length - g.tip_cluster_width:  return g.tip_dx
    if ax >= geo.hinge_half_gap:                             return g.plate_body_dx
    if ax >= g.hinge_span_half_width:                        return g.hinge_edge_dx
    return g.hinge_span_dx


def _graded_display_x(config=DEFAULT_CONFIG, extra_breakpoints=()) -> np.ndarray:
    # 1. base zone boundaries (as today) + every void edge + any caller-
    #    supplied breakpoint (e.g. plate edges, see B)
    # 2. sort, clip to [-half_len, half_len], dedupe with TOLERANCE
    #    (snap near-coincident breakpoints together -- this is what
    #    prevents slivers)
    # 3. seg(a, b, _zone_dx(midpoint)) for each consecutive pair
    # 4. drop degenerate segments (existing filter), concatenate
    # 5. assert np.all(np.diff(xs) > tol)   # tolerance, not > 0
```

Why this is correct by construction: every breakpoint becomes a
`np.linspace` **endpoint**, which is exactly representable, so the
strictly-increasing property holds without float luck. Grading behavior
is unchanged because `dx` is still zone-derived. The existing
degenerate-segment filter already handles a void edge landing exactly
on a zone boundary.

**The invariant this buys**: because every void edge is a breakpoint, no
element column can straddle a void boundary — each column is entirely
inside or entirely outside each void. Membership can then be tested
exactly by column midpoint:

```python
def _column_in_void(x_left, x_right, voids, tol=1e-9) -> bool:
    xm = 0.5 * (x_left + x_right)
    return any(v0 - tol <= xm <= v1 + tol for v0, v1 in voids)
```

### A.3 Node/element generation — emit only referenced nodes

Two-pass, replacing the current single-pass dense-grid loop:

- **Pass 1**: for every (row `j`, column `i`) cell, decide survival —
  the cell is skipped if `_column_in_void(...)` for the physical layer
  that row `j` belongs to. Mark the 4 corner nodes of each surviving
  cell as referenced.
- **Pass 2**: assign IDs **only to referenced nodes**, row-major (stable
  ordering), into a dict `node_id_of[(j, i)] -> nid`.

**Why not keep the full grid and tolerate orphans**: orphan nodes are
numerically survivable (they get `_eps_reg` on the diagonal at
`dynamic.py:1768-1801` and resolve to `du = 0`), but `dynamic.py:1769`
does `K_red = K_red.toarray()` — the kinematically-condensed system is
**densified**, so every unreferenced DOF costs quadratically in an
`n_independent²` dense matrix that is already ~6.6k×6.6k for this model.
Emitting only referenced nodes avoids that entirely.

Note the orphan count depends on the tape layer's `n_rows`: a void in a
layer with `n_rows == 1` (like the current bottom PSA layer) produces
**zero** interior orphans, because node rows `r0` and `r1` are still
consumed by the layers below and above. Only `n_rows >= 2` layers create
interior orphan rows. The two-pass approach handles both uniformly.

**Required companion fix** — `gen_ex12_inp.py:200-201` derives a node's
x-coordinate positionally:

```python
left_disp_bot = [nid for nid in bottom_surface_nids
                 if xs_disp[(nid - 1) % (nx_disp + 1)] <= -geo.hinge_half_gap]
```

This is only valid for a full dense grid and returns the **wrong x for
every node past the first gap** once numbering is sparse. Replace with an
explicit nid→x lookup. (The `run_build()` path already uses
`mesh.nodes[n].x` and is safe.) This is a latent fragility worth fixing
regardless of this feature.

### A.4 Extract the shared builder — do this first

The display mesh is currently generated **twice, independently**:
`gen_ex12_inp.py::generate()` (writes `.inp` text) and
`ex13_unified_model_io.py::run_build()` (builds Python objects). The loop
shapes have already drifted structurally — the `.inp` path groups
elements by `layer_row_spans` (per physical layer), the build path by
`ROW_PID[j]` (per row). Implementing void logic twice guarantees the two
paths diverge; this is precisely the hazard `fold_model_config.py`'s own
docstring says the config module was created to prevent.

**Extract to `dispsolver/mesh/display_builder.py`** (new), returning a
pure data description both callers consume:

```python
@dataclass
class DisplayGrid:
    xs: np.ndarray                         # graded x-coordinates (breakpoint-driven)
    ys: np.ndarray                         # row boundaries, bottom -> top
    node_id_of: Dict[Tuple[int, int], int] # (j, i) -> node id, referenced nodes only
    nodes: List[Tuple[int, float, float]]  # (nid, x, y), emission order
    cells: List[Tuple[int, int, int, Tuple[int,int,int,int]]]
                                           # (layer_idx, row j, col i, (n1,n2,n3,n4))
    layer_row_spans: List[Tuple[int, int, str]]   # (r0, r1, material_name) per layer
    bottom_surface_nids: List[int]         # j == 0, referenced only


def build_display_grid(config) -> DisplayGrid: ...
```

`gen_ex12_inp.py` then formats this into `.inp` text; `run_build()`
turns it into `mesh.add_node`/`mesh.add_element` calls. Void logic,
breakpoints, and node numbering exist in exactly one place.

### A.5 Tie coupling under voids

With "emit only referenced nodes," tie filtering becomes **automatic**:
node row `j=0` is referenced only by elements in row 0, so a void in the
bottom layer means those nodes are never created and therefore never
enter `bottom_surface_nids` or the tie slave set. No special-casing
needed — this is a further argument for A.3's policy over orphan
tolerance.

If the tape layer is *not* the bottom layer, the tie is unaffected
entirely.

`SurfaceTieConstraint._build_tie_pairs` (`dispsolver/constraint/surface_tie.py:60-104`)
already tolerates a sparse slave set — it loops slaves independently and
drops any whose nearest-master distance exceeds `position_tolerance=0.5`.

---

## B. N plates + part-to-part contact

### B.1 `create_folding_plate_parts()` generalization

The two-plate hardcoding is **exactly one literal**
(`dispsolver/mesh/plate_builder.py:64-67`):

```python
    plates_config = [
        ("left",  left_x_range,  left_pivot),
        ("right", right_x_range, right_pivot),
    ]
```

`curr_nid`/`curr_eid` already run continuously across plates and the loop
body is name-agnostic, so generalizing inside `plate_builder.py` is
small: accept `plates: List[PlateSpec]` and keep the four
`left_*`/`right_*` kwargs as a backward-compat shim that constructs the
2-plate list when `plates` is not given.

Config addition (`GeometryConfig`), defaulting to today's geometry so
nothing changes for existing models:

```python
@dataclass
class PlateSpec:
    name: str                       # "left", "right", "center", ... -> part label
    x_range: Tuple[float, float]
    pivot: Tuple[float, float]
    theta_max_deg: float            # per-plate drive angle (see B.3)
```

### B.2 Callers that assume exactly two plates

Must be generalized to loop:

| File | What assumes 2 |
|---|---|
| `examples/gen_ex12_inp.py` | `plates["left"]`/`["right"]` (~:143); 4 NSETs + 2 SURFACEs (:214-228); 2 `*RIGID BODY` + 2 `*TIE` (:232-238); BC block emitting exactly two RP lines with `-theta_rad`/`+theta_rad` (:250-258) |
| `examples/ex13_unified_model_io.py` | `plates["left"], plates["right"]` (:151); 2 RBE2 + 2 ties (:197-213); 4-entry `boundaries` list (:218-223) |
| `examples/ex12_rigid_plate_display_fold_corotational.py` | third caller, hardcoded `(-40,-10)/(10,40)` (:93) |
| `tests/test_rigid_plate_tie.py` | references the 2-plate builder |

**Already N-plate-clean** (no change needed — verify, don't rewrite):
`ex12_abaqus_inp_plate_fold.py:271-276` iterates `result.rbe2_constraints`
generically; `result.penalty_constraints` likewise;
`dispsolver/io/model_builder.py:416-418` derives part names from any
`PLATE_*` elset prefix (so "Plate Center" etc. label correctly in the
viewer for free).

**Soft assumption to preserve**: `_laminate_layer_materials(..., max_node_id=10000)`
(`ex12_abaqus_inp_plate_fold.py`) and `LayerSlipTracker(..., max_node_id=10000)`
(`dispsolver/postprocess/interlayer.py`) both split "display vs plate" by
`nid >= 10000`. Fine for N plates as long as every plate keeps
`base_node_id=10000`.

### B.3 Multi-hinge drive angles

Today: `theta_L = -theta_max`, `theta_R = +theta_max`, hardcoded. For N
plates each plate needs its own target, hence `PlateSpec.theta_max_deg`
above. Keep the existing C²-smooth ramp (`amp(t)`) shared across plates
so all plates stay synchronized in pseudo-time — only the amplitude
differs per plate.

Open design point to confirm with the user when this is implemented:
for a multi-hinge (e.g. Z-fold) layout, plate angles are typically
**alternating sign** and may be relative to the neighbouring plate
rather than to global x. The `PlateSpec.theta_max_deg` field as
specified is absolute; that is the simplest thing that works and can be
extended later.

### B.4 Part-to-part contact pairing

Contact is defined per part-pair — display ↔ plate A, display ↔ plate B,
… — i.e. **one contact object per plate**, constructed by looping the
plate list rather than the current hardcoded `tie_left`/`tie_right`.

This is the point where this plan meets
`dev_log/plan_s2s_contact_stabilization_REVIEWED_20260729_0314.md`. That
review's conclusions apply here unchanged; the two most relevant:

- Locate contact in `dispsolver/contact/` (a contact subsystem already
  exists there — `ContactPair`, `ContactSurface`, `SpatialHashGrid`,
  `.inp` `*CONTACT PAIR` parsing — all wired but never exercised by any
  example or test).
- Any contact object added to `penalty_constraints` sits in the exact
  position of AGENTS.md §4.8's `[CRITICAL, silent]` failure, where
  penalty constraints were skipped by the batch assembly path and the
  display stayed at **exactly zero displacement** while Newton converged
  perfectly every step. Verification must assert contact is *doing
  something*, not merely that the solve converged.

Until that contact work lands, N plates can be driven with the existing
`SurfaceTieConstraint`, one per plate — the tie path already works and
reaches full closure.

---

## C. Sequencing

Land in gated steps; each gate must pass before the next starts.

| Step | Scope | Gate |
|---|---|---|
| **S0** | Extract `dispsolver/mesh/display_builder.py` (A.4) with **no** behavior change — both `gen_ex12_inp.py` and `run_build()` consume it. | Regenerated `.inp` is **byte-identical** to the pre-refactor file; all 3 ex13 modes reach full closure with identical node/element/pid counts. |
| **S1** | Breakpoint-driven `_graded_display_x()` (A.2), still no voids. | With no extra breakpoints, output array is **numerically identical** to today's (`np.allclose` at machine precision, same length). |
| **S2** | Void regions: config field + validation + two-pass node/element emission + nid→x fix (A.1, A.3, A.5). | A void-free config reproduces S1 exactly; a config with voids produces zero orphan nodes, no sliver elements (`min(diff(xs))` above tolerance), `n_inverted == 0`, and full 90°/side closure. |
| **S3** | N-plate generalization (B.1-B.3), still tie-based. | Default 2-plate config produces the **same** result as today; a 3-plate config builds, ties, and solves. |
| **S4** | Part-to-part contact per plate (B.4) — gated on the S2S contact plan. | See the S2S review's own M1-M4 gates. |

S0 and S1 are pure refactors with exact-reproduction gates — do them
first and separately, so that when S2 changes results you know the change
came from voids and nothing else.

---

## D. Verification

1. **Refactor gates (S0/S1)**: byte-identical `.inp`, numerically
   identical x-array. These are the cheapest possible regression checks
   and they make everything downstream interpretable.
2. **Sliver check**: assert `np.min(np.diff(xs)) > 1e-6` for every test
   config, including deliberately adversarial ones — a void edge placed
   exactly on a zone boundary (`x = ±10`, `±8`), and one placed a
   floating-point hair away from it (`10.0000000001`). This is the
   specific failure the current `> 0` assertion cannot catch.
3. **Conformity check**: every emitted element has 4 distinct nodes,
   positive area, and every emitted node is referenced by ≥1 element
   (zero orphans by construction).
4. **Cross-path parity**: all three `ex13` modes (`read` / `roundtrip` /
   `build`) must agree on node count, element count, pid count, and
   `reached_target` for a void-containing config. This check is what
   caught the silent Arruda-Boyce material bugs earlier in this
   project's history and is the single most valuable test here, since
   voids are implemented across two code paths.
5. **Full closure, not a smoke run**: `reached_target=True`, zero
   cutbacks, `n_inverted == 0` at 90°/side. Tip-element inversion
   (AGENTS.md §4.12) appears at ~48°, so a short run proves nothing
   about a mesh change.
6. **Tie integrity under voids**: assert no tie slave node lies inside a
   bottom-layer void, and that the tie still reports a tight gap
   (`check_tie_gap` in `dispsolver/solver/diagnostics.py`).
7. **Visual**: open a void-containing result in the Qt viewer and
   confirm the cutouts appear in the right layer and nowhere else. The
   Part/Layer panel makes per-layer isolation easy.
8. `pytest tests/` at the 145 passed / 1 xfailed baseline;
   `python -m verification.run_all` 9/9 (or 12/12 if the verification
   extension has landed).

---

## E. Pre-existing issues found while planning (flag separately, do not silently fix)

1. **The committed `.inp` deck is stale.**
   `examples/ex12_rigid_plate_display_fold.inp` has 120 element columns
   and plate RPs `10000`/`10094` (⇒ `plate_mesh_nx=30`), but current
   `DEFAULT_CONFIG` yields **240 columns** and `plate_mesh_nx=60`. The
   on-disk deck does not match the config.
2. **`DispFoldApp.py` regenerates the deck only if it is missing**
   (`:89-96`). Combined with (1), the "primary entry point" running its
   **default** `--mode inp` silently uses a stale mesh — and would
   silently ignore any new `layer_void_regions` config. Needs an
   unconditional regenerate or a `--regen` flag; this is a prerequisite
   for the void feature being usable from the default entry point.
3. **`--elem_jit` is a no-op on two paths**: `DispFoldApp.py:111-118`
   drops it when delegating to `run_build`/`run_roundtrip`, and
   `ex13_unified_model_io.py:261-264` declares its own `--elem_jit`
   argument that `main()` never reads.
4. **`README_folding_model.md` config table is stale**: lists
   `tip_dx` default 0.25 (actual 0.125) and `plate_mesh_nx` 30 (actual
   60).
5. **`LayerSlipTracker` probe fragility**
   (`dispsolver/postprocess/interlayer.py:112-117`): it derives
   `n_layers` from a single x-column and raises `ValueError` if that
   disagrees with `layer_materials`. Probes default to `x = ±40` (the
   tips), so hinge-region voids are safe — but a void at a probe column
   would raise at solve start. Worth a clearer error message once voids
   exist.
