# Material-registry refactor, viewer performance/UX overhaul, PSA re-tune — 2026-07-27

Continuation of the ex13/ex12 folding-model work. Six connected pieces
landed this session, in order: (1) material registry + dedup fix, (2)
console model-review print, (3) viewer redraw-speed rewrite, (4) viewer
view/interaction bug fixes + new UX, (5) part naming (Plate Left/Right)
+ CLI viewer flags, (6) PSA layer-order swap + re-tuned modulus.

## 1. Material registry + real dedup (`examples/fold_model_config.py`,
`examples/material_factory.py` (new), `examples/gen_ex12_inp.py`,
`dispsolver/io/model_builder.py`, `examples/ex13_unified_model_io.py`)

Previously PET/PSA were defined via three hardcoded dicts
(`MaterialsConfig.pet/.psa/.steel`), and `LayerSpec.material_family` was
a magic string matched via duplicated `if/elif` branches in
`gen_ex12_inp.py` and `ex13_unified_model_io.py::run_build()`. To give
each of the 14 physical layers its own pid, both files gave each layer
a **unique** `*MATERIAL` name (`PET_1..PET_7`, `PSA_1..PSA_7`) even
though all PET layers (and all PSA layers) are numerically identical —
the same material defined 7x over, purely to exploit
`model_builder.py`'s old "one pid per unique `*MATERIAL` name" rule.

- New `MaterialDef` dataclass (`id`/`name`/`type`/`params`);
  `MaterialsConfig.definitions: Dict[str, MaterialDef]` registry, keyed
  by name — a name now appears exactly once regardless of how many
  layers/sections reference it. `LayerSpec.material_family` renamed
  `material_name` (must be an exact registry key).
- New canonical type tags (`dispsolver/material/type_tags.py`):
  `J2_PLASTIC`/`ARRUDA_BOYCE_VISCO`/`NEOHOOKEAN`, importable from both
  `examples/` and `dispsolver/` (core never imports from examples/).
- New `examples/material_factory.py`: `build_material_instance(mdef)`
  (live object, used by `run_build()`) and `emit_abaqus_material_block(mdef)`
  (`.inp` text, used by `gen_ex12_inp.py`) — one dispatch on `mdef.type`
  each, replacing the two duplicated if/elif chains.
- **The real fix, `dispsolver/io/model_builder.py`**: `_build_sections()`
  now assigns one pid per `*SOLID SECTION` statement *unconditionally*
  (`self._material_name_to_pids: Dict[str, List[int]]`, one-to-many),
  instead of collapsing repeated material names onto one shared pid.
  `_build_materials()` builds each material object **once** per unique
  name and fans it out to every pid that references it. `gen_ex12_inp.py`
  now emits exactly **3** `*MATERIAL` blocks (was 15) and groups elements
  into one ELSET per **physical layer** (was one per mesh row).
  `ModelBuilderResult` gained `material_types`/`part_names` fields
  (see §5).
- Side effect (confirmed OK with user before implementing): `PLATE_LEFT`/
  `PLATE_RIGHT` went from sharing 1 pid to 2 distinct pids (each
  `*SOLID SECTION` now gets its own pid). `ex13_unified_model_io.py::run_build()`
  was later updated (§5) to match, so all three model sources (read/
  roundtrip/build) now have **pid-count parity for the first time**.

## 2. Console "MODEL REVIEW" print (`dispsolver/postprocess/model_review.py`
(new), `ex12_abaqus_inp_plate_fold.py`, `dispsolver/postprocess/viewer.py`)

`format_model_review()` + two adapters
(`print_model_review_from_builder_result`/`print_model_review_from_result`),
generic over plain pid-keyed dicts so the same formatter works from a
live `ModelBuilderResult`/`SimpleNamespace` and from a loaded `Result`.
Prints a "Materials (N distinct)" list (representative properties keyed
off the canonical type tag, not inferred from param-dict key presence)
and a "Parts/Layers" list (element count + label per pid). Called once
in `run_folding_from_result()` before the time-stepping loop (fires for
all 3 ex13 modes + direct ex12 invocation), and once in
`viewer.py::launch_from_result()` (not in the generic `load_result()`,
which non-interactive analysis scripts also use and shouldn't get
console spam).

## 3. Viewer redraw-speed rewrite (`dispsolver/postprocess/viewer.py`)

Every UI interaction (field dropdown, step/scale slider, layer
checkbox) was rebuilding the whole plot from scratch: a Python
`for e in range(solver.n_elem):` loop constructing one `Polygon` object
per element (~3480 elements), `ax.clear()`, and a brand-new colorbar —
this was the actual bottleneck, not matplotlib rendering itself.
Measured before/after with the same interactions:

| Interaction | Before | After |
|---|---|---|
| Layer checkbox toggle | (full rebuild every time) | 28 ms |
| Deformation Scale slider | (full rebuild every time) | 25 ms |
| Field dropdown (warm) | (full rebuild every time) | 31 ms |

Fix: `_compute_frame_arrays()` builds geometry (`coords_deformed[conn]`)
and colors (`vals[conn].mean(axis=1)`) fully vectorized (no per-element
Python object construction). `_get_elem_pids()` caches the pid array by
solver identity (pid never changes within a step). One persistent
`PolyCollection` (`self._pc`), colorbar (bound to a fixed
`self.canvas.cax` rect, not recreated), and hinge-marker `Line2D` are
created once; every subsequent call only does
`set_verts()`/`set_array()`/`set_clim()` — no `ax.clear()`, no object
churn. A separate, unchanged `_update_plot_overlay()` (full rebuild)
still handles the rare "Overlay Steps N>1" mode. Field-switch cost is
dominated by a one-time, pre-existing ~130s JAX JIT compile the first
time a stress/strain field is touched (unrelated to this rewrite,
cached for the rest of the session afterward).

## 4. View-drift bug fix + new interaction UX (`dispsolver/postprocess/viewer.py`)

**Bug**: zoom/pan reset on every field/step/layer-visibility change,
and the view sometimes didn't show the full plot at all. Root cause
(traced into the installed matplotlib 3.10.8 source,
`axes/_base.py::Axes.apply_aspect()`): `_configure_static_axes()` called
`ax.set_aspect('equal', adjustable='datalim')` — with `'datalim'`,
`apply_aspect()` overwrites the stored view limits from `ax.dataLim` on
**every draw**, autoscale-off notwithstanding (that branch only logs a
warning, it doesn't block the mutation). Since `PolyCollection.set_verts()`
changes `dataLim` every frame, the view silently re-fit every redraw.
Compounding it: `ax.set_box_aspect(1)` (also called) *forces*
`adjustable` back to `'datalim'` as an internal side effect regardless
of call order, so the two settings fought each other. Fix: drop
`set_box_aspect(1)` entirely, use `adjustable='box'` (only letterboxes
the rendered box within the fixed rect, never touches view limits), add
`ax.set_autoscale_on(False)` as a second independent guard. Since
matplotlib's aspect machinery no longer keeps the box square for us, a
new `_fit_view()` method does it manually (equalize x/y span, center on
data midpoint) — called once on first draw and from the new "Fit View"
button.

New: mouse-scroll zoom (cursor-anchored, `factor = 0.9 ** event.step`,
scales both xlim/ylim ranges by the same factor to preserve the
equal-span invariant), middle-button drag pan (pixel-to-data delta via
`ax.get_window_extent()`), both wired via `canvas.mpl_connect(...)`
alongside (not replacing) the stock `NavigationToolbar2QT`. "Fit View"
button in the control panel. Part/Layer checkbox list wrapped in a
`QScrollArea` (was overflowing the fixed-width panel once 16 parts
exist). UI font unified to Cascadia Code 9pt — `matplotlib.rcParams`
set at module import time (before any `Figure`/text artist exists) plus
`QApplication`/window-level `setFont()` for the Qt widgets.

## 5. Part naming: "Plate Left"/"Plate Right" (`dispsolver/io/model_builder.py`,
`examples/ex13_unified_model_io.py`, `ex12_abaqus_inp_plate_fold.py`,
`dispsolver/postprocess/viewer.py`, `model_review.py`)

New `part_names: Dict[int, str]` field (pid -> human display-name
override; unset for ordinary display layers, which keep the existing
"Layer N (material)" fallback). `model_builder.py::_build_sections()`
derives it from the elset name (`sec.elset` starting with `"PLATE_"` ->
`"Plate " + suffix.title()`, e.g. `PLATE_LEFT` -> `"Plate Left"`).
`ex13_unified_model_io.py::run_build()` previously gave both plates one
*shared* pid (`STEEL_PID`) — changed to give each side (`plates.items()`
iteration order) its own pid, reusing the same built STEEL material
instance/params for both (materials are stateless, same pattern as
§1), and setting `part_names[pid] = f"Plate {side.capitalize()}"`. This
also achieves the pid-count parity noted in §1 (`build` mode: 15 -> 16
pids). Threaded through `ResultWriter`'s meta (`"pid_part_names"`,
additive/backward-compatible) and both `model_review.py` adapters.

Also added CLI flags to `ex13_unified_model_io.py`: `--viewer` (open
the Qt viewer automatically once the solve finishes) and `--open
PKL_PATH` (skip solving, just open an existing saved result). Needed
`run_folding_from_result()`'s return dict to carry `"result_path"`
(one-line addition — the path was already computed locally for
`writer.save()`).

## 6. PSA layer order swap + modulus re-tune (`examples/fold_model_config.py`)

Per user request: `geometry.layer_pattern` reordered so layer 1 = PSA,
layer 2 = PET (was PET-first) — pure reorder of the `LayerSpec` list,
no other code touched (both `gen_ex12_inp.py` and `run_build()` read
`layer_pattern` directly, so the single config change propagates to
both model sources automatically — exactly the point of §1's registry
work).

PSA `mu`/`K` re-derived for a target small-strain modulus
**E=0.05 MPa** (previously E≈0.5 MPa, per the existing `nu=0.49`
near-incompressible convention documented in this repo's history —
confirmed by reverse-checking the old `K=8.3333` value against
`K=E/(3*(1-2*nu))`, exact match at E=0.5). Isotropic relations at fixed
`nu=0.49`:
```
mu = E / (2*(1+nu))   = 0.05 / 2.98      = 0.016779
K  = E / (3*(1-2*nu)) = 0.05 / 0.06      = 0.83333
```
Both `mu` and `K` are linear in `E` for fixed `nu`, so the new values
are exactly the old ones / 10. `lambda_m` (locking stretch, an assumed
shape parameter, not measured — see AGENTS.md 1.4), Prony, and WLF
terms unchanged.

## Verification

- `.inp` regenerated (`gen_ex12_inp.py`): confirmed pid 1/2/3/4.../
  material_names now alternate `PSA, PET, PSA, PET, ...` (was
  `PET, PSA, ...`); `part_names` = `{15: "Plate Left", 16: "Plate
  Right"}`; PSA `material_params` = `{mu: 0.016779, K: 0.83333, ...}`.
- All 3 `ex13_unified_model_io.py --mode {read,roundtrip,build}` runs
  (post §1-§5, pre-§6 layer/PSA change): identical `nodes=3697
  elements=3480 rbe2_constraints=2 penalty_constraints=2
  reached_target=True`, zero cutbacks in all three logs — full
  cross-mode parity including the new 16-pid split.
- `pytest tests/`: 145 passed, 1 xfailed (unchanged baseline this
  session established via `git stash` A/B — the AGENTS.md-documented
  "135 passed/10 failed" note predates several since-fixed issues and
  is stale).
- `python -m verification.run_all`: 9/9 PASS.
- Qt viewer manual smoke test (headless `PostprocessViewer` construction
  + `launch_from_result` with `QApplication.exec` monkeypatched to
  avoid blocking): 16 Part/Layer checkboxes present and correctly
  labeled ("Plate Left"/"Plate Right" + "Layer N (PET/PSA)"), font/
  rcParams applied (`Cascadia Code`, 9pt, confirmed on both the Qt
  `QApplication` and `matplotlib.rcParams`), review text prints once at
  launch.
- §6 (layer-order swap + PSA E=0.05 MPa): `--mode build` re-run to
  completion -- `nodes=3697 elements=3480 rbe2_constraints=2
  penalty_constraints=2 reached_target=True`, zero cutbacks, full
  90°/side closure with the softer PSA (10x lower modulus) and swapped
  layer order. No solver/mesh changes needed.
