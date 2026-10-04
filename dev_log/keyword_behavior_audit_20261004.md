# Keyword -> solver behavior audit (2026-10-04)

Question: when a deck says something, does the solver actually do it? Not
"does the reader store it" -- the reader stored almost everything. The
audit traces each keyword to what the solver holds when it solves: fixed
dofs, the external load vector, the material and kernel of every element,
the constraint objects.

Method
- Probe decks (2-element C3D8 bar, two-part assembly, 2D quad): read, build
  the solver, evaluate the step's BCs and loads at the end of the step, print
  the resolved state. No solving needed for most cases.
- Corpus sweep: parse all 1895 decks in `benchmark_inp/` (1859 Abaqus
  verification decks plus project decks). For each deck, check which audited
  failure class it would trigger. Parse and model resolution only, about one
  minute.
- Everything that was wrong is now a regression test in
  `tests/test_inp_keyword_behavior.py` (27 cases). Those tests also check
  solver state, not reader state.

The common pattern behind the defects: an unresolvable reference returned
"nothing", and every caller treated "nothing" as "skip this one". The run
continued, converged, and printed a result for a different model.

## Defects found and fixed

| # | Keyword / path | Was | Now |
|---|---|---|---|
| 1 | All names (sets, surfaces, materials, sections, amplitudes, parts, instances) | Case-sensitive dicts. `*NSET, NSET=LEFT` + `*BOUNDARY` on `left` meant the BC was skipped. Also caused the 78-deck `'Assembly' object has no attribute 'sets'` handler errors (`eall` vs `EALL`). | `CaseInsensitiveDict` (`dispsolver/model/names.py`): keeps the original spelling, matches regardless of case. |
| 2 | BC / CLOAD / DLOAD region (2D + 3D solvers, `Model.create_solver*`) | Five private resolvers that disagreed. Unresolved region: BC/load silently not applied. Bare node number: tried the solver's internal id first, so it hit the wrong node whenever deck labels are not 1..N. | One resolver, `dispsolver/model/regions.py`. Unresolved region raises `RegionError`. Numbers are deck labels. |
| 3 | Bare part-level set name in a multi-instance model | Aliased to the first instance that defined it, so a BC meant for part B went on part A. | Alias only when exactly one instance defines it. Otherwise `RegionError` with the qualified names. |
| 4 | `*SOLID/*SHELL SECTION` naming | `SEC_SOLID_<elset>` was model-wide. Abaqus CAE writes `_PickedSet2` in every part, so the last part's material overwrote every other part's. `official/cook_2d.inp` and `cook_3d.inp` have this. | `SEC_<kind>_<part>_<elset>`. |
| 5 | `*SOLID SECTION` with unknown ELSET | Handler error became a warning; the elements got a made-up material (E=4000, nu=0.3). Also searched other parts for the ELSET and took the first match. | Inside a part: must be that part's ELSET. Outside a part: exactly one owning part, otherwise error. Elements without a section raise at system build (deck models). |
| 6 | Section referring to an undefined material | Default material E=4000, no warning. | Raises at system build. |
| 7 | `*NSET/*ELSET` member set name that does not exist | Dropped from the set. | Raises (handler error, so a physics gap). |
| 8 | Assembly `*NSET/*ELSET, INSTANCE=` | Written into the instance's PART, so every other instance of that part got it too. Assembly set labels were treated as global ids. | Stored as (instance, label) members, mapped per instance when flattening. |
| 9 | `*DLOAD GRAV` | g/8 per node per element over the whole mesh: no density, no volume (10^7x too large for steel in mm-t-s units). Region ignored. A blank region shifted the columns and the load was skipped. | ρ·∫N dV·g on the given element set. A blank region means the whole model. Missing `*DENSITY` raises. |
| 10 | `*DLOAD BX/BY/BZ` | Equal split over all nodes using the first-8-node volume, wrong for 20-node bricks. | Consistent ∫N dV for 4-, 6- and 8-node solids. Other element types raise. |
| 11 | `*AMPLITUDE` | Only the first pair per data line was read, so `0,0, 1,1` became the single point (0,0) and the amplitude was a constant 0 (353 corpus decks write several pairs per line). `TIME=TOTAL TIME` and `DEFINITION=PERIODIC` etc. were silently treated as tabular step time. An undefined amplitude name meant scale 1 (BC) or ramp. | Every pair is read. Unsupported definitions or time bases are physics gaps. An undefined name raises. |
| 12 | Pressure with an amplitude object | Amplitude applied twice (`get_magnitude` and the solver both multiplied). | Applied once. |
| 13 | `*BOUNDARY, TYPE=VELOCITY/ACCELERATION` | Imposed as a displacement. | Physics gap. |
| 14 | `*BOUNDARY` unparsable line | `continue`: the line was dropped. | Physics gap. |
| 15 | `*CLOAD` dof 4-6 | Stored, then dropped. | Physics gap. |
| 16 | `*TEMPERATURE` with `*EXPANSION` | Thermal strain silently dropped: nothing passes deck temperatures to the solver. | `NotImplementedError` (wiring `DynamicSolver3D.set_temperature` is open item O3). |
| 17 | Unknown 3D element types (C3D8T, C3D27, typos, ...) | Kernel chosen by node count. | `NotImplementedError`. |
| 18 | C3D8RH, C3D8IH, C3D20R/H/RH, C3D10H, C3D10MH, C3D4H | Ran on kernels missing the hybrid field or reduced integration; nothing said so. | Run as before, but each substitution is a `UserWarning` and recorded in `solver.element_substitutions`. |
| 19 | 2D deck through `execute_inp` | `dim` defaulted to 3. A CPE4 deck reached the 3D solid-shell kernel and crashed in numba (access violation). | `detect_dim()` reads the element types. CPE4/CPS4 removed from the 3D shell list. |
| 20 | `execute_inp`, single step without loads | One-shot `solve_step()`: the whole prescribed displacement in one increment, ignoring `*STATIC` incrementation. Path-dependent results differed. | Every step runs through the step engine. `iterations`, `increments` and `max_increment_iterations` are reported. |
| 21 | 2D `*CLOAD` / pressure in STATIC | Full load from the first increment (3D ramped). Pressure ignored `AMPLITUDE=<name>`. Body/gravity loads matched no branch and were ignored. | Ramped like 3D. Amplitude honored. Body loads raise in 2D. |
| 22 | 2D plane stress (CPS*) | No 2D kernel implements σ33=0, so CPS* ran as plane strain. | `NotImplementedError` unless the material is linear elastic with nu=0 (identical answer). |
| 23 | 2D unknown element types | Kernel by node count. | `NotImplementedError`. |
| 24 | 2D material id 1 | `material_dispatch_2d` treated ids 0 **and 1** as linear elastic. Id 1 is `MAT_HYPERELASTIC_NEOHOOKEAN`, so every 2D Neo-Hooke material ran linear elastic. `solver2d` also used 1 as its own "elastic" id. | Only 0 is linear elastic. `solver2d` uses 0. |
| 25 | `*TIE`, `*RIGID BODY`, `*MPC` model records reaching the solver | No `assemble()`. 2D skipped them (bodies silently untied); 3D crashed in Newton with `AttributeError`. | `add_constraint` refuses them with `NotImplementedError` (3D tie implementation is open item O1). |
| 26 | Deck constraint whose references do not resolve | Warning, then dropped from assembly. | Raises (deck models). |
| 27 | Keyword handler that raised | Downgraded to a warning, even with `strict=True`. Unsupported keywords were only listed. The CLI printed "100% Keywords Handled" when only warnings were present. | `reader.physics_gaps()` lists unsupported non-output keywords plus handler errors. `execute_inp(strict=True)` (the default) refuses with `UnsupportedDeckError`. `run_inp` prints a MODEL GAPS banner, or refuses with `--strict`. Output-only keywords (`*EL PRINT`, `*NODE FILE`, `*RESTART`, ...) and solution controls are not gaps. |
| 28 | `Model.create_solver*(step_name=...)` BC loop | `if entry.is_active:` tested the bound method, which is always truthy, so deactivated BCs were applied. | `entry.is_active()`. |
| 29 | Fixture `tests/fixtures/simple_quad.inp` | No material, no section, `*CLOAD` outside any step. It "converged" with E=4000 and no load. | Valid deck. |

## Corpus numbers (1895 decks, parse only)

- 1906 of 3236 `*STEP`s (59%) omit NLGEOM. 1030 decks omit it in every step.
- 353 decks have multi-pair `*AMPLITUDE` lines (#11).
- 243 decks had elements falling to the default material (#4-#6).
- 123 decks had handler errors that were downgraded (#27). The biggest
  groups were the case mismatch (#1) and `<parameter>` placeholders from
  unsupported `*PARAMETER`.
- 136 decks are `*IMPORT` restarts with no mesh. They now show up as a
  physics gap instead of a numpy broadcast error.

## Pending decision: NLGEOM default

`*STEP` without NLGEOM is read as NLGEOM=YES. The Abaqus/Standard default is
NO. The step engine honors whatever the step says (8ce8844); the question is
only what an omitted parameter means.

Measured on the decks that omit it in every step (1030):
- Only 73 of them have no physics gaps, i.e. could run at all: 20 2D and 53
  3D. The rest are refused anyway for unsupported keywords.
- Of the 53 3D decks, 25 combine C3D8 with a nonlinear material. Under
  NLGEOM=NO a C3D8 runs on kernel 10, which only supports one linear-elastic
  material and refuses everything else. Flipping the default would turn
  those 25 runnable decks into `NotImplementedError` until a small-strain
  C3D8 kernel with general material dispatch exists.
- In-repo decks affected: 7 `benchmark_inp/official` decks (eref3ksg,
  thickcompcyl_8el, hertzcontact_2d_rsurf, beamgap, radstr_holcyl_cax4r,
  anisoplate_s9r5, compositeshells_c3d20r), 3 test fixtures, and inline decks
  in 5 test files. Abaqus's own reference results for those decks are
  NLGEOM=NO results.

## Open items

- O1: 3D `*TIE`. `constraint3d/surface_tie3d.py` (penalty) exists but is not
  wired to model `Tie` records. It needs surface-to-face resolution and a
  patch test (uniform stress across the tie). Abaqus ties are exact MPCs.
- O2: small-strain C3D8 with general materials (kernel 10 is linear elastic
  only). This blocks the NLGEOM default flip.
- O3: wire `*TEMPERATURE` / `*INITIAL CONDITIONS, TYPE=TEMPERATURE` to
  `set_temperature` (uniform fields first). Check the thermal-strain
  reference temperature against Abaqus: α(T)(T−T0) − α(TI)(TI−T0).
- O4: 2D plane stress kernels, and 2D section thickness. Thickness is
  ignored, so an out-of-plane thickness other than 1 scales nothing.
  *Not yet refused* -- see O4 note below.
- O5: `_parse_data_rows` drops empty fields (positional blanks shift
  columns). Fixed locally for `*DLOAD`. 41 call sites still use it.
- O6: `*STATIC` without a max increment defaults to dt_max=0.05. Abaqus's
  default is the step period. This is incrementation (performance), not
  model correctness, but it forces at least 20 increments.
- O7: the 2D solver's dict-material path maps unknown `mat_type` values to
  linear elastic.

O4 note: section thickness is dropped in `build_solver_system`, so no 2D
consumer ever sees it. Refusing it there would break every 2D deck that
writes the default `1.` explicitly. It needs a value check (≠ 1.0) at the
same place the section is resolved. Planned with O4, not done here.
