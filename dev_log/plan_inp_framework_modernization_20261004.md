# Plan: input-keyword framework modernization (2026-10-04)

Scope: the path from an `.inp` file to the objects the solver runs on --
tokenizing, keyword handling, the internal model, flattening, and the
hand-off to the 2D/3D solvers -- reviewed for structure, extensibility and
how well the pieces connect. Evidence comes from today's keyword audit
(`dev_log/keyword_behavior_audit_20261004.md`) and the NLGEOM element/contact
probes (section 6).

## 1. What exists today

Front end (shared): `io/abaqus_lexer.py` tokenizes into `AbaqusKeywordBlock`
(keyword, lower-case param keys, raw data lines, line number) and expands
`*INCLUDE`.

After that there are **two independent pipelines**:

| | Pipeline A ("reader") | Pipeline B ("parser/builder") |
|---|---|---|
| Handling | `AbaqusInputReader` + `KeywordRegistry` decorators (`io/keywords/*.py`, 68 handlers) | `AbaqusParser._handlers` dict (~35 methods) |
| Intermediate | `model.Model` (CAE-style: Part / Assembly / Instance / Step / Load / BC / Interaction) | `io/abaqus_model.AbaqusModel` (flat dataclasses) |
| Flatten | `Assembly.build_solver_system` -> `FlattenedSolverSystem` | `ModelBuilder.build` -> `ModelBuilderResult` |
| Solver | `DynamicSolver2D` / `DynamicSolver3D` (Numba) | legacy `solver/dynamic.DynamicSolver` (JAX/Numba 2D) |
| Users | `execute_inp`, `run_inp` CLI, tests | **ex12 (primary fold reference)**, `fold_model_config`, element-contract tests |

Every fix from today's audit lives in pipeline A only. Pipeline B has its
own case handling, set resolution, BC/load parsing and defaults.

## 2. Structural problems (ranked by how many defects they caused)

P1. **"Unresolved means skip" is the default everywhere.** A name, region,
material or amplitude that does not resolve returns `None`/`[]`, and the
caller skips it. Today's audit found 29 silent-wrong cases, and almost all
reduce to this. Today's fix (one resolver plus raises) is local; the
framework still has no rule that enforces it.

P2. **No parameter or data-line schema.** Each handler pulls
`block.params.get(...)` ad hoc (101 call sites) and parses positional
columns by hand. Unknown parameters are silently ignored: `*CONTACT PAIR,
SMALL SLIDING`, `*SOLID SECTION, CONTROLS=`, `*STEP, INC=` and similar have
no effect and produce no warning. `physics_gaps()` cannot see them.
`_parse_data_rows` drops blank fields, which shifts columns (41 call sites).

P3. **Declared support metadata is dead.** `KeywordSpec` has `support`
(FULL / PARTIAL / STUB / IGNORED), `products`, `level` and `slug` fields,
but `register()` accepts none of them, so every keyword reports FULL. The
output-only/control classification I added today is a separate hard-coded
set.

P4. **The element catalog is in five places**: `assembly.ABAQUS_ELEM_MAP`,
the 3D kernel if-chains (NLGEOM YES and NO), the 2D kernel if-chains,
`_ELEMENT_SUBSTITUTIONS`, and the legacy `_ELEMENT_LARGE_DEF` /
`model_builder.ABAQUS_ELEMENT_MAP`. Nothing ties a type name to its
formulation, its NLGEOM behavior or its supported materials. Section 6
shows the result: a third of the element types silently do not switch
between small and large deformation.

P5. **Loads and BCs are interpreted inside the solvers**, duplicated in 2D
and 3D. Dispatch is by duck typing (`hasattr(load, "get_acceleration_vector")`).
The 2D and 3D copies had drifted: different ramps, different amplitude
handling, body loads silently dropped in 2D.

P6. **The model is mutated during reading, with implicit state.** Handlers
write straight into `reader.model` and lean on `reader.context`,
`current_part`, `current_step` and similar. Context rules are partly in
`valid_contexts` and partly in `parse_blocks` special cases.

P7. **Source locations are lost.** `line_number` stops at the block; model
objects carry no `(file, line)`. So a solve-time error ("BC X region does
not resolve") cannot point at the deck line. `*INCLUDE` expansion also
shifts line numbers.

P8. **Mixed strictness.** `strict` used to govern only the context check;
handler errors became warnings. That is fixed for execution
(`require_complete`), but the reader still collects silently by default.

P9. **Model -> solver hand-off is long procedural code.**
`create_solver3d` / `create_solver2d` / `solve()` each apply BCs, build
interactions and apply predefined fields, partly duplicated (interactions
are built in both `create_solver3d` and `solve`). NLGEOM is applied in
`solve()` for 3D but never in 2D (section 6).

## 3. Target architecture

```
.inp ──lexer──► Blocks (file, line) ──► KeywordSpec.parse(block) ──► typed Record ──► ModelAssembler ──► Model
                                            ▲ schema: params, rows,         (validates refs,           │
                                            │ contexts, support level       source-located errors)    ▼
                                         one registry                                         Compiler (per solver)
                                                                                                   │
                                     ElementCatalog / MaterialCatalog / LoadCatalog ──────────────►│
                                                                                                   ▼
                                                                                FlattenedSolverSystem + StepPlan
```

1. One front end. Pipeline B is replaced by an adapter
   `Model -> ModelBuilderResult` so ex12 / legacy `DynamicSolver` keep
   running on the same reader. The `AbaqusParser` and `AbaqusModel` code
   is then deleted.
2. Declarative keyword specs. Each keyword declares its parameters (name,
   type, enum, default, `supported: bool`), its data-row layout
   (positional fields with types; a blank is a real value), valid contexts
   and support level. The lexer output is validated against the spec before
   any handler runs. Unknown or unsupported parameters become gaps
   automatically.
3. Records, then assembly. A handler returns a typed record (`BoundaryRecord`,
   `SectionRecord`, ...) that carries its source location. The
   `ModelAssembler` places records into the `Model` and resolves references
   with one resolver. Every unresolved reference is an error with
   `file:line`.
4. Catalogs as the single source of truth.
   - `ElementCatalog`: Abaqus type name -> {dimension, family, nodes,
     integration, hybrid/incompatible flags, kernel for NLGEOM=NO, kernel for
     NLGEOM=YES, supported materials, substitution note}. The solvers' kernel
     dispatch, `detect_dim`, the substitution warnings and the contract tests
     all read it. A type without both kernels declared either cannot be used
     under that NLGEOM, or is wrapped explicitly (generic CR).
   - `MaterialCatalog`: keyword -> material id, so 2D/3D share one id table
     (today's id-1 clash came from two tables).
   - `LoadCatalog`: `CLOAD` / `DLOAD` / `DSLOAD` / GRAV / BX -> one load
     evaluator (ramp, amplitude, region, consistent integration) used by both
     solvers.
5. A compiler per solver. `Model + Step -> StepPlan` (fixed dofs as functions
   of time, load evaluators, constraint objects, effective NLGEOM, element
   kernel groups). The solvers consume the plan; they no longer interpret
   model objects.
6. One strictness policy. Gaps are collected with locations. `strict`
   (default for execution) refuses; non-strict prints a gap report and tags
   the results.

## 4. Phased plan (each phase ships green on its own)

| Phase | Content | Exit test |
|---|---|---|
| F0 | Already done today: `CaseInsensitiveDict`, region resolver, `physics_gaps` / `require_complete`, `detect_dim`, keyword-behavior tests. | `tests/test_inp_keyword_behavior.py` |
| F1 | **ElementCatalog** + NLGEOM correctness (section 6): table-driven kernel choice in 2D/3D, generic CR wrapper, small-strain kernels for NO, 2D per-step kernel rebuild. Comes first because it fixes wrong answers, not only structure. | per-type contract: rigid rotation ≈0 under YES; linear and equal to YES at infinitesimal strain under NO; catalog ↔ dispatch integrity check (as `_assert_dispatch_integrity`) |
| F2 | **Keyword schema**: `register()` takes params/rows/support; migrate handlers file by file. Unknown params become gaps. Fix `_parse_data_rows` blanks. Populate support levels; the gap classification reads them. | schema coverage test: every registered keyword has a spec; corpus sweep gap counts are reported, never silently lower |
| F3 | **Source locations**: blocks carry (file, line) through `*INCLUDE`; records and model objects keep them; resolver errors cite them. | error-message tests with line numbers |
| F4 | **LoadCatalog + StepPlan**: one load/BC evaluator for 2D and 3D; `create_solver*` and `solve()` stop duplicating BC/interaction/field setup. | 2D/3D load-vector equality tests on twin decks |
| F5 | **Retire pipeline B**: `Model -> ModelBuilderResult` adapter; switch ex12 and `fold_model_config`; delete `AbaqusParser`, `AbaqusModel`, `ModelBuilder`. | ex12 full fold reproduces the reference (90°/side, zero cutbacks) bit-close; contract tests green |
| F6 | **MaterialCatalog**: one id table; the 2D dict-material path uses it. | material id round-trip test |

Risk notes
- F5 touches the primary reference (ex12). Do it last, behind an A/B on
  the full fold result.
- F2 will raise the gap count of the corpus, because ignored params now
  show up. That is the point; report the numbers rather than tune them down.
- Each phase must keep the 2D/3D solvers' numerical paths bit-identical
  except where a phase fixes a measured defect.

## 5. Contact (task #5) -- fits F1/F4

- C1: `*CONTACT PAIR` drops `SMALL SLIDING`, and the model default is SMALL.
  The Abaqus default is finite sliding.
- C2: PENALTY and NITSCHE ignore `sliding` (projection is always frozen);
  only mortar honors FINITE.
- C3: no contact class reads the effective NLGEOM. All use current
  coordinates and current normals. For NLGEOM=NO small sliding, Abaqus keeps
  the contact plane in its initial configuration; check this against the
  Abaqus Analysis User's Guide before implementing.
- These become fields of the StepPlan (`nlgeom`, `sliding`) passed to the
  constraint at bind time. Nitsche already follows the element kernel
  through `element_point_stress`.

## 6. Element NLGEOM probe (evidence for F1)

One element, linear elastic. `rot` = |f_int| under a 30° rigid rotation /
|f| of a 1 % stretch: about 0 means objective (large-def), O(1) means small
strain. `lin` = f(10u)/f(u): exactly 10 means linear.

3D (`DynamicSolver3D`, kernel rebuilt per NLGEOM):

| type | YES kernel / rot | NO kernel / lin | verdict |
|---|---|---|---|
| C3D8 | 1 / 2e-13 | 10 / 10.000 | OK |
| C3D8R, C3D8RH | 13 / 2e-13 | 8 / 10.000 | OK |
| C3D8I, C3D8IH | 11 / 2e-13 | 0 / **11.14** (rot 1e-14) | NO runs finite-strain TL |
| C3D8H | 2 / 2e-13 | **2** (same kernel) | NO not honored |
| C3D20, C3D20R, C3D20H | 15 / **25** | 15 / 10.000 | YES runs small strain |
| C3D10, C3D10H | 5 / **25** | 5 | YES runs small strain |
| C3D10M | 4 / **25** | 4 | YES runs small strain |
| C3D4 | 6 / **25** | 6 | YES runs small strain |
| C3D4H | 9 / **25** | 9 | YES runs small strain |
| C3D6 | 7 / **25** | 7 | YES runs small strain |

2D (`DynamicSolver2D`, constructed with the step's NLGEOM):

| type | YES | NO | verdict |
|---|---|---|---|
| CPE4, CPE4I, CPE4H, CPE4R, CPE6M | CR, rot ≈1e-14 | small strain, lin 10 | OK |
| CPE3, CPE6, CPE8, CPE8R | same kernel, rot **25** | same | YES runs small strain |
| CAX4, CAX4R, CAX8 | same kernel 17/15 | same | NLGEOM not distinguished (rigid-rotation test does not apply to axisymmetry; needs a translation/hoop check) |

On top of that, 2D deck runs (`execute_inp`, `run_inp`) build the solver
without a step and never rebuild kernels, so **every 2D deck runs the
NLGEOM=YES kernels**, even with `NLGEOM=NO`. `DynamicSolver2D` has no
equivalent of the 3D `_build_kernel_groups` re-run.
