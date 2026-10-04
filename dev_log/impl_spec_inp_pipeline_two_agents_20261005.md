# Implementation spec: .inp pipeline modernization, split between two agents (2026-10-05)

Read first, in this order:
1. `dev_log/design_inp_pipeline_20261005.md`: the target architecture
   (LEX -> PARSE -> BIND -> LOWER -> EXECUTE, catalogs, one diagnostics
   channel). This spec does not repeat its rationale.
2. `AGENTS.md`: project rules. In particular:
   - §3: run `python -m verification.run_all` after any change under
     `dispsolver/solver|element|material|constraint|mesh`;
   - §4.19: a numba `cache=True` caller keeps running a stale inlined callee
     until the caller file is touched;
   - known-failing baselines: do not report them as new failures.
3. `dev_log/keyword_behavior_audit_20261004.md`: the 29 silent-wrong defects
   this work must make structurally impossible.

Two agents work in parallel:
- **Agent A** (the session that wrote this spec) owns the critical path:
  diagnostics, the parser framework, binding, the step plan, and wiring
  everything into the solvers.
- **Agent B** owns separable work behind fixed interfaces: the load and
  material catalogs as pure, tested functions, the corpus regression
  harness, the migration of material keywords onto A's framework, and the
  retirement of the legacy pipeline.

---------------------------------------------------------------------------

## 0. Working agreement (both agents)

### 0.1 Branches and worktrees
- A works on the main checkout, branch `feat/solver-element-abaqus-surpass-20260830`.
- B works in its own worktree on its own branch, created from A's HEAD at
  hand-off:
  ```
  git worktree add ../wht_agentB -b feat/inp-pipeline-agentB
  ```
  B never edits the main checkout.
- B delivers by committing on its branch and telling the user the commit
  ids. A merges B's branch at the sync points (§4). B rebases onto A's
  branch when told a sync point has landed.

### 0.2 File ownership
An agent edits only files it owns. Anything else is a request to the
owner (through the user), not an edit. New files are owned by the agent
that creates them, and must be inside that agent's paths below.

| Path | Owner |
|---|---|
| `dispsolver/io/**` (lexer, reader, keywords, registry, diagnostics, schema, bind) | A |
| `dispsolver/io/keywords/material.py`, **only the material keyword handlers, after sync S2** | B (A keeps the section handlers in that file until B migrates; see B4) |
| `dispsolver/model/regions.py`, `element_catalog.py`, `model.py`, `assembly.py`, `step.py`, `load.py` | A |
| `dispsolver/model/material.py` | B |
| `dispsolver/solver2d/**`, `dispsolver/solver3d/**`, `dispsolver/solver/step_executor.py` | A |
| `dispsolver/loads/**` (new package) | B |
| `dispsolver/material/catalog.py` (new), `dispsolver/material3d/numba_materials.py` | B |
| `tools/inp_corpus/**`, `tests/inp_corpus/**` (new) | B |
| `dispsolver/io/legacy_adapter.py` (new), `dispsolver/io/abaqus_parser.py`, `abaqus_model.py`, `model_builder.py`, `examples/ex12_*`, `examples/ex13_*` | B (task B5 only) |
| `tests/test_<topic>.py` | the agent whose task the test covers; new files only, or files it owns |
| `GEMINI.md`, `dev_log/area_analysis_*`, `dev_log/lessons_learned_contact_harmonization_20261004.md` | **nobody** (another tool edits them) |
| `LESSONS_LEARNED.md` (repo root) | both, **append-only**, a new numbered section per lesson |

### 0.3 Engineering rules
- **No silent fallback, ever.** Anything the code cannot represent becomes
  a `Diagnostic` (§1.1) with a `Loc`, or an exception with the location in
  its message. Never default and continue. Never `continue` past input
  you did not handle.
- **Abaqus semantics must be sourced.** Cite the local manual copies in
  `benchmark_element/reference_abaqus_docs/*.txt` (or the Abaqus manual
  section) in the docstring of anything that encodes a convention: sign of
  a pressure, face numbering, axisymmetric per-radian conventions, data-line
  layouts.
- **Numerics change only on purpose.** If a change can alter a computed
  number, A/B it against the parent commit on the targeted tests, and state
  the delta in the commit message.
- **Tests are pure where the code is pure.** Catalog functions get unit
  tests with closed-form answers (patch tests: uniform pressure -> p·A·n,
  body force -> b·V), with no solver construction needed.
- **Shell pitfall in this environment:** backslashes inside bash heredocs
  are altered. Write patch scripts with the file tool, then run them.
- Before every commit: run the targeted tests for the touched area, plus
  the corpus sweep (§B3) once it exists. When touching
  solver/element/material/constraint/mesh, also run
  `python -m verification.run_all` (current baseline 13/14; Gulati fails
  and is pre-existing).
- Commit messages: `type(scope): summary`, a body that says what was wrong
  before and how it was verified, ending with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Commit only
  the files you own, by path (`git commit -- <paths>`).

---------------------------------------------------------------------------

## 1. Interface contracts (frozen; changes only by agreement)

### 1.1 Diagnostics: `dispsolver/io/diagnostics.py` (A writes it in A1, the first commit after hand-off)

```python
@dataclass(frozen=True)
class Loc:
    file: str          # path as given; display uses basename
    line: int          # 1-based physical line
    def __str__(self): ...          # "deck.inp:12"

Level = Literal["error", "gap", "warning", "info"]
# error: the deck is wrong or cannot be read            -> always refuses to solve
# gap:   valid Abaqus input this solver does not represent -> strict refuses, non-strict runs and reports
# warning: represented, but with a documented substitution (e.g. C3D8RH -> C3D8R)
# info:  noteworthy, no effect on the model

@dataclass(frozen=True)
class Diagnostic:
    level: Level
    code: str          # kebab-case, stable: "unresolved-ref", "unsupported-param", "unsupported-keyword",
                       # "unsupported-value", "bad-row", "missing-param", "element-substitution", ...
    message: str
    loc: Optional[Loc] = None

class Diagnostics:
    def add(self, level, code, message, loc=None) -> None
    def error(self, code, message, loc=None) -> None      # shorthands for each level
    def gap(self, code, message, loc=None) -> None
    def warning(...); def info(...)
    def of(self, *levels) -> list[Diagnostic]
    def raise_if_blocking(self, strict: bool) -> None   # errors always; gaps too when strict
    def __iter__ / __len__
```

B consumes it (corpus harness, catalogs' "unsupported" reporting). B does
not change it.

### 1.2 Load catalog: `dispsolver/loads/` (B implements in B1; A consumes in A4)

```python
# dispsolver/loads/integration.py   -- pure numpy, no solver/model imports
def body_force_weights(X: np.ndarray, elem_type: str) -> np.ndarray:
    """integral(N_a dOmega) for each node a of one element in its reference
    configuration. 3D: volume. 2D plane: area times unit thickness.
    Axisymmetric (CAX*): the convention the solver's CAX kernels use for
    internal forces -- B must read dispsolver/element2d/cax*.py and the
    Abaqus docs and state which (per radian or full ring) in the docstring.
    Every type the element catalog lists as runnable must be covered,
    INCLUDING quadratic elements (C3D10/C3D20/C3D15?/CPE6/CPE8/CAX8), whose
    consistent weights are not an equal split (corner terms negative or zero).
    Unknown type -> raise NotImplementedError(f"... {elem_type}")."""

def face_pressure_forces(X: np.ndarray, elem_type: str, face: int, p: float
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Consistent nodal forces of a uniform pressure p on face `face`
    (Abaqus face numbering, 1-based) of one element in its reference
    configuration. Returns (local node indices on the face, forces (k, dim)).
    Sign: positive p pushes INTO the element (Abaqus). Faces per Abaqus
    element tables: C3D8/C3D20 S1..S6, C3D4/C3D10 S1..S4, C3D6 S1..S5,
    CPE4/CPE8/CAX4/CAX8 S1..S4, CPE3/CPE6 S1..S3."""

def face_area_normal(X, elem_type, face) -> tuple[float, np.ndarray]   # for tests and diagnostics

# dispsolver/loads/catalog.py
@dataclass(frozen=True)
class LoadTypeSpec:
    keyword: str                 # "DLOAD" | "DSLOAD" | "CLOAD"
    load_type: str               # "P1".."P6", "P", "BX", "BY", "BZ", "GRAV", "CENTRIF", ...
    kind: Literal["face_pressure", "body_force", "gravity", "nodal", "unsupported"]
    face: Optional[int] = None
    component: Optional[int] = None   # BX -> 0 ...
    reason: Optional[str] = None      # for unsupported
LOAD_TYPES: dict[tuple[str, str], LoadTypeSpec]
def load_type(keyword: str, load_type: str) -> LoadTypeSpec   # unknown -> kind="unsupported"
```

- The solver's existing helpers are the oracle B must match exactly where
  they overlap: `dynamic3d._volume_shape_integrals` (C3D8/C3D4/C3D6),
  `_integrate_face_pressure_quad4`, `_integrate_face_pressure_quad8`,
  `model/interaction._HEX_FACE_LOCAL_NODES`, and the 2D
  `solver/traction.SurfaceTraction`.
- B does not edit the solvers. A replaces the solver code with calls into
  this package in A4.

### 1.3 Material catalog: `dispsolver/material/catalog.py` (B implements in B2; A consumes in A5)

```python
@dataclass(frozen=True)
class MaterialSpec:
    mat_id: int                 # the numba id (numba_materials.MAT_*)
    name: str                   # "LINEAR_ELASTIC", "NEOHOOKEAN", "J2_PLASTICITY", ...
    linear_elastic: bool
    dims: frozenset[int]        # {2, 3} where the 2D/3D dispatch really implements it
    needs_F: bool               # uses the deformation gradient (not only small strain)
MATERIALS: dict[int, MaterialSpec]
def material_id(material: "dispsolver.model.material.Material") -> int     # one mapping, used by 2D and 3D
def pack_props(material, dim: int) -> np.ndarray                          # the props vector the kernels read
def is_linear_elastic(mat_id: int) -> bool
```

- **Integrity test:** every branch of `material_dispatch_3d` and
  `material_dispatch_2d` maps to exactly one `MaterialSpec`, and vice
  versa. Id 1 is NEOHOOKEAN in both dimensions. Today
  `solver2d/dynamic2d.py` has its own dict-material mapping, and unknown
  `mat_type` silently becomes linear elastic there (audit O7).
- B lists every place that currently maps material keywords to ids
  (`model/material.to_numba_props`, `solver2d` / `solver3d` material
  setup, `io/model_builder`) in the B2 commit message, with any
  disagreement found. B fixes those only inside its own files; the
  solver-side switch is A5.

### 1.4 Schema framework: `dispsolver/io/schema.py` (A writes it in A2; B uses it in B4)
The exact API is published in A2's commit and in the module docstring.
Shape, for planning:

```python
@keyword("ELASTIC", context={MATERIAL}, support="FULL")
class Elastic(KeywordSpec):
    class Params(Schema):
        type: Literal["ISOTROPIC", "ENGINEERING CONSTANTS", "ORTHOTROPIC", "ANISOTROPIC"] = "ISOTROPIC"
        dependencies: int = 0
        moduli: Literal["LONG TERM", "INSTANTANEOUS"] = Param(unsupported=True)
    class Row(RowSchema):
        E: float; nu: float; temperature: float | None = None
    def to_records(self, p, rows, ctx) -> Iterable[Record]: ...
```

Records are frozen dataclasses carrying `loc`. Handlers never receive raw
`block.params`, never mutate the model and never read reader state.

---------------------------------------------------------------------------

## 2. Agent A tasks (critical path; this session implements them)

| # | Task | Done when |
|---|---|---|
| A0 | Finish `ElementCatalog` as the only element dispatch: 3D wired (in progress), 2D `_build_kernel_groups` -> `ElementCatalog.resolve`, connectivity trimming / co-rotation / elastic-only all from `KERNELS`; equivalence check against the pre-catalog chains for every type × NLGEOM × nu × material class; catalog <-> dispatch integrity test | the old chains are deleted; equivalence differences are only the intended ones (shells refused, CPE4IH notes); `test_element_nlgeom_contract`, `test_inp_keyword_behavior`, deck06 green |
| A1 | `io/diagnostics.py` (§1.1); the reader writes into `Diagnostics`; `physics_gaps()` / `require_complete()` / CLI banner become views of it | all reader warnings carry `Loc`; existing tests green; **sync S0** |
| A2 | `io/schema.py` framework + adapter so legacy handlers keep running; migrate BOUNDARY, CLOAD, DLOAD, DSLOAD, NSET, ELSET, SURFACE, SOLID/SHELL SECTION, PART / ASSEMBLY / INSTANCE / STEP scope keywords, STATIC / DYNAMIC, AMPLITUDE | migrated handlers contain no `params.get`; snippet tests per keyword; corpus sweep shows no new errors; **sync S2** (the framework is published, with two migrated keywords as reference examples, early in A2) |
| A3 | BIND pass: scoped case-insensitive symbol tables; every `Ref[...]` resolved there; `regions.py` becomes its backend | no name resolution left in handlers or solvers; unresolved -> `error` with `Loc` |
| A4 | `StepPlan` (LOWER) shared by 2D and 3D; BC / load application via `dispsolver/loads` (B1); delete per-solver `_update_step_bcs` / `_compute_step_external_loads` bodies | 2D and 3D produce identical `f_ext` on twin decks; deck06 / Hertz / reader tests green; `run_all` 13/14; **sync S3** |
| A5 | Wire `MaterialCatalog` (B2) into 2D/3D material setup; `ContactCatalog` (enforcement × sliding × NLGEOM -> class, mode) | integrity tests green; no material-id mapping left outside the catalog |
| A6 | Merge B's branch at each sync point; final corpus and regression report | — |

---------------------------------------------------------------------------

## 3. Agent B tasks (hand this section to Agent B)

**Context for B:** you work in worktree `../wht_agentB` on branch
`feat/inp-pipeline-agentB`. Read §0 (rules, ownership) and §1 (the
contracts you implement or consume) before starting. Do not edit files you
do not own; ask through the user instead. Each task ends with a commit on
your branch. Report its id and the verification output to the user.

### B1. Load catalog (`dispsolver/loads/`): start immediately
- Implement §1.2. Cover:
  - every element type that `ElementCatalog` marks runnable in 2D or 3D;
  - every DLOAD / DSLOAD type the corpus uses, as a supported or
    unsupported `LoadTypeSpec`.

  Get the type list from `tools/inp_corpus` once B3 exists, or by grep in
  `benchmark_inp/`.
- Quadratic elements: consistent weights by Gauss quadrature of the
  element's own shape functions. Hard-coded "corner = -V/20"-style
  constants are allowed only if a test derives them.
- Tests: new file `tests/test_load_catalog.py`, pure, no solver. For each
  type:
  - Σ body weights = volume (or area, or the axisymmetric measure);
  - Σ face forces = p · A · n for each face, with n pointing out of the
    element and the force inward for p > 0;
  - exact match with the solver oracles listed in §1.2 where they overlap;
  - distorted (non-affine) elements: weights positive where theory says so,
    sums still exact.
- Done when the tests pass and `ruff check dispsolver/loads` is clean.
  Deliver -> **sync S1** (A wires it in A4).

### B2. Material catalog (`dispsolver/material/catalog.py`): start immediately
- Implement §1.3 and its integrity test, in new file
  `tests/test_material_catalog.py`.
- Audit and list every existing material -> id mapping and every
  disagreement between them. Fix the ones that live in your files
  (`model/material.py`, `material3d/numba_materials.py`).
- Numba caution (AGENTS.md §4.19): if you edit `numba_materials.py`, touch
  every `cache=True` caller file afterwards. Run the element/material tests
  and `python -m verification.run_all`.
- Done when the integrity test passes, no numeric change is reported by the
  targeted tests unless intended and stated, and `run_all` is 13/14.
  Deliver -> **sync S1**.

### B3. Corpus regression harness (`tools/inp_corpus/`, `tests/inp_corpus/`): start immediately
- `tools/inp_corpus/sweep.py`:
  - parse every `.inp` under `benchmark_inp/` with
    `dispsolver.io.abaqus_reader._parse_source(path, dim=detect_dim(path), strict=False)`,
    in a process pool;
  - record per deck: `parse_ok`, the exception class and message, sorted
    unique diagnostic codes with counts (after S0; before that, reader
    warnings and `physics_gaps()`), element types, and the steps' NLGEOM
    flags;
  - write JSON.

  A one-off prototype exists at
  `C:\Users\GOODMAN\.claude\jobs\284d3b9f\tmp\sweep_f2.py`: use the idea,
  not the code.
- `tests/inp_corpus/baseline.json` committed. Test
  `tests/inp_corpus/test_corpus_baseline.py`:
  - **fast mode** (default, < 60 s): a fixed, representative subset of
    about 150 decks, chosen by a deterministic rule recorded in the file;
  - **full mode** (`-m corpus_full`): all decks;
  - fails if any deck gains an error/gap code, or stops parsing, compared
    with the baseline;
  - `python -m tools.inp_corpus.sweep --update-baseline` rewrites the
    baseline, and the diff is then reviewed in the commit.
- Register the `corpus_full` marker in the pytest config. Do not change
  other settings.
- Done when the fast test runs green on the current branch head and the
  baseline is committed. A and B both run it before every commit from then
  on.

### B4. Migrate the material keywords onto A's schema framework: after sync S2
- Keywords: MATERIAL, ELASTIC, EXPANSION, PLASTIC, CREEP, HYPERELASTIC,
  VISCOELASTIC, RATE DEPENDENT, DAMAGE INITIATION, DAMAGE EVOLUTION,
  DRUCKER PRAGER (+ HARDENING), CONCRETE * (5), MULLINS EFFECT, DENSITY, TRS,
  ORIENTATION.
- Follow A's two reference migrations exactly: `Params` / `Row` schemas,
  a pure `to_records`, no `params.get`, every unsupported option declared
  `Param(unsupported=...)`.
- Material records map to ids **only** through B2's `material_id` /
  `pack_props`.
- One snippet test per keyword in `tests/test_material_keywords.py`:
  deck snippet -> record -> resulting `Material` (id, props).
- Done when the corpus baseline (B3) shows no new error/gap codes, except
  ones that expose a real unsupported option (list them in the commit
  message), and the material tests are green.

### B5. Retire the legacy pipeline: after sync S3
- `dispsolver/io/legacy_adapter.py`: `model_to_builder_result(model) ->
  ModelBuilderResult`, built from the new reader's `Model`, holding what
  `read_abaqus_input` returns today.
- Switch `dispsolver.io.read_abaqus_input` to new reader + adapter.
- A/B against the old pipeline:
  - `examples/ex12_abaqus_inp_plate_fold.py` full run (AGENTS.md §1.0
    criteria: 90°/side reached, zero cutbacks, iteration counts, final
    displacement field within 1e-8 relative);
  - `tests/element_contract`;
  - `examples/ex13_unified_model_io.py`, all modes.
- Only after the user approves the A/B report: delete `abaqus_parser.py`,
  `abaqus_model.py`, `model_builder.py` and their tests, in a separate
  commit.

---------------------------------------------------------------------------

## 4. Sync points

| Sync | Trigger | Effect |
|---|---|---|
| S0 | A1 committed (diagnostics) | B3 switches from reader warnings to `Diagnostics`; B rebases |
| S1 | B1 and B2 committed | A merges B's branch; A4 and A5 consume them |
| S2 | A2 framework + reference migrations committed | B rebases and starts B4 |
| S3 | A4 committed (StepPlan) | B rebases and starts B5 |
| Final | all tasks done | A merges, runs the full corpus (`-m corpus_full`), deck06, Hertz, element contract, `run_all`, and reports |

If a contract in §1 turns out to be wrong, the agent that finds it stops
using the wrong part, writes the needed change into its commit message or a
note to the user, and the owner of the contract changes it. Neither agent
works around a contract in its own code.

---------------------------------------------------------------------------

## 5. Verification commands (copy-paste)

```
# targeted, fast
python -m pytest tests/test_inp_keyword_behavior.py tests/test_source_location.py tests/test_keyword_schema.py -q
python -m pytest tests/test_element_nlgeom_contract.py tests/test_contact_nlgeom_sliding.py -q
python -m pytest tests/inp_corpus -q                       # after B3
# solver-side (when touching solver/element/material/constraint/mesh)
python -m verification.run_all                             # baseline 13/14 (Gulati pre-existing)
python -m pytest tests/test_deck06_contact.py -q           # ~7 min; 21 passed, 3 xfailed expected
# full corpus
python -m pytest tests/inp_corpus -m corpus_full -q        # after B3
```

Known pre-existing failures (do not report as regressions):
- `tests/test_deformable_soft_nonlinear_contact.py::test_augmented_lagrangian_now_wired_for_deformable_hard`
- `tests/test_dual_mortar_2d.py::test_uzawa_augmented_lagrangian_contraction`
- `verification.run_all`: Gulati two-point bending.
