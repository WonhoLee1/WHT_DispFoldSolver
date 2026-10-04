# Design: the .inp input pipeline as a compiler (2026-10-05)

Goal: a structure in which adding an Abaqus keyword is a small, local,
declarative change; in which the classes of defect found this week (silent
skips, name mismatches, duplicated dispatch tables, lost locations,
2D/3D drift) cannot be written; and in which every unsupported thing is
reported with its deck location.

## 1. Why the current shape keeps producing defects

The current reader works like this: a handler receives raw strings
(`block.params.get("ref node", block.params.get("ref_node", ...))`). It
mutates the model directly, reading reader-wide state (`current_part`,
`current_step`, `context`). Then the solvers interpret model objects with
`hasattr` and their own if-chains.

Every defect fixed this week traces back to one of four properties of that
shape:

| Property | Defects it produced |
|---|---|
| Raw strings in handlers | hyphenated names lost, aliases forgotten, param declared in a spec yet ignored by its handler (`*STEP, AMPLITUDE`) |
| Resolution scattered over handlers, flattening and solvers, each with its own "not found -> skip" | 29 silent skips (case, sets, sections, regions), PART_DEFAULT swallowing instance meshes |
| The same fact written in several places | element dispatch in 6 places (the catalog read by nobody), material ids clashing between 2D and 3D, 2D/3D load code drifting |
| Locations dropped after the lexer | errors at solve time with no deck line |

A design is "modern" here if it removes these properties structurally, not
if it adds checks on top of them.

## 2. Target: five stages, each with one job

The established shape for this problem is a compiler (lexer -> parser ->
AST -> semantic analysis -> IR -> backend), as in language servers or
schema-driven config loaders:

```
 .inp ──► 1 LEX ──► Block(kw, raw params, raw rows, Loc)
                │
          2 PARSE (per keyword, schema-driven, PURE) ──► Record (typed, frozen, Loc)
                │                                      + Diagnostics
          3 BIND (one pass, symbol tables) ──► Model (resolved references only)
                │                                      + Diagnostics
          4 LOWER (per step) ──► StepPlan (solver-agnostic arrays + evaluators)
                │
          5 EXECUTE ── DynamicSolver2D / 3D consume StepPlan only
```

### Stage 1: Lex (done, `abaqus_lexer.py`, F3)
Physical lines with `Loc(file, line)`, `*INCLUDE` expanded per file, keyword
continuation, indented keywords. Output: `Block`. No interpretation.

### Stage 2: Parse, declarative and pure
A keyword is one class. The schema is the code, so it cannot drift from
what the handler reads.

```python
@keyword("BOUNDARY", context=STEP | MODEL, support=FULL)
class Boundary(KeywordSpec):
    class Params(Schema):
        op: Literal["MOD", "NEW"] = "MOD"
        amplitude: Ref[Amplitude] | None = None           # resolved at BIND
        type: Literal["DISPLACEMENT"] = Param(unsupported=["VELOCITY", "ACCELERATION"])
        user: Flag = Param(unsupported=True)

    class Row(RowSchema):                                  # one data line
        region: NodeRegionRef                              # set name, INST.SET, node label
        first_dof: DofOrType                               # 1..6 or ENCASTRE/PINNED/XSYMM...
        last_dof: int | None = None
        value: float = 0.0

    def to_records(self, p: Params, rows: list[Row], loc) -> Iterable[Record]:
        for r in rows:
            yield BoundaryRecord(region=r.region, dofs=r.dof_range(), value=r.value,
                                 amplitude=p.amplitude, op=p.op, loc=r.loc)
```

- **Parameters and rows are validated before user code runs.** Types,
  enums, aliases, case, hyphen/blank/underscore spelling and defaults all
  come from the schema. An unknown parameter, an unsupported value or a bad
  row is a `Diagnostic(level, code, Loc, message)`. There is no `params.get`
  anywhere.
- **`to_records` is a pure function.** Input is typed values; output is
  frozen records. It does not touch the model or reader state, so it can be
  unit-tested from a three-line deck snippet.
- **Context is declared, not tracked by hand.** PART / ASSEMBLY / INSTANCE /
  STEP / MATERIAL scoping is a small stack in the driver; a record carries
  its scope (`scope=Instance("I1")`). The PART_DEFAULT bug class disappears
  because nothing "falls back" to a default part.

### Stage 3: Bind, one resolution pass with symbol tables
- Scoped, case-insensitive symbol tables (`model > part > instance`,
  `assembly`, `step`). Every `Ref[...]` in a record is resolved here, in
  one place, with Abaqus rules: qualified `INST.SET` / `ASM.INST.SET`,
  labels, ambiguity detection.
- Unresolved means `Diagnostic(ERROR, "unresolved-ref", loc)`, always. No
  resolver returns "nothing" to a caller that skips.
- Binding is order-independent where Abaqus allows forward references,
  because it runs after all parsing.
- Output: a model whose objects hold resolved handles (node index arrays,
  material objects), and each one keeps the `Loc` of the record that made it.

### Stage 4: Lower, Model + Step to StepPlan (solver-agnostic IR)
- `StepPlan` holds:
  - prescribed DOFs as `(dof_indices, value_fn(t))`;
  - load evaluators (`f_ext(t)` built from the load catalog: consistent
    integration, ramp/step amplitude per procedure);
  - constraint objects;
  - element groups `(kernel_id, rows, conn)` from the element catalog;
  - effective NLGEOM, contact mode.
- 2D and 3D share this stage; they differ only in the kernel tables. That
  removes the 2D/3D drift class.

### Stage 5: Execute
The solvers keep only the Newton/increment machinery and numerical
kernels. They never read model objects, set names or element-type strings.

## 3. Catalogs: data, not code paths

Every place where "Abaqus name -> behaviour" was an if-chain becomes a
registry of frozen descriptors, read by exactly one consumer and checked by
an integrity test:

| Catalog | Maps | Consumer |
|---|---|---|
| `ElementCatalog` + `KERNELS` | type -> kernel per NLGEOM / material / nu; kernel -> small-strain, co-rotated, elastic-only, node count | LOWER (element groups) |
| `MaterialCatalog` | `*ELASTIC` / `*PLASTIC` / `*HYPERELASTIC, type` -> material id + prop packer, per dimension | BIND |
| `LoadCatalog` | CLOAD / DLOAD type (P1..P6, BX, GRAV, ...) / DSLOAD -> evaluator | LOWER |
| `BoundaryCatalog` | ENCASTRE / PINNED / XSYMM ... -> dof sets | PARSE (row type) |
| `ContactCatalog` | enforcement × sliding × NLGEOM -> constraint class + mode | LOWER |

**Integrity tests (run in CI, fast):**
- Every registered keyword has a schema.
- Every catalog entry points to a kernel/evaluator that exists.
- For every element type × NLGEOM: the kernel the solver actually assembles
  equals the catalog's. This is the check the old catalog lacked.

## 4. Diagnostics as the only error channel

```python
@dataclass(frozen=True)
class Diagnostic:
    level: Literal["error", "gap", "warning", "info"]   # gap = physics not represented
    code: str                                           # "unresolved-ref", "unsupported-param", ...
    loc: Loc
    message: str
```

- PARSE, BIND and LOWER all append to one list. Nothing else warns, prints
  or raises for an input problem.
- Policy is applied once, at the end. `strict` refuses on any error or gap.
  Non-strict runs and attaches the list to the result.
- The CLI prints one table sorted by file:line. A language server or IDE
  could use the same list directly.

## 5. How a new keyword is added

1. Write one `@keyword` class: schemas plus `to_records`.
2. If it introduces a new physical object, add a catalog entry and its
   evaluator or kernel.
3. Add a three-line snippet test: deck -> records -> resolved model state ->
   StepPlan arrays.

Nothing else changes: no reader edits, no solver if-chains, no second
pipeline.

## 6. What already exists, and the migration

| Stage | Status |
|---|---|
| 1 LEX | done (F3): locations, includes, indentation, continuation |
| 2 PARSE | partial (F2): `ParamSpec`/`RowSpec` exist, but handlers still read raw strings and mutate the model |
| 3 BIND | partial: `regions.py` resolver and case-insensitive dicts; resolution is still also done inside handlers |
| 4 LOWER | missing: the solvers still build BCs/loads from model objects |
| Catalogs | `ElementCatalog` now drives the 3D solver (in progress); 2D, material, load and contact catalogs to do |
| Diagnostics | ad hoc: reader warnings, `physics_gaps()`, exceptions |
| Pipeline B (`AbaqusParser`/`ModelBuilder`) | still used by ex12; retired at the end via a Model -> ModelBuilderResult adapter |

Migration order (each step lands green and is A/B-checked against the
corpus sweep and the targeted solver tests):

- **M1.** `Diagnostic` + `Loc` types. Reader warnings and `physics_gaps()`
  become views of the diagnostic list. Small; it unblocks everything else.
- **M2.** Schema-driven PARSE framework (`Schema`/`RowSchema`, typed
  fields, `Ref[...]`). Migrate keywords one file at a time, highest-defect
  first:
  - BOUNDARY / CLOAD / DLOAD / DSLOAD;
  - NSET / ELSET / SURFACE;
  - sections;
  - materials;
  - the rest.

  During migration, old-style handlers keep running behind an adapter.
- **M3.** BIND pass with scoped symbol tables. Handlers stop resolving
  names.
- **M4.** `StepPlan` + `LoadCatalog`; both solvers consume it. Delete the
  per-solver BC/load code.
- **M5.** Element (finish 2D) / Material / Contact catalogs, each with its
  integrity test.
- **M6.** Retire pipeline B behind an A/B of the ex12 full fold.

Error-proofing comes from the structure: typed frozen records, one
resolution pass, one diagnostics channel, catalogs with integrity tests,
and the corpus sweep in CI as the regression net.
