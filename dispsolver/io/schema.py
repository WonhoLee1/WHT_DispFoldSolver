"""Schema-driven keyword parsing (the PARSE stage).

A keyword is one class. Its parameters and data lines are typed schemas, and a
pure `to_records` turns them into frozen records. Records reach the model only
through an applier registered for the record type -- the one place that may
touch the reader (the BIND pass replaces appliers later).

    @keyword("BOUNDARY", support="FULL")
    class Boundary(Keyword):
        class Params(Schema):
            op: Literal["MOD", "NEW"] = "MOD"
            amplitude: Optional[str] = None
            type: Literal["DISPLACEMENT", "VELOCITY", "ACCELERATION"] = Param(
                "DISPLACEMENT", supported=("DISPLACEMENT",))
            user: bool = Param(False, unsupported="user subroutine DISP")

        class Row(RowSchema):
            region: str
            first: str
            last: Optional[int] = None
            value: Optional[float] = None

        def to_records(self, p, rows, ctx):
            ...yield BoundaryRecord(..., loc=ctx.loc)

    @applies(BoundaryRecord)
    def _apply(reader, rec): ...

What the framework decides, so no keyword writes it again
---------------------------------------------------------
Parameter names are resolved against the schema plus the Abaqus keyword index
(`keyword_index.py`), with Abaqus's abbreviation rule (exact, else unique
prefix; '_' / '-' / blanks are equivalent). Then:

    declared, supported            -> converted to the field's type
    declared, Param(unsupported=)  -> gap  'unsupported-param'   (block skipped)
    declared, Param(control=True)  -> info 'control-param-ignored' (ignored)
    not declared, in the index     -> gap  'unsupported-param'   (block skipped)
    in neither (undocumented)      -> gap  'unknown-param'      (block skipped)
    ambiguous abbreviation         -> error 'ambiguous-param'
    required field missing         -> error 'missing-param'     (block skipped)
    bad value / not a choice       -> error 'bad-value'         (block skipped)
    choice outside Param(supported=) -> gap 'unsupported-value' (block skipped)

An undeclared parameter Abaqus defines is a gap automatically: a keyword does
not have to list every option it lacks to be honest about it. A gap refuses
the whole block rather than dropping the one parameter -- *AMPLITUDE with its
SCALEX ignored would be a different curve, not a smaller one. A parameter that
genuinely cannot change the model (an output or solution-technique switch) is
declared Param(control=True) on purpose. A declared field
the index does not know is refused at registration (`Param(undocumented=True)`
to override, with a reason) -- the schema cannot drift from Abaqus.

Field types: str, int, float, bool (a flag: present -> True; YES/NO accepted),
Literal[...] (choices, resolved by abbreviation), Optional[X]. Row fields are
positional; a final `Tuple[X, ...]` field takes the remaining columns (a blank
cell is None there, so positions are kept).
`Data = Flat(float, group=2)` instead of `Row` reads all data values as one
sequence in groups (e.g. *AMPLITUDE time/value pairs, up to four per line).

Inside `to_records`: raise `Unsupported(msg)` for valid input this solver does
not represent (gap) and `InputError(msg)` for wrong input (error); both refuse
the whole block. To report without refusing, yield `Note(level, code, msg)`. `ctx` is a
frozen `KeywordCtx` -- the reader itself is not reachable from here.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Optional, Tuple, Type, Union

from dispsolver.io.diagnostics import Loc
from dispsolver.io.keyword_index import AmbiguousName, IndexKeyword, abaqus_index, norm, resolve_name

_MISSING = object()


# ── errors raised inside to_records ─────────────────────────────────────────

class Unsupported(NotImplementedError):
    """Valid Abaqus input this solver does not represent (reported as a gap)."""


class InputError(ValueError):
    """The deck is wrong (reported as an error)."""


# ── field declarations ──────────────────────────────────────────────────────

class Param:
    """Optional metadata for a schema field; the first argument is the default."""

    __slots__ = ("default", "name", "unsupported", "control", "supported", "undocumented",
                 "abaqus_default")

    def __init__(self, default: Any = _MISSING, *, name: Optional[str] = None,
                 unsupported: Optional[str] = None, control: bool = False,
                 supported: Optional[Tuple[str, ...]] = None, undocumented: Optional[str] = None,
                 abaqus_default: Optional[str] = None):
        self.default = default
        self.name = name                    # Abaqus spelling when it is not the attribute name
        self.unsupported = unsupported      # reason; its presence is a gap and refuses the block
        self.control = control              # solution control: reported as info and ignored
        self.supported = supported          # the subset of choices this solver implements
        self.undocumented = undocumented    # reason a name absent from the Abaqus index is accepted
        self.abaqus_default = abaqus_default  # reason this default differs from the Keywords Guide on purpose


@dataclass(frozen=True)
class FieldSpec:
    attr: str
    abq: str                                # Abaqus name, normalized
    kind: str                               # str int float bool choice rest
    choices: Tuple[str, ...] = ()
    elem: str = ""                          # element kind of a 'rest' field
    optional: bool = False
    default: Any = _MISSING
    unsupported: Optional[str] = None
    control: bool = False
    supported: Optional[Tuple[str, ...]] = None
    undocumented: Optional[str] = None
    abaqus_default: Optional[str] = None

    @property
    def required(self) -> bool:
        return self.default is _MISSING and not self.optional


_SCALARS = {str: "str", int: "int", float: "float", bool: "bool"}


def _classify(tp) -> Tuple[str, Tuple[str, ...], str, bool]:
    """(kind, choices, rest element kind, optional) of an annotation."""
    origin = typing.get_origin(tp)
    if origin is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) != 1:
            raise TypeError(f"unsupported union {tp}")
        kind, choices, elem, _ = _classify(args[0])
        return kind, choices, elem, True
    if origin is typing.Literal:
        return "choice", tuple(norm(c) for c in typing.get_args(tp)), "", False
    if origin in (tuple, Tuple):
        args = typing.get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis and args[0] in _SCALARS:
            return "rest", (), _SCALARS[args[0]], False
        raise TypeError(f"only Tuple[X, ...] is supported, got {tp}")
    if tp in _SCALARS:
        return _SCALARS[tp], (), "", False
    raise TypeError(f"unsupported field type {tp}")


class _Typed:
    """Immutable value object built from annotated class attributes."""

    _fields: ClassVar[Tuple[FieldSpec, ...]] = ()

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        hints = typing.get_type_hints(cls)
        specs = []
        for attr, tp in hints.items():
            if attr.startswith("_") or typing.get_origin(tp) is ClassVar:
                continue
            kind, choices, elem, optional = _classify(tp)
            raw = cls.__dict__.get(attr, _MISSING)
            meta = raw if isinstance(raw, Param) else Param(raw)
            default = meta.default
            if default is _MISSING and optional:
                default = None
            if kind == "rest" and default is _MISSING:
                default = ()
            if kind == "choice" and isinstance(default, str):
                default = norm(default)
            supported = tuple(norm(s) for s in meta.supported) if meta.supported else None
            if supported and kind == "choice" and not set(supported) <= set(choices):
                raise TypeError(f"{cls.__qualname__}.{attr}: supported {supported} not among {choices}")
            specs.append(FieldSpec(attr, norm(meta.name or attr), kind, choices, elem, optional,
                                   default, meta.unsupported, meta.control, supported,
                                   meta.undocumented, meta.abaqus_default))
        kinds = [s.kind for s in specs]
        if "rest" in kinds[:-1]:
            raise TypeError(f"{cls.__qualname__}: a Tuple[X, ...] field must be the last field")
        cls._fields = tuple(specs)

    def __init__(self, **values):
        for s in self._fields:
            object.__setattr__(self, s.attr, values.get(s.attr, None if s.default is _MISSING else s.default))
        object.__setattr__(self, "given", frozenset(values.pop("_given", ())))
        object.__setattr__(self, "loc", values.pop("_loc", None))

    def __setattr__(self, k, v):
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __repr__(self) -> str:
        body = ", ".join(f"{s.attr}={getattr(self, s.attr)!r}" for s in self._fields)
        return f"{type(self).__name__}({body})"

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and all(
            getattr(self, s.attr) == getattr(other, s.attr) for s in self._fields)

    __hash__ = None  # type: ignore[assignment]


class Schema(_Typed):
    """Keyword-line parameters. `given` holds the Abaqus names present in the deck."""


class NoParams(Schema):
    pass


class RowSchema(_Typed):
    """One data line, fields in column order; `loc` is that line's location."""


@dataclass(frozen=True)
class Flat:
    """All data values as one sequence, split into groups of `group`."""
    type: type = float
    group: int = 1


# ── records ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Record:
    """Base of every parse record; subclasses are frozen dataclasses."""
    loc: Optional[Loc] = field(default=None, kw_only=True, compare=False)


_APPLIERS: Dict[type, Callable[[Any, Record], None]] = {}


def applies(record_type: type):
    """Register the function that lands `record_type` records in the model."""
    def deco(fn):
        if record_type in _APPLIERS:
            raise ValueError(f"two appliers for {record_type.__name__}")
        _APPLIERS[record_type] = fn
        return fn
    return deco


@dataclass(frozen=True)
class Note(Record):
    """A diagnostic decided while parsing, landed in input order with the
    other records (to_records cannot report directly: it never sees the reader)."""
    level: str
    code: str
    message: str


def _apply_note(reader, rec: Note) -> None:
    reader.diagnostics.add(rec.level, rec.code, rec.message, rec.loc)


_APPLIERS[Note] = _apply_note


def apply_record(reader, rec: Record) -> None:
    fn = _APPLIERS.get(type(rec))
    if fn is None:
        raise RuntimeError(f"no applier registered for {type(rec).__name__}")
    fn(reader, rec)


# ── keyword classes ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class KeywordCtx:
    """What a keyword may know about where it stands; never the reader."""
    keyword: str
    context: str                            # KeywordContext name: GLOBAL PART ASSEMBLY INSTANCE STEP MATERIAL
    step: Optional[str] = None              # current step name (None: model data)
    dim: int = 3
    loc: Optional[Loc] = None


@dataclass(frozen=True)
class Lines:
    """The data lines as text, stripped (e.g. the *STEP subheading)."""


class Keyword:
    name: ClassVar[str] = ""
    Params: ClassVar[Type[Schema]] = NoParams
    Row: ClassVar[Optional[Type[RowSchema]]] = None
    Data: ClassVar[Optional[Union[Flat, Lines]]] = None
    # A scope keyword (*STEP, *PART, ...) is applied even when a parameter is
    # a gap or an error: refusing it would push every following keyword out of
    # its context. The problem is still reported; the parameter keeps its default.
    refuse_on_gap: ClassVar[bool] = True
    # A keyword that opens a scope names the keyword that closes it. If the
    # opener is refused anyway (a missing required parameter, a bad data line),
    # the reader skips the scope's body up to that keyword with one diagnostic
    # instead of reading it in the wrong context.
    closes_with: ClassVar[Optional[str]] = None

    def to_records(self, p: Schema, rows: list, ctx: KeywordCtx) -> Iterable[Record]:
        raise NotImplementedError


Report = Callable[..., None]     # (level, code, message, param=<Abaqus name> for parameter problems)


def parse_params(kw: str, schema: Type[Schema], params: Dict[str, str], report: Report,
                 where: str = "", entry: Optional[IndexKeyword] = None,
                 refuse: bool = True) -> Optional[Schema]:
    """Parameters of one keyword line -> a Schema instance, or None when the
    block must be skipped (an error, or a choice this solver does not implement).
    With refuse=False only a missing required parameter skips the block."""
    if entry is None:
        entry = abaqus_index().get(kw)
    by_abq = {s.abq: s for s in schema._fields}
    names = set(by_abq) | (set(entry.params) if entry else set())
    at = f" at {where}" if where else ""
    values: Dict[str, Any] = {}
    given: List[str] = []
    ok = True
    for key, raw in params.items():
        try:
            abq = resolve_name(key, names)
        except AmbiguousName as e:
            report("error", "ambiguous-param", f"*{kw}{at}: parameter {e}", param=key.upper())
            ok = False if refuse else ok
            continue
        if abq is None:
            # not in the Keywords Guide; Abaqus accepts some undocumented
            # parameters (*VISCOELASTIC, FORMULATION= in its own verification
            # decks), so this is a gap, not proof the deck is wrong
            report("gap", "unknown-param", f"*{kw}{at}: {key.upper()} is not a documented parameter of *{kw}",
                   param=key.upper())
            ok = False if refuse else ok
            continue
        if abq in given:
            report("warning", "duplicate-param", f"*{kw}{at}: {abq} given more than once; the last setting is used")
        given.append(abq)
        s = by_abq.get(abq)
        if s is None:
            report("gap", "unsupported-param", f"*{kw}, {abq}{at} (unsupported parameter)", param=abq)
            ok = False if refuse else ok
            continue
        if s.unsupported:
            report("gap", "unsupported-param", f"*{kw}, {abq}{at} (unsupported parameter: {s.unsupported})",
                   param=abq)
            ok = False if refuse else ok
            continue
        if s.control:
            report("info", "control-param-ignored", f"*{kw}, {abq}{at} (solution control, ignored)", param=abq)
            continue
        try:
            v = _convert(s, raw)
        except (ValueError, AmbiguousName) as e:
            report("error", "bad-value", f"*{kw}{at}: {abq}={raw}: {e}")
            ok = False if refuse else ok
            continue
        if s.kind == "choice" and s.supported is not None and v not in s.supported:
            report("gap", "unsupported-value", f"*{kw}{at}: {abq}={v} is not implemented "
                                               f"(supported: {', '.join(s.supported)})")
            ok = False if refuse else ok
            continue
        values[s.attr] = v
    for s in schema._fields:
        if s.required and s.attr not in values and not (s.unsupported or s.control):
            report("error", "missing-param", f"*{kw}{at}: missing required parameter '{s.abq}'")
            ok = False
    if not ok:
        return None
    return schema(**values, _given=given)


_TRUE = {"YES", "ON", "TRUE"}
_FALSE = {"NO", "OFF", "FALSE"}


def _scalar(kind: str, raw: str):
    t = str(raw).strip()
    if kind == "str":
        return t
    if kind == "float":
        return float(t)
    if kind == "int":
        f = float(t)
        if not f.is_integer():
            raise ValueError("expected an integer")
        return int(f)
    raise TypeError(kind)


def _convert(s: FieldSpec, raw: str):
    if s.kind == "bool":
        u = norm(raw)
        if u in _TRUE:
            return True
        if u in _FALSE:
            return False
        raise ValueError("expected a flag (no value) or YES/NO")
    if s.kind == "choice":
        v = resolve_name(raw, s.choices)
        if v is None:
            raise ValueError(f"expected one of {', '.join(s.choices)}")
        return v
    return _scalar(s.kind, raw)


def _split(line: str) -> List[str]:
    parts = [c.strip() for c in line.split(",")]
    while parts and parts[-1] == "":
        parts.pop()
    return parts


def parse_rows(kw: str, row: Type[RowSchema], lines: List[str], locs: List[Any],
               report: Optional[Report] = None) -> list:
    """Data lines -> RowSchema instances. Raises InputError on a bad line.
    Values beyond the documented items are reported (warning) and ignored --
    official Abaqus verification decks carry such lines and Abaqus runs them."""
    out = []
    fields_ = row._fields
    for text, loc in zip(lines, locs):
        cols = _split(text)
        if not cols:
            continue
        vals: Dict[str, Any] = {}
        rest = fields_ and fields_[-1].kind == "rest"
        fixed = fields_[:-1] if rest else fields_
        if not rest and len(cols) > len(fixed):
            if report is None:
                raise InputError(f"*{kw} at {loc}: {len(cols)} values, at most {len(fixed)} expected: {text!r}")
            report("warning", "extra-data-items",
                   f"*{kw} at {loc}: {len(cols)} values, {len(fixed)} expected; extra ignored: {text!r}")
            cols = cols[:len(fixed)]
        for s, c in zip(fixed, cols):
            if c == "":
                continue
            try:
                vals[s.attr] = _convert(s, c) if s.kind in ("bool", "choice") else _scalar(s.kind, c)
            except (ValueError, AmbiguousName) as e:
                raise InputError(f"*{kw} at {loc}: {s.attr}={c!r}: {e}") from None
        if rest:
            s = fields_[-1]
            try:
                # a blank cell keeps its position (None): '..., 9.81, , -1.' is (9.81, None, -1.)
                vals[s.attr] = tuple(None if c == "" else _scalar(s.elem, c) for c in cols[len(fixed):])
            except ValueError as e:
                raise InputError(f"*{kw} at {loc}: {s.attr}: {e}") from None
        missing = [s.attr for s in fixed if s.required and s.attr not in vals]
        if missing:
            raise InputError(f"*{kw} at {loc}: missing {', '.join(missing)} in {text!r}")
        out.append(row(**vals, _loc=loc))
    return out


def parse_flat(kw: str, data: Flat, lines: List[str], where: str = "") -> List[tuple]:
    kind = _SCALARS[data.type]
    try:
        flat = [_scalar(kind, c) for ln in lines for c in _split(ln) if c != ""]
    except ValueError as e:
        raise InputError(f"*{kw} at {where}: {e}") from None
    if len(flat) % data.group:
        raise InputError(f"*{kw} at {where}: {len(flat)} values, not a multiple of {data.group}")
    return [tuple(flat[i:i + data.group]) for i in range(0, len(flat), data.group)]


def _data_locs(block) -> List[Optional[Loc]]:
    base = Loc.of(block)
    nums = list(getattr(block, "data_line_numbers", []) or [])
    out = []
    for i in range(len(block.data_lines)):
        out.append(Loc(base.file, nums[i]) if base is not None and i < len(nums) else base)
    return out


def run_keyword(cls: Type[Keyword], block, ctx: KeywordCtx, report: Report) -> List[Record]:
    """PARSE one block: parameters, data, to_records. Pure; returns the records
    (empty when the block was refused, with the reason reported)."""
    p = parse_params(cls.name, cls.Params, block.params, report, str(block.location or ""),
                     refuse=cls.refuse_on_gap)
    if p is None:
        return []
    locs = _data_locs(block)
    if cls.Row is not None:
        rows = parse_rows(cls.name, cls.Row, block.data_lines, locs, report)
    elif isinstance(cls.Data, Lines):
        rows = [ln.strip() for ln in block.data_lines if ln.strip()]
    elif cls.Data is not None:
        rows = parse_flat(cls.name, cls.Data, block.data_lines, str(block.location or ""))
    else:
        if any(ln.strip() for ln in block.data_lines):
            raise InputError(f"*{cls.name} at {block.location}: takes no data lines")
        rows = []
    out: List[Record] = []
    try:
        for rec in cls().to_records(p, rows, ctx):
            out.append(rec)
    except Exception:
        # the block is refused, but what it had already decided to report stands
        for rec in out:
            if isinstance(rec, Note):
                report(rec.level, rec.code, rec.message)
        raise
    return out


# ── registration ────────────────────────────────────────────────────────────

SCHEMA_KEYWORDS: Dict[str, Type[Keyword]] = {}


def check_against_index(cls: Type[Keyword]) -> List[str]:
    """Schema field names Abaqus does not define for this keyword."""
    entry = abaqus_index().get(cls.name)
    if entry is None:
        return [f"*{cls.name} is not an Abaqus keyword"]
    return [f"*{cls.name}, {s.abq}" for s in cls.Params._fields
            if s.abq not in entry.params and not s.undocumented]


# Defaults the framework filled from the index / left alone, for the report
# and tests/test_abaqus_defaults.py: (keyword, parameter, value | reason).
DEFAULT_FILLS: List[Tuple[str, str, Any]] = []
DEFAULT_SKIPS: List[Tuple[str, str, str]] = []


def _index_default(s: FieldSpec, raw: str):
    """The index's documented default as the field's own type (ValueError when
    the text is not a value of that type)."""
    if s.kind == "bool":
        u = norm(raw)
        if u in _TRUE:
            return True
        if u in _FALSE:
            return False
        raise ValueError(f"not a flag value: {raw!r}")
    if s.kind == "choice":
        v = resolve_name(raw, s.choices)
        if v is None:
            raise ValueError(f"not one of {s.choices}: {raw!r}")
        return v
    if s.kind == "rest":
        raise ValueError("a Tuple field has no parameter default")
    return _scalar(s.kind, raw)


def apply_abaqus_defaults(cls: Type[Keyword]) -> List[str]:
    """Make every declared parameter default the one the Keywords Guide
    documents (user rule 2026-10-05: an omitted item takes the Abaqus default,
    never an in-house one).

        field default == documented     -> fine
        field default differs           -> returned as a problem, unless the
                                           field carries Param(abaqus_default=reason)
        Optional field, documented fixed default -> the documented value is filled in
        documented default depends on the usage (several different values over
        the keyword's usages) -> left alone and listed in DEFAULT_SKIPS

    `p.given` still says which parameters the deck actually wrote."""
    entry = abaqus_index().get(cls.name)
    if entry is None:
        return []
    problems: List[str] = []
    specs = []
    for s in cls.Params._fields:
        ip = entry.params.get(s.abq)
        if ip is None or not ip.default or s.kind == "rest":
            specs.append(s)
            continue
        if len(ip.defaults) > 1:
            DEFAULT_SKIPS.append((cls.name, s.abq, "default depends on the usage: "
                                  + ", ".join(d or "(none documented)" for d in ip.defaults)))
            specs.append(s)
            continue
        try:
            want = _index_default(s, ip.default)
        except ValueError as e:
            if s.abaqus_default is None:
                problems.append(f"*{cls.name}, {s.abq}: Abaqus default {ip.default!r} is not a value of the "
                                f"field ({e}); Param(abaqus_default=reason) to accept")
            specs.append(s)
            continue
        have = s.default
        if have is _MISSING or have is None:
            if s.optional and s.abaqus_default is None and not (s.unsupported or s.control):
                s = dataclasses.replace(s, default=want)
                DEFAULT_FILLS.append((cls.name, s.abq, want))
            elif s.abaqus_default is None and not s.optional and have is _MISSING:
                pass                                    # required: the deck must give it
            elif s.abaqus_default is None and not (s.unsupported or s.control):
                problems.append(f"*{cls.name}, {s.abq}: default None, Abaqus documents {want!r}")
        elif have != want and s.abaqus_default is None and not (s.unsupported or s.control):
            problems.append(f"*{cls.name}, {s.abq}: default {have!r}, Abaqus documents {want!r}; "
                            f"Param(abaqus_default=reason) if that is on purpose")
        specs.append(s)
    cls.Params._fields = tuple(specs)
    return problems


def _param_specs(schema: Type[Schema]) -> list:
    """Registry ParamSpecs mirroring the schema, for introspection only: the
    reader validates a schema keyword with the schema itself."""
    from dispsolver.io.keyword_registry import ParamSpec
    types = {"str": str, "int": int, "float": float, "bool": bool, "choice": str, "rest": tuple}
    return [ParamSpec(s.abq, type=types[s.kind], default=None if s.default is _MISSING else s.default,
                      required=s.required, choices=list(s.choices) or None,
                      supported=not s.unsupported, control=s.control)
            for s in schema._fields]


def keyword(name: str, *, support: str = "FULL", contexts=None, aliases=(), description: str = ""):
    """Register a Keyword class with the reader's registry. `contexts` defaults
    to the Abaqus levels in the keyword index."""
    from dispsolver.io.keyword_index import contexts_for
    from dispsolver.io.keyword_registry import GLOBAL_KEYWORD_REGISTRY, KeywordContext

    def deco(cls: Type[Keyword]):
        cls.name = norm(name)
        bad = check_against_index(cls)
        if bad:
            raise TypeError(f"{cls.__name__}: not in the Abaqus keyword index: {bad} "
                            f"(Param(undocumented=reason) to accept)")
        bad = apply_abaqus_defaults(cls)
        if bad:
            raise TypeError(f"{cls.__name__}: parameter default differs from the Abaqus Keywords Guide: {bad}")
        if contexts is None:
            ctxs = {KeywordContext[c] for c in contexts_for(abaqus_index().get(cls.name))}
        else:
            ctxs = set(contexts)

        def handler(reader, block):
            ctx = KeywordCtx(cls.name, reader.context.name,
                             reader.current_step.name if reader.current_step is not None else None,
                             reader.dim, Loc.of(block))
            def report(level, code, msg, param=None):
                reader.report(level, code, msg, block)
                if param is not None:             # legacy view: reader.unsupported_params
                    kind = {"unknown-param": "unknown", "ambiguous-param": "unknown",
                            "unsupported-param": "unsupported", "control-param-ignored": "control"}[code]
                    if (cls.name, param, kind) not in reader.unsupported_params:
                        reader.unsupported_params.append((cls.name, param, kind))

            try:
                records = run_keyword(cls, block, ctx, report)
            except Exception:
                if cls.closes_with:
                    reader.skip_scope(cls.name, cls.closes_with, block)
                raise
            if cls.closes_with and not any(not isinstance(r, Note) for r in records):
                reader.skip_scope(cls.name, cls.closes_with, block)
            for rec in records:
                apply_record(reader, rec)

        GLOBAL_KEYWORD_REGISTRY.register(cls.name, valid_contexts=ctxs, aliases=list(aliases),
                                         description=description or (cls.__doc__ or ""),
                                         support=support, schema=cls,
                                         params=_param_specs(cls.Params))(handler)
        SCHEMA_KEYWORDS[cls.name] = cls
        return cls
    return deco


def parse_text(text: str, context: str = "STEP", step: Optional[str] = "Step-1", dim: int = 3):
    """Test helper: PARSE every block of `text` with its schema keyword, no
    reader. Returns (records, [(level, code, message)]); Note records are
    returned as diagnostics, as the reader would report them."""
    from dispsolver.io.abaqus_lexer import tokenize_string
    diags: List[Tuple[str, str, str]] = []
    recs: List[Record] = []
    for b in tokenize_string(text):
        cls = SCHEMA_KEYWORDS.get(abaqus_index().resolve_keyword(b.keyword) or b.keyword)
        if cls is None:
            raise KeyError(f"*{b.keyword} has no schema")
        ctx = KeywordCtx(cls.name, context, step, dim, Loc.of(b))
        for rec in run_keyword(cls, b, ctx, lambda lv, c, m, param=None: diags.append((lv, c, m))):
            if isinstance(rec, Note):           # what the Note applier would report
                diags.append((rec.level, rec.code, rec.message))
            else:
                recs.append(rec)
    return recs, diags
