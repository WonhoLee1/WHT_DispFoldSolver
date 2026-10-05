"""BIND pass: resolve every name a parsed Model refers to, in one place.

`SymbolTable` holds case-insensitive tables built from a parsed `Model`;
`bind_model` walks every reference-carrying field and reports a reference that
does not resolve as an ``unresolved-ref`` / ``ambiguous-ref`` error instead of
letting a later consumer skip it or fall back to a default (the "never build a
different model silently" rule). Validation only: the model is not mutated and
no consumer is changed (plan_a3_bind_20261005.md, phase P1).

Name rules mirror `model/regions.py`, which resolves the same names against the
flattened solver system:
  * labels are case-insensitive;
  * an assembly-level set/surface is found by its own name; a part-level one by
    ``INST.NAME`` (``ASM.INST.NAME`` is accepted, the assembly prefix is dropped);
  * a bare part-level name resolves only if exactly one instance's part defines
    it, otherwise `AmbiguousRef`;
  * ``ALL``, a bare node/element number (single-instance models only), or
    ``INST.<n>``; a step-level ``*NSET`` overlay for node regions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from dispsolver.io.diagnostics import Diagnostics, Loc
from dispsolver.model.names import CaseInsensitiveDict
from dispsolver.model.regions import ambiguity_text, strip_assembly_prefix

KINDS = ("step", "material", "section", "orientation", "amplitude", "contact_property", "part",
         "instance", "nset", "elset", "surface", "node_region", "elem_region")


class BindError(ValueError):
    """Base of the two reference failures."""

    def __init__(self, kind: str, name: Any, message: str):
        super().__init__(message)
        self.kind = kind
        self.name = name


class UnresolvedRef(BindError):
    """The name matches nothing."""


class AmbiguousRef(BindError):
    """A bare part-level name is defined by several instances."""

    def __init__(self, kind: str, name: Any, candidates: List[str]):
        super().__init__(kind, name, ambiguity_text(kind, name, candidates))
        self.candidates = sorted(candidates)


_SET_ATTRS = {"nset": ("node_sets", "sets"), "elset": ("element_sets", "sets"), "surface": ("surfaces",)}


class SymbolTable:
    def __init__(self, model: Any):
        self.model = model
        asm = getattr(model, "root_assembly", None)
        self.assembly = asm
        self.assembly_name = str(getattr(asm, "name", "") or "")
        self.simple: Dict[str, Any] = {
            "step": getattr(model, "steps", {}),
            "material": getattr(model, "materials", {}),
            "section": getattr(model, "sections", {}),
            "orientation": getattr(model, "orientations", {}),
            "amplitude": getattr(model, "amplitudes", {}),
            "contact_property": getattr(model, "contact_properties", {}),
            "part": getattr(model, "parts", {}),
            "instance": getattr(asm, "instances", {}) if asm is not None else {},
        }
        self.instances: Dict[str, Any] = CaseInsensitiveDict(self.simple["instance"])
        self.analytical_surfaces = getattr(model, "analytical_rigid_surfaces", {}) or {}

    # ── lookup helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _find(table: Any, name: str) -> Any:
        """Case-insensitive lookup in any mapping; None when absent."""
        if table is None:
            return None
        if name in table:
            return table[name]
        up = name.upper()
        for k in table:
            if isinstance(k, str) and k.upper() == up:
                return table[k]
        return None

    def _strip_assembly(self, key: str) -> str:
        return strip_assembly_prefix(self.assembly_name, key)

    def _part_find(self, part: Any, kind: str, name: str) -> Any:
        for attr in _SET_ATTRS[kind]:
            obj = self._find(getattr(part, attr, None), name)
            if obj is not None:
                return obj
        return None

    def _asm_find(self, kind: str, name: str) -> Any:
        attr = {"nset": "node_sets", "elset": "element_sets", "surface": "surfaces"}[kind]
        obj = self._find(getattr(self.assembly, attr, None), name) if self.assembly is not None else None
        if obj is None and kind == "surface":
            obj = self._find(self.analytical_surfaces, name)
        return obj

    def _scoped(self, kind: str, name: str) -> Any:
        """nset / elset / surface by assembly name, INST.NAME, or unique bare part-level name."""
        key = self._strip_assembly(str(name).strip())
        obj = self._asm_find(kind, key)
        if obj is not None:
            return obj
        for iname, inst in self.instances.items():                 # INST.NAME
            if key.upper().startswith(iname.upper() + "."):
                obj = self._part_find(inst.part, kind, key[len(iname) + 1:])
                if obj is not None:
                    return obj
        hits = [(iname, self._part_find(inst.part, kind, key)) for iname, inst in self.instances.items()]
        hits = [(i, o) for i, o in hits if o is not None]
        if len(hits) == 1:
            return hits[0][1]
        if len(hits) > 1:
            raise AmbiguousRef(kind, name, [f"{i}.{key}" for i, _ in hits])
        raise UnresolvedRef(kind, name, f"{kind} {name!r} is not defined in this model")

    def _number(self, kind: str, key: str, what: str) -> Any:
        """A node / element number: bare (one instance) or INST.<n>."""
        attr = "nodes" if kind == "node" else "elements"
        if key.isdigit():
            if len(self.instances) != 1:
                raise UnresolvedRef(kind + "_region", key,
                                    f"bare {kind} number {key} in a model with instances "
                                    f"{sorted(self.instances)}; write it as INSTANCE.{key}")
            iname, inst = next(iter(self.instances.items()))
            if int(key) not in getattr(inst.part, attr, {}):
                raise UnresolvedRef(kind + "_region", key, f"{kind} {key} does not exist")
            return (kind, iname, int(key))
        if "." in key:
            inst_name, tail = key.rsplit(".", 1)
            if tail.isdigit():
                inst = self._find(self.instances, inst_name)
                if inst is None or int(tail) not in getattr(inst.part, attr, {}):
                    raise UnresolvedRef(kind + "_region", key, f"{kind} {key!r} does not exist")
                return (kind, inst_name, int(tail))
        return None

    # ── public API ─────────────────────────────────────────────────────────
    def resolve(self, kind: str, name: Any, step: Any = None) -> Any:
        """Resolve `name` of `kind`. Raises UnresolvedRef / AmbiguousRef."""
        if kind not in KINDS:
            raise ValueError(f"unknown reference kind {kind!r}")
        if name is None or not str(name).strip():
            raise UnresolvedRef(kind, name, f"empty {kind} reference")
        sname = str(name).strip()
        if kind in self.simple:
            obj = self._find(self.simple[kind], sname)
            if obj is None:
                raise UnresolvedRef(kind, name, f"{kind} {name!r} is not defined in this model")
            return obj
        if kind in _SET_ATTRS:
            return self._scoped(kind, sname)
        if kind == "node_region":
            return self._region("nset", "node", sname, step)
        return self._region("elset", "element", sname, step)

    def _region(self, set_kind: str, num_kind: str, sname: str, step: Any) -> Any:
        key = self._strip_assembly(sname)
        try:
            return self._scoped(set_kind, key)
        except UnresolvedRef:
            pass
        if key.upper() == "ALL":
            return "ALL"
        got = self._number(num_kind, key, sname)
        if got is not None:
            return got
        if step is not None:
            overlay = getattr(step, "step_nsets" if set_kind == "nset" else "step_elsets", {}) or {}
            for oname, ids in overlay.items():
                if oname.upper() == key.upper():
                    return ids
        raise UnresolvedRef(num_kind + "_region", sname,
                            f"{num_kind} region {sname!r} is not a {set_kind}"
                            f"{', surface' if num_kind == 'node' else ''} or {num_kind} number of this model")

    def resolve_in_part(self, part: Any, kind: str, name: str) -> Any:
        """Part-local lookup (section assignments refer to the part's own sets)."""
        obj = self._part_find(part, kind, str(name).strip())
        if obj is None:
            raise UnresolvedRef(kind, name, f"{kind} {name!r} is not defined in part {getattr(part, 'name', '?')!r}")
        return obj


# ── the walk ───────────────────────────────────────────────────────────────

def _loc(obj: Any) -> Optional[Loc]:
    s = getattr(obj, "source_location", None)
    if not s or not isinstance(s, str):
        return None
    f, sep, ln = s.rpartition(":")
    if sep and ln.isdigit():
        return Loc(f, int(ln))
    return None


class _Binder:
    def __init__(self, model: Any, diagnostics: Diagnostics):
        self.model = model
        self.diag = diagnostics
        self.sym = SymbolTable(model)

    def check(self, what: str, obj: Any, kind: str, name: Any, level: str = "error", step: Any = None,
              part: Any = None) -> None:
        try:
            if kind == "surface_or_nset":
                try:
                    self.sym.resolve("surface", name)
                except UnresolvedRef:
                    self.sym.resolve("node_region", name, step)
            elif kind == "any_region":
                try:
                    self.sym.resolve("node_region", name, step)
                except UnresolvedRef:
                    self.sym.resolve("elem_region", name, step)
            elif kind == "pressure_region":
                try:
                    self.sym.resolve("surface", name)
                except UnresolvedRef:
                    self.sym.resolve("elem_region", name, step)
            elif part is not None:
                self.sym.resolve_in_part(part, kind, name)
            else:
                self.sym.resolve(kind, name, step)
        except BindError as e:
            code = "ambiguous-ref" if isinstance(e, AmbiguousRef) else "unresolved-ref"
            loc = _loc(obj)
            self.diag.add(level, code, f"{what}: {e}" + (f" [{loc}]" if loc else ""), loc)

    def run(self) -> None:
        m = self.model
        for step in m.steps.values():
            self.step_entities(step)
            self.outputs(step)
        for sec in m.sections.values():
            w = f"section {sec.name}"
            if getattr(sec, "material_name", None):
                self.check(w + " material", sec, "material", sec.material_name)
            if getattr(sec, "orientation_name", None):
                self.check(w + " orientation", sec, "orientation", sec.orientation_name)
        for part in m.parts.values():
            for sa in getattr(part, "section_assignments", []):
                w = f"part {part.name} section assignment"
                sec = self.sym._find(m.sections, sa.section_name)
                self.check(w + " section", sec if sec is not None else part, "section", sa.section_name)
                self.check(w + " region", sec if sec is not None else part, "elset", sa.region, part=part)
        for ia in m.interactions.values():
            self.interaction(ia)
        for c in m.constraints.values():
            self.constraint(c)
        for t in m.transforms.values():
            if t.get("nset"):
                self.check(f"*TRANSFORM {t.get('name', '')} nset", t, "node_region", t["nset"])

    def step_entities(self, step: Any) -> None:
        for entry in step.boundary_conditions.values():
            if getattr(entry, "status", None) is not None and entry.status.name != "CREATED":
                continue                      # propagated copy: checked where it was created
            bc = entry.entity
            w = f"boundary condition {bc.name}"
            self.check(w + " region", bc, "node_region", bc.region, step=step)
            if isinstance(getattr(bc, "amplitude", None), str):
                self.check(w + " amplitude", bc, "amplitude", bc.amplitude)
        for entry in step.loads.values():
            if getattr(entry, "status", None) is not None and entry.status.name != "CREATED":
                continue
            ld = entry.entity
            w = f"load {ld.name}"
            self.check(w + " step", ld, "step", ld.create_step_name)
            region = getattr(ld, "region", None)
            cls = type(ld).__name__
            if cls == "ConcentratedForce":
                self.check(w + " region", ld, "node_region", region, step=step)
            elif cls == "Pressure":
                self.check(w + " region", ld, "pressure_region", region, step=step)
            elif region not in (None, ""):                # Gravity / BodyForce: blank = whole model
                self.check(w + " region", ld, "elem_region", region, step=step)
            if isinstance(getattr(ld, "amplitude", None), str):
                self.check(w + " amplitude", ld, "amplitude", ld.amplitude)
        for pf in getattr(step, "predefined_fields", {}).values():
            self.check(f"predefined field {pf.name} region", pf, "any_region", pf.region, step=step)

    def outputs(self, step: Any) -> None:
        for reqs, label in ((step.field_output_requests, "field"), (step.history_output_requests, "history")):
            for rq in reqs.values():
                reg = getattr(rq, "region", None)
                if isinstance(reg, str) and reg.strip() and reg.strip().upper() != "WHOLEMODEL":
                    self.check(f"{label} output request {rq.name} region", rq, "any_region", reg,
                               level="warning", step=step)

    def surface_pair(self, w: str, obj: Any, src: Any) -> None:
        """main/secondary and their legacy master/slave aliases hold the same names: check each once."""
        seen = set()
        for attr in ("main", "secondary", "master", "slave"):
            v = getattr(src, attr, None)
            if isinstance(v, str) and v.strip() and v.strip().upper() not in seen:
                seen.add(v.strip().upper())
                self.check(f"{w} {attr} surface", obj, "surface_or_nset", v)

    def interaction(self, ia: Any) -> None:
        w = f"interaction {ia.name}"
        ref = getattr(ia, "interaction_property_ref", None)
        if ref:
            self.check(w + " interaction property", ia, "contact_property", ref)
        self.surface_pair(w, ia, ia)

    def constraint(self, c: Any) -> None:
        w = f"constraint {c.name}"
        cls = type(c).__name__
        if cls == "Tie":
            self.surface_pair(w, c, c)
        elif cls in ("Coupling", "KinematicCoupling", "DistributingCoupling"):
            rp = getattr(c, "ref_point", None)
            if isinstance(rp, (str, int)) and not isinstance(rp, bool) and str(rp).strip():
                self.check(w + " reference node", c, "node_region", str(rp))
            if isinstance(c.surface, str) and c.surface.strip():
                self.check(w + " surface", c, "surface_or_nset", c.surface)
        elif cls == "RigidBody":
            rp = getattr(c, "ref_point", None)
            if isinstance(rp, (str, int)) and not isinstance(rp, bool) and str(rp).strip():
                self.check(w + " reference node", c, "node_region", str(rp))
            for attr in ("tie_nodes", "pin_nodes"):
                v = getattr(c, attr, None)
                if isinstance(v, str) and v.strip():
                    self.check(f"{w} {attr}", c, "node_region", v)
            v = getattr(c, "body_elements", None)
            if isinstance(v, str) and v.strip():
                self.check(w + " elset", c, "elem_region", v)


def bind_model(model: Any, diagnostics: Diagnostics) -> SymbolTable:
    """Report every unresolved / ambiguous reference of `model` into `diagnostics`."""
    b = _Binder(model, diagnostics)
    b.run()
    return b.sym
