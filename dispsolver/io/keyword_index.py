"""Abaqus keyword index: what Abaqus defines, independent of what we implement.

`abaqus_keyword_index.json` is extracted from the Abaqus Keywords Guide by
`tools/abaqus_keyword_index/build_index.py` (structure only: keyword names,
data type, levels, per-usage parameter names / values / defaults, data-line
layouts). The reader consults it to tell three cases apart that a
hand-written registry cannot:

    * a keyword / parameter Abaqus defines and we implement     -> handled
    * one Abaqus defines and we do not implement                -> gap
    * one Abaqus does not define (typo, wrong deck dialect)     -> error

Abbreviations are legal Abaqus input: "Keywords, parameters, and, in most
cases, parameter values need not be spelled out completely, but there must be
enough characters given to distinguish them from other keywords, parameters,
and parameter values" (Abaqus Analysis User's Guide, Input syntax rules).
`resolve_keyword` / `resolve_param` / `resolve_value` implement that rule:
an exact match wins, otherwise a unique prefix; an ambiguous prefix is an
error, never a guess.

Requiredness in the index is per usage (e.g. *BOUNDARY's STEP is required
only for submodel boundary conditions), so the index is NOT used to demand
parameters; the keyword's own schema does that.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

_INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "abaqus_keyword_index.json")


def norm(name: str) -> str:
    """Canonical spelling of a keyword or parameter name: upper case, single
    spaces; '_' and '-' read as a space (decks use all three)."""
    return " ".join(str(name).replace("_", " ").replace("-", " ").split()).upper()


class AmbiguousName(ValueError):
    """A prefix that matches more than one name."""

    def __init__(self, given: str, candidates: Iterable[str]):
        self.given = given
        self.candidates = sorted(candidates)
        super().__init__(f"{given!r} is ambiguous: {', '.join(self.candidates)}")


def resolve_name(given: str, names: Iterable[str]) -> Optional[str]:
    """Exact match, else the unique name `given` is a prefix of, else None.
    Raises AmbiguousName for a prefix of several names."""
    g = norm(given)
    pool = list(names)
    if g in pool:
        return g
    # compare with blanks removed too: Abaqus ignores blanks on keyword lines
    gc = g.replace(" ", "")
    hits = [n for n in pool if n.startswith(g) or n.replace(" ", "").startswith(gc)]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        return None
    raise AmbiguousName(given, hits)


@dataclass(frozen=True)
class IndexParam:
    name: str
    values: Tuple[str, ...] = ()          # documented keyword values (empty: free value)
    default: Optional[str] = None
    flag: bool = False                    # stands alone, no value


@dataclass(frozen=True)
class IndexKeyword:
    name: str
    slug: str
    data_type: str                        # model / history / both
    levels: Tuple[str, ...]               # part, part instance, assembly, model, step
    assembly_supported: bool
    params: Dict[str, IndexParam]         # union over all usages

    def resolve_param(self, given: str) -> Optional[str]:
        return resolve_name(given, self.params)


class KeywordIndex:
    def __init__(self, doc: dict):
        self._kw: Dict[str, IndexKeyword] = {}
        for name, e in doc["keywords"].items():
            params: Dict[str, IndexParam] = {}
            for u in e.get("usages", []):
                for p in u.get("required", []) + u.get("optional", []):
                    pn = norm(p["name"])
                    old = params.get(pn)
                    vals = tuple(dict.fromkeys((old.values if old else ()) + tuple(p.get("values", ()))))
                    params[pn] = IndexParam(pn, vals,
                                            (old.default if old and old.default else p.get("default")),
                                            bool(p.get("flag")) or bool(old and old.flag))
            self._kw[norm(name)] = IndexKeyword(norm(name), e.get("slug", ""), e.get("data_type", ""),
                                                tuple(e.get("levels", ())),
                                                bool(e.get("assembly_supported", True)), params)
        self.names: FrozenSet[str] = frozenset(self._kw)

    def __contains__(self, name: str) -> bool:
        return norm(name) in self._kw

    def get(self, name: str) -> Optional[IndexKeyword]:
        return self._kw.get(norm(name))

    def resolve_keyword(self, given: str) -> Optional[str]:
        return resolve_name(given, self.names)

    def __len__(self) -> int:
        return len(self._kw)


@lru_cache(maxsize=1)
def abaqus_index() -> KeywordIndex:
    with open(_INDEX_PATH, encoding="utf-8") as f:
        return KeywordIndex(json.load(f))


# Abaqus levels -> reader contexts. A part-level keyword also appears at the
# top level of a deck without *PART (the whole model is one part) and inside
# an instance that carries its own mesh.
LEVEL_CONTEXTS = {
    "part": ("PART", "GLOBAL", "INSTANCE"),
    "part instance": ("INSTANCE",),
    "assembly": ("ASSEMBLY",),
    "model": ("GLOBAL",),
    "step": ("STEP",),
}


def contexts_for(entry: IndexKeyword) -> List[str]:
    """KeywordContext names where Abaqus allows this keyword."""
    out: List[str] = []
    for lv in entry.levels:
        for c in LEVEL_CONTEXTS.get(lv, ()):
            if c not in out:
                out.append(c)
    return out
