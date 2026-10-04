"""
keyword_registry.py
===================
Declarative Keyword Registry and Context Management for Abaqus Input Reader.
Based on SIMULIA Abaqus Keywords Guide and Model Definition rules.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set, Any, Union
from dataclasses import dataclass, field

from dispsolver.io.abaqus_lexer import AbaqusKeywordBlock


class KeywordContext(Enum):
    """Execution context/scope where an Abaqus keyword is valid."""
    GLOBAL = auto()      # Model-level data (top-level outside Part/Assembly/Step)
    PART = auto()        # Inside *PART ... *END PART
    ASSEMBLY = auto()    # Inside *ASSEMBLY ... *END ASSEMBLY (outside Instance)
    INSTANCE = auto()    # Inside *INSTANCE ... *END INSTANCE
    STEP = auto()        # Inside *STEP ... *END STEP (history data)
    MATERIAL = auto()    # Material sub-property block inside *MATERIAL


def normalize_param_name(name: str) -> str:
    """Normalize keyword parameter name by stripping, uppercasing, and standardizing separators."""
    return name.upper().strip().replace("-", " ").replace("_", " ")


@dataclass
class ParamSpec:
    """Specification for a parameter on an Abaqus keyword."""
    name: str                                           # Normalized uppercase name, e.g. "NAME", "ELSET"
    type: type = str                                    # Expected type: str, int, float, bool
    default: Any = None                                 # Default value if omitted
    required: bool = False                              # Mandatory parameter?
    choices: Optional[List[Any]] = None                 # Valid values (case-insensitive strings or values)
    supported: bool = True                              # Whether this solver implements this parameter
    aliases: List[str] = field(default_factory=list)    # Alternative spellings or abbreviations (e.g. AMP for AMPLITUDE)
    description: str = ""

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ParamSpec):
            return self.name == other.name
        return False


@dataclass
class RowSpec:
    """Specification for data lines following an Abaqus keyword."""
    min_columns: int = 0
    max_columns: Optional[int] = None
    column_types: List[type] = field(default_factory=list)
    repeating: bool = True
    description: str = ""


@dataclass
class KeywordSpec:
    """Specification of an Abaqus keyword and its parser handler."""
    name: str                                           # Normalized uppercase keyword, e.g. "NODE"
    handler: Callable[[Any, AbaqusKeywordBlock], None]  # function(reader, block)
    valid_contexts: Set[KeywordContext] = field(default_factory=lambda: {KeywordContext.GLOBAL})
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    products: str = "Standard"                          # Abaqus products offering this keyword (e.g. "Standard")
    keyword_type: str = ""                              # "Model data" / "History data" / "Model or history data"
    level: str = ""                                     # Abaqus level (e.g. "Part, Assembly, Step")
    support: str = "FULL"                               # FULL / PARTIAL / STUB (parse-store, unconnected) / IGNORED
    slug: str = ""                                      # SIMACAEKEYRefMap simakey-r-*.htm tracker
    params: Dict[str, ParamSpec] = field(default_factory=dict)
    rows: Optional[RowSpec] = None


class KeywordRegistry:
    """Singleton/Central registry for Abaqus keyword handlers."""

    def __init__(self):
        self._handlers: Dict[str, KeywordSpec] = {}

    def register(
        self,
        keyword: str,
        valid_contexts: Optional[Set[KeywordContext]] = None,
        aliases: Optional[List[str]] = None,
        description: str = "",
        support: str = "FULL",
        products: str = "Standard",
        keyword_type: str = "",
        level: str = "",
        params: Optional[Union[Dict[str, Union[ParamSpec, dict, type]], List[Union[ParamSpec, str]]]] = None,
        rows: Optional[Union[RowSpec, type, List[type]]] = None,
        slug: str = "",
    ):
        """Decorator to register a keyword handler function with declarative schemas."""
        if valid_contexts is None:
            valid_contexts = {KeywordContext.GLOBAL}
        if aliases is None:
            aliases = []

        norm_kw = keyword.upper().strip()

        norm_params: Dict[str, ParamSpec] = {}
        if params is not None:
            if isinstance(params, dict):
                for p_name, p_val in params.items():
                    k_norm = normalize_param_name(p_name)
                    if isinstance(p_val, ParamSpec):
                        ps = p_val
                    elif isinstance(p_val, type):
                        ps = ParamSpec(name=k_norm, type=p_val)
                    elif isinstance(p_val, dict):
                        ps = ParamSpec(name=k_norm, **p_val)
                    elif p_val is None:
                        ps = ParamSpec(name=k_norm)
                    else:
                        raise TypeError(f"Invalid param spec for {k_norm}: {p_val}")
                    norm_params[k_norm] = ps
                    for alias in ps.aliases:
                        norm_params[normalize_param_name(alias)] = ps
            elif isinstance(params, (list, tuple, set)):
                for item in params:
                    if isinstance(item, ParamSpec):
                        k_norm = normalize_param_name(item.name)
                        norm_params[k_norm] = item
                        for alias in item.aliases:
                            norm_params[normalize_param_name(alias)] = item
                    elif isinstance(item, str):
                        k_norm = normalize_param_name(item)
                        norm_params[k_norm] = ParamSpec(name=k_norm)
                    else:
                        raise TypeError(f"Invalid param item: {item}")

        norm_rows: Optional[RowSpec] = None
        if rows is not None:
            if isinstance(rows, RowSpec):
                norm_rows = rows
            elif isinstance(rows, type):
                norm_rows = RowSpec(column_types=[rows], repeating=True)
            elif isinstance(rows, (list, tuple)):
                norm_rows = RowSpec(column_types=list(rows), repeating=True)

        def decorator(fn: Callable[[Any, AbaqusKeywordBlock], None]):
            spec = KeywordSpec(
                name=norm_kw,
                handler=fn,
                valid_contexts=valid_contexts,
                aliases=[a.upper().strip() for a in aliases],
                description=description or fn.__doc__ or "",
                products=products,
                keyword_type=keyword_type,
                level=level,
                support=support.upper().strip(),
                slug=slug,
                params=norm_params,
                rows=norm_rows,
            )
            self._handlers[norm_kw] = spec
            for alias in spec.aliases:
                self._handlers[alias] = spec
            return fn

        return decorator

    def get_handler(self, keyword: str) -> Optional[KeywordSpec]:
        """Look up a keyword specification by name."""
        norm_kw = keyword.upper().strip()
        return self._handlers.get(norm_kw)

    def is_registered(self, keyword: str) -> bool:
        return keyword.upper().strip() in self._handlers

    @property
    def registered_keywords(self) -> List[str]:
        return sorted(list(self._handlers.keys()))

    def all_specs(self) -> Dict[str, KeywordSpec]:
        """Return a mapping of canonical keyword names to their KeywordSpec."""
        res: Dict[str, KeywordSpec] = {}
        for kw, spec in self._handlers.items():
            if spec.name not in res:
                res[spec.name] = spec
        return res


# Global default registry instance
GLOBAL_KEYWORD_REGISTRY = KeywordRegistry()

