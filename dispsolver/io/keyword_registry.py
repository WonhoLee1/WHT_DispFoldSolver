"""
keyword_registry.py
===================
Declarative Keyword Registry and Context Management for Abaqus Input Reader.
Based on SIMULIA Abaqus Keywords Guide and Model Definition rules.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set, Any
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


@dataclass
class KeywordSpec:
    """Specification of an Abaqus keyword and its parser handler."""
    name: str                                           # Normalized uppercase keyword, e.g. "NODE"
    handler: Callable[[Any, AbaqusKeywordBlock], None]  # function(reader, block)
    valid_contexts: Set[KeywordContext] = field(default_factory=lambda: {KeywordContext.GLOBAL})
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    products: str = ""          # Abaqus products offering this keyword (e.g. "Standard")
    keyword_type: str = ""      # "Model data" / "History data" / "Model or history data"
    level: str = ""             # Abaqus level (e.g. "Part, Assembly, Step")
    support: str = "FULL"       # FULL / PARTIAL / STUB (parse-store, unconnected) / IGNORED
    slug: str = ""              # SIMACAEKEYRefMap simakey-r-*.htm tracker


class KeywordRegistry:
    """Singleton/Central registry for Abaqus keyword handlers."""

    def __init__(self):
        self._handlers: Dict[str, KeywordSpec] = {}

    def register(
        self,
        keyword: str,
        valid_contexts: Optional[Set[KeywordContext]] = None,
        aliases: Optional[List[str]] = None,
        description: str = ""
    ):
        """Decorator to register a keyword handler function."""
        if valid_contexts is None:
            valid_contexts = {KeywordContext.GLOBAL}
        if aliases is None:
            aliases = []

        norm_kw = keyword.upper().strip()

        def decorator(fn: Callable[[Any, AbaqusKeywordBlock], None]):
            spec = KeywordSpec(
                name=norm_kw,
                handler=fn,
                valid_contexts=valid_contexts,
                aliases=[a.upper().strip() for a in aliases],
                description=description or fn.__doc__ or ""
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


# Global default registry instance
GLOBAL_KEYWORD_REGISTRY = KeywordRegistry()
