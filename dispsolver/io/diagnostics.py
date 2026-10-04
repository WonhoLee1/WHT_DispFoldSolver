"""One channel for every problem found in an input deck.

Every stage of the .inp pipeline (lexer, keyword parsing, binding, lowering)
reports through a `Diagnostics` collector instead of printing, warning or
raising on its own. The policy -- refuse or run -- is applied once, at the
end (`raise_if_blocking`). See dev_log/design_inp_pipeline_20261005.md §4.

Levels
  error    the deck is wrong or cannot be read            -> always blocks a run
  gap      valid Abaqus input this solver does not represent -> blocks a strict run
  warning  represented, with a documented substitution or caveat
  info     noteworthy, no effect on the model
"""

from __future__ import annotations

import os
import warnings as _warnings
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Literal, Optional, Union

Level = Literal["error", "gap", "warning", "info"]
_LEVELS = ("error", "gap", "warning", "info")


@dataclass(frozen=True)
class Loc:
    """A deck location: file (as given) and 1-based physical line."""
    file: str
    line: int

    def __str__(self) -> str:
        return f"{os.path.basename(self.file) if self.file else '<input>'}:{self.line}"

    @classmethod
    def of(cls, block) -> Optional["Loc"]:
        """Location of a lexer block (or None)."""
        if block is None:
            return None
        return cls(getattr(block, "source_file", "") or "", int(getattr(block, "line_number", 0) or 0))


@dataclass(frozen=True)
class Diagnostic:
    level: Level
    code: str           # stable kebab-case identifier, e.g. "unresolved-ref"
    message: str        # complete human-readable text (includes the location where useful)
    loc: Optional[Loc] = None

    def __str__(self) -> str:
        return self.message


class BlockingDiagnostics(ValueError):
    """Raised by Diagnostics.raise_if_blocking; carries the blocking entries."""

    def __init__(self, message: str, diagnostics: List[Diagnostic]):
        super().__init__(message)
        self.diagnostics = diagnostics


class Diagnostics:
    """Ordered collector. Entries are de-duplicated by (level, code, message)."""

    def __init__(self, echo: bool = True):
        self._items: List[Diagnostic] = []
        self._seen: set = set()
        self.echo = echo        # mirror non-info entries to Python warnings (legacy behaviour)

    def add(self, level: Level, code: str, message: str, loc: Optional[Loc] = None) -> Diagnostic:
        if level not in _LEVELS:
            raise ValueError(f"unknown diagnostic level {level!r}")
        d = Diagnostic(level, code, message, loc)
        key = (level, code, message)
        if key not in self._seen:
            self._seen.add(key)
            self._items.append(d)
            if self.echo and level != "info":
                _warnings.warn(message, stacklevel=3)
        return d

    def error(self, code: str, message: str, loc: Optional[Loc] = None) -> Diagnostic:
        return self.add("error", code, message, loc)

    def gap(self, code: str, message: str, loc: Optional[Loc] = None) -> Diagnostic:
        return self.add("gap", code, message, loc)

    def warning(self, code: str, message: str, loc: Optional[Loc] = None) -> Diagnostic:
        return self.add("warning", code, message, loc)

    def info(self, code: str, message: str, loc: Optional[Loc] = None) -> Diagnostic:
        return self.add("info", code, message, loc)

    def of(self, *levels: Level) -> List[Diagnostic]:
        return [d for d in self._items if d.level in levels]

    def blocking(self, strict: bool) -> List[Diagnostic]:
        return self.of("error", "gap") if strict else self.of("error")

    def raise_if_blocking(self, strict: bool, exc_type: type = BlockingDiagnostics, header: str = "") -> None:
        bad = self.blocking(strict)
        if not bad:
            return
        lines = [str(d) for d in bad[:30]]
        if len(bad) > 30:
            lines.append(f"... and {len(bad) - 30} more")
        text = (header or "the deck has problems that block the run:") + "\n  " + "\n  ".join(lines)
        if issubclass(exc_type, BlockingDiagnostics):
            raise exc_type(text, bad)
        raise exc_type(text)

    def notice_sink(self, level: Level = "warning", code: str = "solver-notice") -> "_NoticeSink":
        """A list-like object for legacy code that reports by `notices.append(msg)`."""
        return _NoticeSink(self, level, code)

    def __iter__(self) -> Iterator[Diagnostic]:
        return iter(list(self._items))

    def __len__(self) -> int:
        return len(self._items)

    def table(self) -> str:
        """Human-readable table, one line per entry, in input order."""
        return "\n".join(f"{d.level.upper():7s} {d.code:28s} {d.message}" for d in self._items)


class _NoticeSink(list):
    def __init__(self, diags: Diagnostics, level: Level, code: str):
        super().__init__()
        self._d, self._level, self._code = diags, level, code

    def append(self, msg) -> None:  # type: ignore[override]
        super().append(msg)
        self._d.add(self._level, self._code, str(msg))
