"""Abaqus .inp keyword tokenizer — Lexer stage of the parser pipeline.

Transforms .inp text into a list of AbaqusKeywordBlock tokens, each carrying
where it came from (source file and physical line, through *INCLUDE).

Abaqus input syntax handled here (Abaqus Analysis User's Guide, "Input
syntax rules"):
  * a keyword line starts with "*" (leading blanks allowed -- Abaqus' own
    verification decks indent keywords); "**" starts a comment line;
  * a keyword line ending in a comma continues on the next line;
  * the keyword name is everything before the first comma (any number of
    words, case-insensitive, blanks normalised);
  * parameters are "NAME=value" (names may contain blanks and hyphens, e.g.
    PRESSURE-OVERCLOSURE) or bare flags;
  * *INCLUDE, INPUT=file inserts another file at that point.

History: the previous lexer recognised a keyword only in column 1 (85 corpus
decks / 937 indented keyword lines were swallowed as data of the preceding
block), dropped hyphenated parameter names to bare flags (*SURFACE BEHAVIOR,
PRESSURE-OVERCLOSURE=EXPONENTIAL became HARD), had no keyword-line
continuation, recognised at most 3-word keyword names, pasted *INCLUDE text
in place (line numbers after an include were shifted, file names lost) and
left a missing include as an unexplained *INCLUDE block.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


class InpSyntaxError(ValueError):
    """A deck that cannot be tokenized (missing/cyclic *INCLUDE, broken keyword line)."""


@dataclass
class AbaqusKeywordBlock:
    """A single keyword block from an Abaqus .inp file.

    A keyword block begins with a *KEYWORD line (with optional parameters)
    and ends at the next *KEYWORD or end-of-file.
    """
    keyword: str                               # normalized uppercase keyword name
    params: dict = field(default_factory=dict)  # keyword parameters (lowercase keys)
    data_lines: List[str] = field(default_factory=list)  # raw data lines (strings)
    line_number: int = 0                       # physical line of the keyword in source_file
    source_file: str = ""                      # file (or "<string>") the keyword line is in
    data_line_numbers: List[int] = field(default_factory=list)  # physical line of each data line

    @property
    def data_text(self) -> str:
        """All data lines joined as a single string for re-parsing."""
        return "\n".join(self.data_lines)

    @property
    def location(self) -> str:
        """'file:line' of the keyword line (file shortened to its base name)."""
        name = os.path.basename(self.source_file) if self.source_file else "<input>"
        return f"{name}:{self.line_number}"

    def data_location(self, i: int) -> str:
        """'file:line' of data line i."""
        name = os.path.basename(self.source_file) if self.source_file else "<input>"
        if 0 <= i < len(self.data_line_numbers):
            return f"{name}:{self.data_line_numbers[i]}"
        return self.location


# Blocks whose data rows are free lists/pairs: a trailing comma continues
# the logical line. Other blocks (e.g. ELEMENT records) are left unmerged.
_CONTINUATION_KEYWORDS = frozenset({"NSET", "ELSET", "AMPLITUDE"})

_MAX_INCLUDE_DEPTH = 16


def _parse_keyword_params(param_text: str) -> dict:
    """Parse a keyword parameter string into a dict with lower-case keys.

    "NAME=A1, TIME=STEP TIME" -> {'name': 'A1', 'time': 'STEP TIME'};
    "PRESSURE-OVERCLOSURE=EXPONENTIAL" -> {'pressure-overclosure': 'EXPONENTIAL'};
    a segment without "=" is a bare flag: "SMALL SLIDING" -> {'small sliding': 'yes'}.
    """
    params = {}
    if not param_text or not param_text.strip():
        return params
    for part in _split_param_segments(param_text):
        part = part.strip()
        if not part:
            continue
        eq = _first_unquoted(part, "=")
        if eq > 0:
            key = " ".join(part[:eq].split()).lower()
            # Quotes are kept: data-line references to a quoted name keep
            # theirs too, and the two must compare equal.
            params[key] = part[eq + 1:].strip()
        else:
            params[" ".join(part.split()).lower()] = "yes"
    return params


def _first_unquoted(text: str, ch: str) -> int:
    quote = ""
    for i, c in enumerate(text):
        if quote:
            if c == quote:
                quote = ""
        elif c in "\"'":
            quote = c
        elif c == ch:
            return i
    return -1


def _split_param_segments(param_text: str) -> List[str]:
    """Split a parameter string on commas outside quotes."""
    segments = []
    buf: List[str] = []
    quoted = False
    quote_char = ""
    for ch in param_text:
        if quoted:
            buf.append(ch)
            if ch == quote_char:
                quoted = False
        elif ch in ("'", '"'):
            quoted = True
            quote_char = ch
            buf.append(ch)
        elif ch == ",":
            segments.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    segments.append("".join(buf))
    return segments


def _read_file_with_fallback(filepath: str) -> str:
    """Read file with utf-8 first, fallback to cp949 (Korean) then latin-1."""
    for enc in ("utf-8", "cp949"):
        try:
            with open(filepath, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(filepath, "r", encoding="latin-1") as f:
        return f.read()


# (text, source file, physical line number)
_Line = Tuple[str, str, int]


def _is_comment(s: str) -> bool:
    return s.lstrip().startswith("**")


def _is_keyword(s: str) -> bool:
    t = s.lstrip()
    return t.startswith("*") and not t.startswith("**")


def _include_target(keyword_text: str) -> Optional[str]:
    body = keyword_text.lstrip()[1:]
    comma = body.find(",")
    name = " ".join((body if comma < 0 else body[:comma]).split()).upper()
    if name != "INCLUDE":
        return None
    params = _parse_keyword_params(body[comma + 1:] if comma >= 0 else "")
    return params.get("input")


def _physical_lines(text: str, source: str, base_dir: str,
                    stack: Tuple[str, ...]) -> List[_Line]:
    """Lines of `text` with their origin, *INCLUDE expanded recursively.
    Comment lines are dropped here; keyword-line continuation is resolved so
    the *INCLUDE line itself is complete before it is acted on."""
    raw = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: List[_Line] = []
    i = 0
    while i < len(raw):
        line, lineno = raw[i], i + 1
        i += 1
        if _is_comment(line) or not line.strip():
            continue
        if not _is_keyword(line):
            out.append((line, source, lineno))
            continue
        # keyword line, possibly continued by a trailing comma
        kw = line.rstrip()
        while kw.endswith(",") and i < len(raw):
            nxt = raw[i]
            if _is_comment(nxt) or not nxt.strip():
                i += 1
                continue
            if _is_keyword(nxt):
                break   # a trailing comma followed by a new keyword: nothing to continue
            kw = kw + " " + nxt.strip()
            i += 1
        target = _include_target(kw)
        if target is None:
            out.append((kw, source, lineno))
            continue
        where = f"{os.path.basename(source)}:{lineno}"
        full = os.path.normpath(os.path.join(base_dir, target))
        if not os.path.isfile(full):
            raise InpSyntaxError(f"{where}: *INCLUDE, INPUT={target}: file not found ({full})")
        key = os.path.normcase(os.path.abspath(full))
        if key in stack:
            chain = " -> ".join(os.path.basename(s) for s in stack + (key,))
            raise InpSyntaxError(f"{where}: *INCLUDE, INPUT={target}: cyclic include ({chain})")
        if len(stack) >= _MAX_INCLUDE_DEPTH:
            raise InpSyntaxError(f"{where}: *INCLUDE nesting deeper than {_MAX_INCLUDE_DEPTH}")
        out.extend(_physical_lines(_read_file_with_fallback(full), full, os.path.dirname(full),
                                   stack + (key,)))
    return out


def _blocks_from_lines(lines: Sequence[_Line]) -> List[AbaqusKeywordBlock]:
    blocks: List[AbaqusKeywordBlock] = []
    cur: Optional[AbaqusKeywordBlock] = None
    for text, source, lineno in lines:
        if _is_keyword(text):
            body = text.lstrip()[1:]
            comma = body.find(",")
            name = " ".join((body if comma < 0 else body[:comma]).split()).upper()
            if not name:
                raise InpSyntaxError(f"{os.path.basename(source)}:{lineno}: keyword line without a keyword name")
            cur = AbaqusKeywordBlock(keyword=name,
                                     params=_parse_keyword_params(body[comma + 1:] if comma >= 0 else ""),
                                     line_number=lineno, source_file=source)
            blocks.append(cur)
            continue
        if cur is None:
            continue   # data before the first keyword (e.g. heading text without *HEADING)
        cur.data_lines.append(text.strip())
        cur.data_line_numbers.append(lineno)
    for b in blocks:
        _merge_data_continuations(b)
    return blocks


def _merge_data_continuations(block: AbaqusKeywordBlock) -> None:
    """Trailing '-' continuation on any data line, and trailing-comma
    continuation for the list keywords; line numbers follow the first line."""
    lines, nums = [], []
    for s, n in zip(block.data_lines, block.data_line_numbers):
        if s.endswith("-") and len(s) > 1 and not s.startswith("-"):
            s = s[:-1].rstrip(",").rstrip()
        lines.append(s)
        nums.append(n)
    if block.keyword in _CONTINUATION_KEYWORDS:
        merged, mnums, buf, first = [], [], "", 0
        for s, n in zip(lines, nums):
            if not buf:
                first = n
            buf = (buf + " " + s) if buf else s
            if buf.rstrip().endswith(","):
                continue
            merged.append(buf)
            mnums.append(first)
            buf = ""
        if buf:
            merged.append(buf)
            mnums.append(first)
        lines, nums = merged, mnums
    block.data_lines, block.data_line_numbers = lines, nums


def tokenize(filepath: str) -> List[AbaqusKeywordBlock]:
    """Tokenize an Abaqus .inp file into a list of AbaqusKeywordBlock.

    Raises FileNotFoundError if the file does not exist and InpSyntaxError for
    a missing or cyclic *INCLUDE (with the including file:line)."""
    full = os.path.abspath(filepath)
    text = _read_file_with_fallback(full)
    lines = _physical_lines(text, full, os.path.dirname(full), (os.path.normcase(full),))
    return _blocks_from_lines(lines)


def tokenize_string(text: str, source_name: str = "<string>",
                    base_dir: Optional[str] = None) -> List[AbaqusKeywordBlock]:
    """Tokenize .inp text held in memory. *INCLUDE paths are resolved against
    base_dir (default: the current directory)."""
    lines = _physical_lines(text, source_name, base_dir or os.getcwd(), ())
    return _blocks_from_lines(lines)


def _merge_trailing_commas(data_lines: List[str]) -> List[str]:
    """Join logical lines split by trailing commas (kept for callers that
    re-merge raw data lines)."""
    merged: List[str] = []
    buf = ""
    for dl in data_lines:
        if not dl.strip():
            continue
        buf = (buf + " " + dl.strip()) if buf else dl.strip()
        if buf.rstrip().endswith(","):
            continue
        merged.append(buf)
        buf = ""
    if buf:
        merged.append(buf)
    return merged
