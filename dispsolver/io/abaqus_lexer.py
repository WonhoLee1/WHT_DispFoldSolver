"""Abaqus .inp keyword tokenizer — Lexer stage of the parser pipeline.

Transforms raw .inp text into a list of AbaqusKeywordBlock tokens.
Handles: comments, line continuation, free-format data, case-insensitive keywords.
"""

import re
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class AbaqusKeywordBlock:
    """A single keyword block from an Abaqus .inp file.

    A keyword block begins with a *KEYWORD line (with optional parameters)
    and ends at the next *KEYWORD or end-of-file.
    """
    keyword: str                               # normalized uppercase keyword name
    params: dict = field(default_factory=dict)  # keyword parameters (lowercase keys)
    data_lines: List[str] = field(default_factory=list)  # raw data lines (strings)
    line_number: int = 0                       # starting line number in source

    @property
    def data_text(self) -> str:
        """All data lines joined as a single string for re-parsing."""
        return "\n".join(self.data_lines)


# Regex patterns
RE_COMMENT = re.compile(r"^\*\*")            # ** comment line
RE_KEYWORD = re.compile(r"^\*([\w-]+(?:[ \t]+[\w-]+)?)"  # *KEYWORD, *KEY-WORD, *END STEP
                        r"(?:,\s*(.*))?", re.IGNORECASE | re.MULTILINE)
RE_PARAM   = re.compile(r"(\w[\w.]*(?:\s+[\w.]+)*)\s*=\s*([^,]*)")


def _parse_keyword_params(param_text: str) -> dict:
    """Parse keyword parameter string into a dict.

    Example: "NAME=A1, TIME=STEP, VALUE=1.0" -> {'name': 'A1', 'time': 'STEP', 'value': '1.0'}
    """
    params = {}
    if not param_text or not param_text.strip():
        return params
    for part in param_text.split(","):
        part = part.strip()
        if not part:
            continue
        m = RE_PARAM.match(part)
        if m:
            key = m.group(1).lower()
            val = m.group(2).strip()
            params[key] = val
        else:
            # Bare keyword (no = sign) — treat as boolean flag
            bare = part.strip()
            if bare:
                params[bare.lower()] = "yes"
    return params


def _read_file_with_fallback(filepath: str) -> str:
    """Read file with utf-8 first, fallback to cp949 for Korean encoding."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(filepath, "r", encoding="cp949") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(filepath, "r", encoding="latin-1") as f:
                return f.read()


def tokenize(filepath: str) -> List[AbaqusKeywordBlock]:
    """Tokenize an Abaqus .inp file into a list of AbaqusKeywordBlock.

    Args:
        filepath: Path to the .inp file.

    Returns:
        List of AbaqusKeywordBlock tokens in order of appearance.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    text = _read_file_with_fallback(filepath)

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove inline comments (everything after ** on any line)
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        ci = line.find("**")
        if ci >= 0:
            cleaned.append(line[:ci])
        else:
            cleaned.append(line)
    text = "\n".join(cleaned)

    # --- Split into keyword blocks ---
    # Strategy: find all *KEYWORD lines, split text at those positions
    keyword_positions = []
    for m in RE_KEYWORD.finditer(text):
        keyword_positions.append((m.start(), m.group(1), m.group(2) or ""))

    blocks: List[AbaqusKeywordBlock] = []

    for idx, (start_pos, kw_name, kw_params) in enumerate(keyword_positions):
        # Determine the end of this block (next keyword or EOF)
        if idx + 1 < len(keyword_positions):
            end_pos = keyword_positions[idx + 1][0]
        else:
            end_pos = len(text)

        block_text = text[start_pos:end_pos]
        block_lines = block_text.split("\n")

        # Handle line continuation: lines ending with "," continue on next line
        # Abaqus allows data lines to continue with "-" continuation character
        # For simplicity, remove trailing continuation markers
        keyword_line = block_lines[0]
        data_raw = block_lines[1:] if len(block_lines) > 1 else []

        # Clean data lines: remove empty trailing lines
        while data_raw and not data_raw[-1].strip():
            data_raw.pop()

        # Unwrap continuation: if a data line ends with "-",
        # append next line (minus whitespace) and remove the "-"
        merged_data = []
        for dl in data_raw:
            stripped = dl.strip()
            if not stripped:
                if merged_data:
                    merged_data.append("")
                continue
            # Check for continuation: trailing "-" on a data line
            if stripped.endswith("-") and len(stripped) > 1 and not stripped.startswith("-"):
                merged_data.append(stripped[:-1].rstrip(",").rstrip())
            else:
                merged_data.append(stripped)

        parsed_params = _parse_keyword_params(kw_params)

        block = AbaqusKeywordBlock(
            keyword=kw_name.upper(),
            params=parsed_params,
            data_lines=merged_data,
            line_number=text[:start_pos].count("\n") + 1
        )
        blocks.append(block)

    return blocks


def tokenize_string(text: str, source_name: str = "<string>") -> List[AbaqusKeywordBlock]:
    """Tokenize an Abaqus .inp string directly (for testing)."""
    # Write to temp approach — but for testing, just process in-memory
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".inp", text=True)
    try:
        os.write(fd, text.encode("utf-8"))
        os.close(fd)
        return tokenize(path)
    finally:
        os.unlink(path)
