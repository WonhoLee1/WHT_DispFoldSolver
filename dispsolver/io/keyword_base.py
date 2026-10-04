from __future__ import annotations

from typing import Any, List

def _parse_data_rows(data_lines: List[str], dtype=float, default: Any = None) -> List[List[Any]]:
    """Parse comma/whitespace-separated data lines into a table of values.
    Supports Abaqus trailing comma ',' line continuation.
    Preserves blank/empty fields between commas as default (or empty string if dtype is str).
    """
    merged_lines = []
    curr = ""
    for line in data_lines:
        line_str = line.strip()
        if not line_str:
            continue
        if curr:
            curr = curr + " " + line_str
        else:
            curr = line_str

        if curr.endswith(","):
            continue
        else:
            merged_lines.append(curr)
            curr = ""
    if curr:
        merged_lines.append(curr)

    rows = []
    for line_str in merged_lines:
        if "," in line_str:
            parts = [p.strip() for p in line_str.split(",")]
            while parts and parts[-1] == "":
                parts.pop()
        else:
            parts = [p.strip() for p in line_str.split() if p.strip()]
        if not parts:
            continue
        row = []
        for p in parts:
            if p == "":
                row.append("" if dtype is str else default)
            else:
                try:
                    row.append(dtype(p))
                except (ValueError, TypeError):
                    row.append(p)
        rows.append(row)
    return rows


class _AbortParse(Exception):
    """Hard parse abort for keywords with no internal landing.

    Re-raised by parse_blocks: continuing past e.g. *INCLUDE would
    silently solve a different model than the deck describes.
    """


