from __future__ import annotations

from typing import Any, List

def _parse_data_rows(data_lines: List[str], dtype=float) -> List[List[Any]]:
    """Parse comma/whitespace-separated data lines into a table of values."""
    rows = []
    for line in data_lines:
        line_str = line.strip()
        if not line_str:
            continue
        parts = [p.strip() for p in line_str.split(",") if p.strip()]
        if not parts:
            # Fallback to whitespace split if no commas present
            parts = [p.strip() for p in line_str.split() if p.strip()]
        if not parts:
            continue
        row = []
        for p in parts:
            try:
                row.append(dtype(p))
            except ValueError:
                row.append(p)  # string
        rows.append(row)
    return rows


class _AbortParse(Exception):
    """Hard parse abort for keywords with no internal landing.

    Re-raised by parse_blocks: continuing past e.g. *INCLUDE would
    silently solve a different model than the deck describes.
    """


