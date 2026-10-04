"""
tools/inp_corpus/sweep.py
=========================
Corpus regression harness: parses .inp decks under benchmark_inp/ in parallel
and records parsing status, errors, element types, NLGEOM flags, gaps, and warnings.
"""

from __future__ import annotations

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import glob
import json
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BENCHMARK_DIR = os.path.join(REPO_ROOT, "benchmark_inp")
DEFAULT_BASELINE_PATH = os.path.join(REPO_ROOT, "tests", "inp_corpus", "baseline.json")


def _extract_element_types(model: Any) -> List[str]:
    """Collect unique sorted element types across parts and instances in model."""
    types = set()
    if hasattr(model, "parts") and model.parts:
        for part in model.parts.values():
            if hasattr(part, "elements") and part.elements:
                for item in part.elements.values():
                    if isinstance(item, (tuple, list)) and len(item) > 0 and isinstance(item[0], str):
                        types.add(item[0].upper().strip())
                    elif hasattr(item, "type"):
                        types.add(str(item.type).upper().strip())

    if hasattr(model, "root_assembly") and model.root_assembly:
        instances = getattr(model.root_assembly, "instances", {})
        if isinstance(instances, dict):
            for inst in instances.values():
                if hasattr(inst, "elements") and inst.elements:
                    for item in inst.elements.values():
                        if isinstance(item, (tuple, list)) and len(item) > 0 and isinstance(item[0], str):
                            types.add(item[0].upper().strip())
                        elif hasattr(item, "type"):
                            types.add(str(item.type).upper().strip())
    return sorted(types)


def _extract_nlgeom(model: Any) -> List[bool]:
    """Collect NLGEOM boolean flags for all steps (excluding Initial)."""
    flags = []
    if hasattr(model, "steps") and model.steps:
        for name, step in model.steps.items():
            if str(name).lower() == "initial":
                continue
            flags.append(bool(getattr(step, "nlgeom", False)))
    return flags


def sweep_one_deck(full_path: str) -> Dict[str, Any]:
    """Parse one .inp deck and summarize results."""
    rel_path = os.path.relpath(full_path, BENCHMARK_DIR).replace("\\", "/")
    record: Dict[str, Any] = {
        "path": rel_path,
        "parse_ok": False,
        "error": None,
        "element_types": [],
        "nlgeom_steps": [],
        "gaps": [],
        "warnings": [],
    }

    try:
        from dispsolver.io.abaqus_reader import _parse_source, detect_dim
        dim = detect_dim(full_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader, model = _parse_source(full_path, dim=dim, strict=False)

        record["parse_ok"] = True
        record["element_types"] = _extract_element_types(model)
        record["nlgeom_steps"] = _extract_nlgeom(model)

        # Diagnostics / Gaps / Warnings
        if hasattr(reader, "diagnostics") and reader.diagnostics:
            # S0 Diagnostics object support
            diag_codes = collections.Counter(d.code for d in reader.diagnostics)
            record["diagnostics"] = dict(sorted(diag_codes.items()))
        else:
            gaps = reader.physics_gaps() if hasattr(reader, "physics_gaps") else []
            record["gaps"] = sorted(str(g) for g in gaps)
            warns = getattr(reader, "warnings", [])
            # Extract unique warning summaries
            clean_warns = set()
            for w in warns:
                w_str = str(w)
                if " at line" in w_str:
                    w_str = w_str.split(" at line")[0]
                clean_warns.add(w_str[:120])
            record["warnings"] = sorted(clean_warns)

    except Exception as e:
        record["parse_ok"] = False
        record["error"] = f"{type(e).__name__}: {str(e)[:200]}"

    return record


def run_sweep(
    inp_files: Optional[List[str]] = None,
    workers: int = 6,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Run sweep over the specified files (or all under benchmark_inp/)."""
    if inp_files is None:
        pattern = os.path.join(BENCHMARK_DIR, "**", "*.inp")
        inp_files = sorted(glob.glob(pattern, recursive=True))

    if limit is not None and limit > 0:
        inp_files = inp_files[:limit]

    print(f"Sweeping {len(inp_files)} decks with {workers} worker processes...")
    t0 = time.time()

    # Use multiprocessing ProcessPoolExecutor
    if workers > 1 and len(inp_files) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(sweep_one_deck, inp_files, chunksize=16))
    else:
        records = [sweep_one_deck(f) for f in inp_files]

    elapsed = time.time() - t0
    ok_count = sum(1 for r in records if r["parse_ok"])
    err_count = len(records) - ok_count
    print(f"Sweep complete in {elapsed:.2f}s: {ok_count} passed, {err_count} failed")
    return records


def main():
    parser = argparse.ArgumentParser(description="Abaqus INP Corpus Regression Sweep")
    parser.add_argument("--update-baseline", action="store_true", help="Rewrite tests/inp_corpus/baseline.json")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--workers", type=int, default=6, help="Worker processes (default: 6)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of decks for quick check")
    args = parser.parse_args()

    # Ensure REPO_ROOT is on sys.path
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    records = run_sweep(workers=args.workers, limit=args.limit)

    target_path = args.output
    if args.update_baseline:
        target_path = DEFAULT_BASELINE_PATH

    if target_path:
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        # Store as dict keyed by path for fast lookup
        data = {r["path"]: r for r in records}
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
        print(f"Saved {len(data)} baseline records to {target_path}")


if __name__ == "__main__":
    main()
