"""
tests/inp_corpus/test_corpus_baseline.py
========================================
Regression test verifying that .inp decks under benchmark_inp/ do not regress
in parsing status, element extraction, or error/gap count compared to baseline.json.

Modes:
- Fast mode (default, < 60s): Tests ~150 representative decks selected deterministically.
- Full mode (`pytest -m corpus_full`): Sweeps all 1895 decks.
"""

from __future__ import annotations

import json
import os
import pytest
from typing import Any, Dict, List

from tools.inp_corpus.sweep import (
    REPO_ROOT,
    BENCHMARK_DIR,
    DEFAULT_BASELINE_PATH,
    sweep_one_deck,
    run_sweep,
)


def load_baseline() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(DEFAULT_BASELINE_PATH):
        pytest.fail(
            f"Corpus baseline file not found at: {DEFAULT_BASELINE_PATH}\n"
            "Run `python -m tools.inp_corpus.sweep --update-baseline` first."
        )
    with open(DEFAULT_BASELINE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def select_fast_subset(all_paths: List[str]) -> List[str]:
    """
    Deterministic selection rule for ~150 representative decks:
    1. Priority matching: Decks containing representative benchmark keywords
       ('deck06', 'hertz', 'cook', 'rollup', 'cantilever', 'bending', 'contact',
        'gasket', 'seal', 'plastic').
    2. Even striding: Sample every Nth deck from the sorted list to ensure broad
       coverage across directories, element types, and historical decks.
    3. Return sorted unique list of ~150 decks.
    """
    all_sorted = sorted(all_paths)
    priority_keywords = (
        "deck06", "hertz", "cook", "rollup", "cantilever",
        "bending", "contact", "gasket", "seal", "plastic"
    )

    selected = set()
    # 1. Key benchmark decks
    for p in all_sorted:
        lower = p.lower()
        if any(kw in lower for kw in priority_keywords):
            selected.add(p)
            if len(selected) >= 75:
                break

    # 2. Stride sampling across the full set to sample broadly
    stride = max(1, len(all_sorted) // 80)
    for i in range(0, len(all_sorted), stride):
        selected.add(all_sorted[i])

    return sorted(selected)


BASELINE_DATA = load_baseline()
ALL_PATHS = sorted(BASELINE_DATA.keys())
FAST_PATHS = select_fast_subset(ALL_PATHS)


def _assert_no_regression(rel_path: str, current: Dict[str, Any], baseline: Dict[str, Any]) -> None:
    """Compare current deck sweep result against baseline record."""
    # 1. Parsing status
    if baseline.get("parse_ok", False):
        assert current["parse_ok"], (
            f"Regression in {rel_path}: Previously parsed OK, but now failed with error: {current.get('error')}"
        )
    
    # 2. Element types (all previously recognized types must still be recognized)
    base_elems = set(baseline.get("element_types", []))
    curr_elems = set(current.get("element_types", []))
    missing_elems = base_elems - curr_elems
    assert not missing_elems, (
        f"Regression in {rel_path}: Element types missing compared to baseline: {missing_elems}"
    )

    # 3. New gaps checking
    base_gaps = set(baseline.get("gaps", []))
    curr_gaps = set(current.get("gaps", []))
    new_gaps = curr_gaps - base_gaps
    assert not new_gaps, (
        f"Regression in {rel_path}: New gaps introduced compared to baseline: {new_gaps}"
    )


@pytest.mark.parametrize("rel_path", FAST_PATHS)
def test_deck_fast_subset(rel_path: str):
    """Fast mode: Regression check on deterministically chosen ~150 representative decks."""
    baseline_record = BASELINE_DATA[rel_path]
    full_path = os.path.join(BENCHMARK_DIR, rel_path.replace("/", os.sep))
    current_record = sweep_one_deck(full_path)
    _assert_no_regression(rel_path, current_record, baseline_record)


@pytest.mark.corpus_full
def test_full_corpus_sweep():
    """Full mode: Sweeps all 1895 decks under benchmark_inp/ in parallel and checks for regressions."""
    records = run_sweep(workers=6)
    regressions = []

    for r in records:
        rel_path = r["path"]
        if rel_path not in BASELINE_DATA:
            continue
        baseline_record = BASELINE_DATA[rel_path]
        try:
            _assert_no_regression(rel_path, r, baseline_record)
        except AssertionError as e:
            regressions.append(str(e))

    if regressions:
        sample_msg = "\n".join(regressions[:10])
        pytest.fail(f"Corpus full sweep detected {len(regressions)} regressions:\n{sample_msg}")
