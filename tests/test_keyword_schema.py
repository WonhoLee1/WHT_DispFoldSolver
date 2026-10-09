"""Tests for Phase F2: Declarative keyword schemas, parameter validation, and blank row preservation."""

from __future__ import annotations

import pytest

from dispsolver.io.keyword_registry import (
    GLOBAL_KEYWORD_REGISTRY,
    KeywordContext,
    KeywordSpec,
    ParamSpec,
    RowSpec,
    normalize_param_name,
)
from dispsolver.io.keyword_base import _parse_data_rows
from dispsolver.io.abaqus_reader import AbaqusInputReader, _parse_source


def test_all_68_keywords_have_complete_spec():
    """Verify that all 68 canonical keywords are registered with complete schemas."""
    import dispsolver.io  # ensure all keyword modules are imported
    specs = GLOBAL_KEYWORD_REGISTRY.all_specs()
    assert len(specs) >= 68, f"Expected at least 68 canonical keywords, found {len(specs)}"

    valid_support_levels = {"FULL", "PARTIAL", "STUB", "IGNORED"}

    for name, spec in specs.items():
        assert isinstance(spec, KeywordSpec), f"Spec for {name} is not KeywordSpec"
        assert spec.name == name, f"Spec name mismatch: {spec.name} vs {name}"
        assert spec.support in valid_support_levels, f"{name}: invalid support '{spec.support}'"
        assert isinstance(spec.valid_contexts, (set, frozenset)), f"{name}: valid_contexts not set"
        assert len(spec.valid_contexts) > 0, f"{name}: valid_contexts is empty"
        assert isinstance(spec.params, dict), f"{name}: params is not dict"
        assert len(spec.description.strip()) > 0, f"{name}: description is empty"

        # Verify each registered parameter spec
        for p_name, p_spec in spec.params.items():
            assert isinstance(p_spec, ParamSpec), f"{name}.{p_name} is not ParamSpec"
            assert isinstance(p_spec.name, str) and len(p_spec.name) > 0
            assert isinstance(p_spec.supported, bool)
            assert isinstance(p_spec.aliases, list)

        if spec.rows is not None:
            assert isinstance(spec.rows, RowSpec), f"{name}: rows is not RowSpec"
            assert spec.rows.min_columns >= 0
            if spec.rows.max_columns is not None:
                assert spec.rows.max_columns >= spec.rows.min_columns


def test_param_spec_aliases_and_normalization():
    """Verify normalize_param_name and parameter alias resolution across keywords."""
    assert normalize_param_name("ref-node") == "REF NODE"
    assert normalize_param_name("REF_NODE") == "REF NODE"
    assert normalize_param_name("  ref   node  ") == "REF   NODE" or normalize_param_name(" ref_node ") == "REF NODE"

    # Abbreviations (MAT, ORI, GEN, AMP) are resolved by Abaqus's rule -- any
    # unambiguous prefix -- against the keyword index (dispsolver/io/
    # keyword_index.py), no longer by per-keyword alias entries.
    from dispsolver.io.keyword_index import abaqus_index
    ix = abaqus_index()
    for kw, full, abbreviations in (("SOLID SECTION", "MATERIAL", ("MAT",)),
                                    ("SOLID SECTION", "ORIENTATION", ("ORIENT", "ORI")),
                                    ("NSET", "GENERATE", ("GEN",)),
                                    ("CLOAD", "AMPLITUDE", ("AMP",))):
        spec = GLOBAL_KEYWORD_REGISTRY.get_handler(kw)
        assert spec is not None and full in spec.params
        for ab in abbreviations:
            assert ix.get(kw).resolve_param(ab) == full, (kw, ab)


def test_blank_column_preservation_in_parse_data_rows():
    """Verify that _parse_data_rows preserves blanks (empty fields) between commas as None or default."""
    # Numeric rows with internal blank
    rows = _parse_data_rows(["1, 2, , 4"], dtype=float)
    assert len(rows) == 1
    assert rows[0] == [1.0, 2.0, None, 4.0]

    # String rows with leading blank (e.g. , GRAV, 9.81)
    rows_str = _parse_data_rows([", GRAV, 9.81"], dtype=str)
    assert len(rows_str) == 1
    assert rows_str[0] == ["", "GRAV", "9.81"]

    # Trailing empty comma from line continuation should not produce extra trailing elements
    rows_cont = _parse_data_rows(["10.0, 20.0, "], dtype=float)
    assert len(rows_cont) == 1
    assert rows_cont[0] == [10.0, 20.0]

    # Multiple blank columns preserved
    rows_multi = _parse_data_rows(["1, , , 4"], dtype=float)
    assert len(rows_multi) == 1
    assert rows_multi[0] == [1.0, None, None, 4.0]


def test_static_step_blank_time_incrementation():
    """Verify *STATIC row '0.1, 1.0, , 0.1' parses without column shifting.
    
    Abaqus convention: initial_inc=0.1, time_period=1.0, min_inc=default (None/1e-5), max_inc=0.1.
    If blanks were dropped, max_inc (0.1) would have shifted into min_inc!
    """
    deck = """
*HEADING
Test Static Blanks
*NODE
1, 0.0, 0.0
2, 1.0, 0.0
3, 1.0, 1.0
4, 0.0, 1.0
*ELEMENT, TYPE=CPS4, ELSET=EALL
1, 1, 2, 3, 4
*MATERIAL, NAME=MAT1
*ELASTIC
1000.0, 0.3
*SOLID SECTION, ELSET=EALL, MATERIAL=MAT1
*STEP, NAME=STEP1
*STATIC
0.1, 1.0, , 0.05
*END STEP
"""
    reader, model = _parse_source(deck, dim=2)
    step = model.steps["STEP1"]
    assert step.dt_init == 0.1
    assert step.time_period == 1.0
    # min_inc was omitted (blank), so it retains default 1e-5
    assert step.dt_min == 1e-5
    # max_inc is 0.05 (4th column), NOT shifted into min_inc
    assert step.dt_max == 0.05


def test_unsupported_and_unknown_parameters_detection():
    """Verify that unsupported parameters and unknown parameters are recorded in reader.unsupported_params."""
    deck = """
*HEADING
Test Unsupported Params
*NODE
1, 0.0, 0.0
2, 1.0, 0.0
3, 1.0, 1.0
4, 0.0, 1.0
*ELEMENT, TYPE=CPS4, ELSET=EALL
1, 1, 2, 3, 4
*MATERIAL, NAME=MAT1
*ELASTIC, COMPRESSION FACTOR=2.0
1000.0, 0.3
*SOLID SECTION, ELSET=EALL, MATERIAL=MAT1
*STEP, NAME=STEP1, FOOBAR_NONEXISTENT=123
*STATIC
0.1, 1.0
*END STEP
"""
    reader, model = _parse_source(deck, dim=2)

    # Check unknown parameter on STEP
    unknown_entries = [p for p in reader.unsupported_params if p[2] == "unknown"]
    assert any(p[0] == "STEP" and "FOOBAR_NONEXISTENT" in p[1].upper() for p in unknown_entries)

    # Check unsupported parameter on ELASTIC (COMPRESSION FACTOR is supported=False)
    unsupported_entries = [p for p in reader.unsupported_params if p[2] == "unsupported"]
    assert any(p[0] == "ELASTIC" and "COMPRESSION" in p[1].upper() for p in unsupported_entries)

    # Verify physics_gaps reflects this gap
    gaps = reader.physics_gaps()
    assert any("COMPRESSION" in g for g in gaps)
    assert any("FOOBAR_NONEXISTENT" in g for g in gaps)


def test_missing_required_parameter_warning():
    """Verify that omitting a required parameter generates a diagnostic warning."""
    deck = """
*HEADING
Test Missing Required Param
*NODE
1, 0.0, 0.0
2, 1.0, 0.0
3, 1.0, 1.0
4, 0.0, 1.0
*ELEMENT, TYPE=CPS4
1, 1, 2, 3, 4
"""
    # *ELEMENT requires TYPE parameter (ELSET is optional).
    # What if we have *SOLID SECTION without MATERIAL?
    deck_no_mat = """
*HEADING
Missing Material on Section
*NODE
1, 0.0, 0.0
2, 1.0, 0.0
3, 1.0, 1.0
4, 0.0, 1.0
*ELEMENT, TYPE=CPS4, ELSET=EALL
1, 1, 2, 3, 4
*SOLID SECTION, ELSET=EALL
"""
    reader, model = _parse_source(deck_no_mat, dim=2)
    assert any("missing required parameter 'MATERIAL'" in w for w in reader.warnings)
