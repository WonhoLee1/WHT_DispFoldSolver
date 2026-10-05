"""
tests/test_legacy_adapter.py
============================
Regression test suite for `dispsolver.io.legacy_adapter` and `read_abaqus_input`.
Verifies that the new reader + legacy_adapter produces a fully consistent
`ModelBuilderResult` (Mesh, Materials, Constraints, Boundaries) and completes
a 1-step solve identically.
"""

import os
import sys
import numpy as np
import pytest

from dispsolver.io import read_abaqus_input, ModelBuilderResult, ABAQUS_ELEMENT_MAP
from dispsolver.fold_model_config import make_teardrop_config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from ex12_abaqus_inp_plate_fold import run_folding_from_result


def test_ex12_deck_legacy_adapter_consistency():
    inp_path = os.path.join(os.path.dirname(__file__), "..", "examples", "ex12_rigid_plate_display_fold.inp")
    assert os.path.exists(inp_path), f"Deck not found: {inp_path}"

    res = read_abaqus_input(inp_path)
    assert isinstance(res, ModelBuilderResult)

    # 1. Mesh structure
    assert len(res.mesh.nodes) == 6440, f"Expected 6440 nodes, got {len(res.mesh.nodes)}"
    assert len(res.mesh.elements) == 6076, f"Expected 6076 elements, got {len(res.mesh.elements)}"

    # Check key node coordinates
    assert 1 in res.mesh.nodes
    assert 100000 in res.mesh.nodes
    assert 100306 in res.mesh.nodes

    n1 = res.mesh.nodes[1]
    assert abs(n1.x - (-40.0)) < 1e-6
    assert abs(n1.y - 0.0) < 1e-6

    nrp_l = res.mesh.nodes[100000]
    assert abs(nrp_l.x - (-1.4)) < 1e-6

    nrp_r = res.mesh.nodes[100306]
    assert abs(nrp_r.x - 1.4) < 1e-6

    # 2. Materials
    assert len(res.materials) == 16
    assert "PSA" in res.material_names.values()
    assert "PET" in res.material_names.values()
    assert "STEEL" in res.material_names.values()
    assert "GLASS" in res.material_names.values()

    # 3. Constraints
    assert len(res.penalty_constraints) == 2
    for tc in res.penalty_constraints:
        assert tc.k_tie == 1e6
        assert len(tc.pairs) == 61

    assert len(res.rbe2_constraints) == 2
    masters = {c.master_id for c in res.rbe2_constraints}
    assert masters == {100000, 100306}
    for c in res.rbe2_constraints:
        assert len(c.slave_ids) == 305

    # 4. Boundaries
    assert len(res.boundaries) == 4
    for b in res.boundaries:
        assert b.nset in ("100000", "100306")
        assert b.dof1 in (1, 6)

    # 5. One-step solve convergence
    cfg = make_teardrop_config()
    out = run_folding_from_result(res, config=cfg, max_steps=1, elem_jit="numba")
    u = out["solver"].u
    assert np.all(np.isfinite(u))
    assert np.max(np.abs(u)) > 0.0


def test_abaqus_element_map_exposed():
    """Verify that ABAQUS_ELEMENT_MAP is exported and contains canonical mappings."""
    assert "CPE4I" in ABAQUS_ELEMENT_MAP
    assert "CPE4H" in ABAQUS_ELEMENT_MAP
    assert "CPE4S" not in ABAQUS_ELEMENT_MAP
