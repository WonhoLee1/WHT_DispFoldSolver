"""
test_cli_run_inp.py
===================
Verification suite for Abaqus .inp solver CLI runner & launcher engine.
"""

import os
import tempfile
import pytest

from dispsolver.cli.run_inp import run_inp_file, peek_dim_from_file


def test_cli_run_inp_simple_quad_vtkhdf():
    """Verify end-to-end execution of simple_quad.inp with VTKHDF output."""
    inp_path = os.path.join("tests", "fixtures", "simple_quad.inp")
    assert os.path.exists(inp_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        summary = run_inp_file(
            inp_path=inp_path,
            dim="auto",
            elem_jit="numba",
            export_format="vtkhdf",
            out_dir=tmpdir,
            verbose=False,
        )

        assert summary["converged"] is True
        assert summary["dimension"] == 2
        assert len(summary["exported_files"]) >= 1
        assert summary["exported_files"][0].endswith(".vtkhdf")
        assert os.path.exists(summary["exported_files"][0])

        json_path = os.path.join(tmpdir, "simple_quad_summary.json")
        assert os.path.exists(json_path)


def test_cli_run_inp_3d_transform_deck():
    """Verify CLI execution of 3D C3D8 transform deck."""
    deck = """*HEADING
3D Transform CLI Test Deck
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 1.0, 1.0, 0.0
4, 0.0, 1.0, 0.0
5, 0.0, 0.0, 1.0
6, 1.0, 0.0, 1.0
7, 1.0, 1.0, 1.0
8, 0.0, 1.0, 1.0
*ELEMENT, TYPE=C3D8, ELSET=EALL
1, 1, 2, 3, 4, 5, 6, 7, 8
*NSET, NSET=FIX_BACK
1, 4, 5, 8
*NSET, NSET=FRONT
2, 3, 6, 7
*TRANSFORM, NSET=FRONT, TYPE=R
1.0, 1.0, 0.0
*MATERIAL, NAME=STEEL3D
*ELASTIC
200000.0, 0.3
*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL3D
*STEP
*STATIC
*BOUNDARY
FIX_BACK, 1, 3, 0.0
FRONT, 2, 2, 0.0
*CLOAD
7, 1, 100.0
*END STEP
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        inp_file = os.path.join(tmpdir, "test_3d_cube.inp")
        with open(inp_file, "w", encoding="utf-8") as f:
            f.write(deck)

        summary = run_inp_file(
            inp_path=inp_file,
            dim="auto",
            elem_jit="numba",
            export_format="vtkhdf",
            out_dir=tmpdir,
            verbose=False,
        )

        assert summary["converged"] is True
        assert summary["dimension"] == 3
        assert len(summary["exported_files"]) >= 1
        assert summary["exported_files"][0].endswith(".vtkhdf")
