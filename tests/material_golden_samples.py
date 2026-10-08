"""
tests/material_golden_samples.py
================================
Sample materials + the single-element probe used to capture and re-check the
(id, props) every material gets in the 3D solver.  Imported by
tests/test_material_wiring.py and by the one-off generator of
tests/fixtures/material_ids_9f5e6e9.json (run with the OLD tree on PYTHONPATH).
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DECK_DIRS = [ROOT / "benchmark_inp" / "verification", ROOT / "benchmark_inp" / "official"]


def probe_3d(mat):
    """(id, props[64]) the 3D solver assigns to one C3D8 element of `mat`."""
    from dispsolver.mesh3d.mesh3d import Mesh3D
    from dispsolver.solver3d.dynamic3d import DynamicSolver3D
    mesh = Mesh3D()
    for i, (x, y, z) in enumerate([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                                   (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]):
        mesh.add_node(i + 1, x, y, z)
    mesh.add_element(1, [1, 2, 3, 4, 5, 6, 7, 8], "C3D8", pid=0)
    s = DynamicSolver3D(mesh, materials={0: mat})
    return int(s.elem_mat_types[0]), np.array(s.elem_props[0])


def deck_files():
    out = []
    for d in DECK_DIRS:
        out += sorted(glob.glob(str(d / "**" / "*.inp"), recursive=True))
    return out


def deck_materials(path):
    """{key: Material} for one deck, key = '<deck path rel. to repo>::<name>'."""
    from dispsolver.io.abaqus_reader import read_inp
    rel = os.path.relpath(path, ROOT).replace("\\", "/")
    try:
        model = read_inp(path, dim=3, strict=False)
    except Exception:
        return {}
    return {f"{rel}::{n}": m for n, m in model.materials.items()}
