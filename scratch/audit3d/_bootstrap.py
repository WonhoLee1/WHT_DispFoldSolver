"""
Bootstrap to import dispsolver.element3d submodules directly, bypassing the
package's __init__.py, which currently (2026-09-12, mid-edit by the concurrent
3D session) raises NameError on import:

    dispsolver/element3d/c3d8_corotational_numba.py:198: NameError:
    name '_EMPTY_2D_CONTROLS' is not defined

(it's used as a parameter default at line 198 but only defined at line 411).
This is a READ-ONLY audit; we never edit that file. Instead we register a
stub package module in sys.modules with the correct __path__ so Python's
import machinery can still resolve `dispsolver.element3d.<submodule>` and
relative imports (`from .base3d import ...`) inside those submodules,
without ever executing the real (currently broken) __init__.py.
"""
import sys
import types
import os

ROOT = r"D:\PythonCodeStudy\WHT_DispFoldSolver"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import dispsolver  # noqa: E402  (parent package imports fine)

_PKG_DIR = os.path.join(ROOT, "dispsolver", "element3d")
if "dispsolver.element3d" not in sys.modules:
    stub = types.ModuleType("dispsolver.element3d")
    stub.__path__ = [_PKG_DIR]
    stub.__package__ = "dispsolver.element3d"
    sys.modules["dispsolver.element3d"] = stub

import re
import importlib.util


def load_element3d_module(modname):
    """Load dispsolver.element3d.<modname>, working around the concurrent
    session's in-progress edit that currently makes these files raise
    NameError on import (a module-level constant, e.g. _EMPTY_2D_CONTROLS,
    is used as a function-default at a line before its own module-level
    definition later in the same file). We never edit the file on disk;
    we only reorder the handful of `NAME = np.empty(...)` constant
    definitions to the top of an in-memory copy of the source before exec.
    """
    full_name = f"dispsolver.element3d.{modname}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    path = os.path.join(_PKG_DIR, f"{modname}.py")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    const_pat = re.compile(r"^(_EMPTY_\w+)\s*=\s*np\.empty\([^\n]*\)\s*$", re.MULTILINE)
    consts = const_pat.findall(src)
    if consts:
        # Pull every "NAME = np.empty(...)" line out and re-insert right after
        # the imports (before first @njit / def), so all such constants are
        # defined before any function default references them.
        lines = src.split("\n")
        const_lines = []
        remaining = []
        for line in lines:
            if const_pat.match(line):
                const_lines.append(line)
            else:
                remaining.append(line)
        # find insertion point: after the last top-level import line
        insert_at = 0
        for i, line in enumerate(remaining):
            if line.startswith("import ") or line.startswith("from "):
                insert_at = i + 1
        patched = remaining[:insert_at] + [""] + const_lines + [""] + remaining[insert_at:]
        src = "\n".join(patched)

    spec = importlib.util.spec_from_file_location(full_name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "dispsolver.element3d"
    sys.modules[full_name] = mod
    exec(compile(src, path, "exec"), mod.__dict__)
    setattr(sys.modules["dispsolver.element3d"], modname, mod)
    return mod


def load_full_element3d_package():
    """Populate the dispsolver.element3d stub with everything the real
    __init__.py exports, using load_element3d_module for the files the
    concurrent session's in-progress edit currently breaks (corotational,
    hybrid), and plain imports for the rest. Never touches any file on disk.
    """
    pkg = sys.modules["dispsolver.element3d"]
    if getattr(pkg, "_fully_loaded", False):
        return pkg

    from dispsolver.element3d.base3d import SolidElement3D, QuadraturePointState3D
    from dispsolver.element3d.c3d8_eas_jax import Hexa8EASElement
    from dispsolver.element3d.c3d8_fbar_jax import Hexa8FbarElement
    from dispsolver.element3d.c3d4_anp_jax import Tetra4ANPElement
    from dispsolver.element3d.c3d10m_jax import Tetra10Element

    corot = load_element3d_module("c3d8_corotational_numba")
    hybrid = load_element3d_module("c3d8_hybrid_numba")

    pkg.SolidElement3D = SolidElement3D
    pkg.QuadraturePointState3D = QuadraturePointState3D
    pkg.Hexa8EASElement = Hexa8EASElement
    pkg.Hexa8FbarElement = Hexa8FbarElement
    pkg.Tetra4ANPElement = Tetra4ANPElement
    pkg.Tetra10Element = Tetra10Element
    pkg.compute_c3d8_corotational_element_umat_numba = corot.compute_c3d8_corotational_element_umat_numba
    pkg.assemble_mesh_c3d8_corotational_numba = corot.assemble_mesh_c3d8_corotational_numba
    pkg.compute_element_rotation_3d = corot.compute_element_rotation_3d
    pkg.compute_c3d8_hybrid_element_umat_numba = hybrid.compute_c3d8_hybrid_element_umat_numba
    pkg.assemble_mesh_c3d8_hybrid_numba = hybrid.assemble_mesh_c3d8_hybrid_numba
    pkg._fully_loaded = True
    return pkg
