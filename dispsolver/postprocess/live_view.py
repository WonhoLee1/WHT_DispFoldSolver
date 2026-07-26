"""
live_view.py
============
Bridge a saved `Result` (see `result_io.py`) into something
`PostprocessViewer`/`_ResultCache` (`viewer.py`) can read as if it were a
live `DynamicSolver` -- so a finished run can be inspected in the
interactive Qt viewer without holding the solver object, and without
re-solving.

Why an adapter rather than changing `viewer.py`'s data access
------------------------------------------------------------------
`_ResultCache.compute_field` and `PostprocessViewer._update_plot` read
exactly nine solver attributes plus `mesh.elements[eid].pid`.
`ResultSolverAdapter` reproduces that surface by slicing an already-
loaded `Result` -- no recomputation, no file I/O per step switch, and
`viewer.py` needs no `isinstance(solver, DynamicSolver)` relaxation
because it never checked the type to begin with.

Stress fields need more than displacement: `compute_field` calls
`mat.pk2_voigt(F, params, state)` on a real material object with the
element's actual internal-variable state. That only exists if the
result was saved with `material_objects=`/`state=`
(`ResultWriter`/`ex12_abaqus_inp_plate_fold.py::run_folding_from_result`)
-- i.e. pickle-backed results from a run made after this feature was
added. `Result.has_state` / `bool(result.material_objects)` tell you
which parts of a given result will actually work before you try.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import numpy as np

from .result_io import Result


class ResultSolverAdapter:
    """Read-only, solver-shaped view of one step of a `Result`.

    Exposes exactly what `viewer.py` reads: `.n_elem`, `.conn`,
    `.coords`, `.u`, `.state`, `.material`, `.materials`, `.mesh`
    (`.elements[eid].pid` only), `.elem_ids`, `.time`, `.n_dofs`.

    `.material`/`.materials` come from `result.material_objects` (a
    `{pid: MaterialModel}` dict) when present. `.material` is set to an
    arbitrary one of them only for API parity with `DynamicSolver`
    (which does the same thing, see `dynamic.py:643`) -- `viewer.py`'s
    per-pid material lookup (added alongside this adapter) uses
    `.materials`, not `.material`, for anything that actually varies by
    part.
    """

    def __init__(self, result: Result, step: int = -1):
        self.result = result
        self.step = step

        n_nodes = result.n_nodes
        self.n_dofs = 2 * n_nodes
        self.n_elem = result.n_cells
        self.coords = result.points[:, :2]
        self.elem_ids = result.element_ids
        self.time = float(result.times[result.n_steps + step if step < 0 else step])

        # solver.u is the flat (n_dofs,) nodal vector, in-plane only --
        # Result.displacement_flat already returns exactly that shape.
        self.u = result.displacement_flat(step)

        # solver.conn: (n_elem, 4) 0-based node indices per element.
        # Result stores VTK-style Connectivity/Offsets (variable-length
        # cells in general); every element here is a QUAD4, so this is a
        # reshape, not a general unpack.
        offs = result.offsets
        conn = result.connectivity
        n_per_elem = offs[1:] - offs[:-1]
        if self.n_elem and not np.all(n_per_elem == n_per_elem[0]):
            raise ValueError(
                "ResultSolverAdapter assumes a uniform element type "
                "(all QUAD4); this result has mixed element node counts"
            )
        self.conn = conn.reshape(self.n_elem, n_per_elem[0]) if self.n_elem else conn

        self.state = result.state(step) if result.has_state else None

        self.materials = dict(result.material_objects)
        self.material = next(iter(self.materials.values())) if self.materials else None
        # Numeric params per pid (mu/lambda_m/K, E/nu, ...) -- needed
        # alongside `.materials` for material classes whose pk2_voigt takes
        # params as an argument rather than storing them on the instance
        # (e.g. ViscoelasticMaterial's base material). `result.materials`
        # is the existing JSON-safe param summary, pid keys stringified by
        # `_plain()` at save time -- re-keyed to int here to match
        # `element_pid`'s dtype.
        self.material_params = {
            int(k): v for k, v in (result.materials or {}).items()
            if str(k).lstrip("-").isdigit()
        }

        # Tiny stand-in for `mesh.elements[eid].pid` -- the only mesh
        # access `viewer.py` makes (`viewer.py:718,535`).
        pid_by_eid = {int(eid): int(pid) for eid, pid in
                      zip(result.element_ids, result.element_pid)}
        self.mesh = SimpleNamespace(
            elements={eid: SimpleNamespace(pid=pid) for eid, pid in pid_by_eid.items()}
        )

    def __repr__(self) -> str:
        return (f"ResultSolverAdapter(step={self.step}, time={self.time:.4g}, "
                f"n_elem={self.n_elem}, has_state={self.state is not None}, "
                f"materials={sorted(self.materials)})")
