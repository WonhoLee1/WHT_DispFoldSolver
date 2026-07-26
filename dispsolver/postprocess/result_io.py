"""
result_io.py
============
Persist a completed (or aborted) solve to disk and read it back for
analysis, plotting and animation without re-running the solver.

Why this exists
---------------
Before this module the only way to inspect a result was to hold the live
`DynamicSolver` object: `PostprocessViewer(solver)` /
`launch_from_solver(solver)` take a solver, not a file, and the VTKHDF
exporters are write-only (for ParaView) with no reader. Any new question
about a finished run -- "how much did the layers slip?", "animate the
fold" -- meant re-solving from scratch (~5 min for the ex12 display
fold). This module closes that loop.

Data layout -- modeled on VTKHDF
--------------------------------
VTKHDF already solves the transient-FE storage problem well, so the
in-memory payload borrows its shape rather than inventing a new one.
That also makes `to_vtkhdf()` a thin projection instead of a
translation:

    geometry    Points (n_nodes,3) reference/undeformed
                Connectivity / Offsets / Types   (static topology)
    model       NodeIds, ElementIds, ElementPid, ElementTypes,
                Materials, Constraints, Meta      <- see note below
    point_data  {name: (n_steps*n_nodes, ...)}    concatenated
    cell_data   {name: (n_steps*n_cells, ...)}    concatenated
    steps       times (n_steps,)

Per-step arrays are stored **concatenated**, exactly like VTKHDF, so a
VTKHDF file can be produced by copying slabs verbatim. Callers never do
the arithmetic: `Result.displacement(step)` and friends slice for you.

The `model` block is the deliberate addition. VTKHDF is a *rendering*
format -- 0-based point indices, no notion of original node ids, part
ids, materials or constraints. Analysis needs all of those (e.g. "which
rows are PSA?"), so they live here and are dropped on the way out to
VTKHDF.

Storage backends
----------------
Chosen by file extension:

    .pkl / .pickle   pickle  (default; fast, no schema to maintain)
    .h5 / .hdf5 / .foldres   HDF5 via h5py

The pickle payload is a **plain dict of numpy arrays and builtin types**,
never a class instance. Unpickling therefore does not depend on this
module's class definitions, so refactoring `Result` cannot invalidate
previously saved runs -- the usual reason pickle-based result files rot.
(Standard pickle caveat still applies: only load files you produced.)

Materials/constraints are stored as JSON-safe summaries, not live
objects, for the same reason.

Usage
-----
    from dispsolver.postprocess import ResultWriter, load_result

    writer = ResultWriter(mesh, meta={"case": "ex12"})
    for ...:
        solver.solve_step(dt)
        writer.add_step(solver.time, solver.u, scalars={"theta_deg": th})
    writer.save("run.pkl")            # also on abort -- partial is useful

    res = load_result("run.pkl")
    u_final = res.displacement()      # (n_nodes, 2), last step
    xy = res.deformed_points(step=10) # (n_nodes, 3)
"""

from __future__ import annotations

import datetime as _dt
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np

FORMAT_VERSION = (1, 0)
ROOT = "FoldResult"

# VTK cell type codes (same values VTKHDF uses)
VTK_TRIANGLE = 5
VTK_QUAD = 9

_PICKLE_EXT = {".pkl", ".pickle"}
_HDF5_EXT = {".h5", ".hdf5", ".foldres"}


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _vtk_type(elem) -> int:
    """VTK cell code for a mesh element, from its node count.

    Element-type *strings* here describe the formulation
    (Q4_COROTATIONAL, Q4_VISCO_SIMO, Q4_EAS, ...), not the geometry, and
    new ones get added regularly -- so the geometric cell type comes from
    the node count, and the formulation string is preserved separately in
    `model["ElementTypes"]`.
    """
    n = elem.n_nodes
    if n == 4:
        return VTK_QUAD
    if n == 3:
        return VTK_TRIANGLE
    raise ValueError(f"unsupported element with {n} nodes: {elem!r}")


def _plain(obj: Any) -> Any:
    """Best-effort conversion to picklable/JSON-safe builtin types.

    Material and constraint metadata holds live objects (NeoHookean(),
    ArrudaBoyce(), SurfaceTieConstraint()). Analysis only needs to
    *identify* them, so unknown objects degrade to their repr rather than
    dragging the whole solver object graph into the result file.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_plain(v) for v in obj]
    return repr(obj)


def _backend(path: str) -> str:
    ext = os.path.splitext(str(path))[1].lower()
    if ext in _PICKLE_EXT:
        return "pickle"
    if ext in _HDF5_EXT:
        return "hdf5"
    raise ValueError(
        f"cannot infer result format from {path!r}; use one of "
        f"{sorted(_PICKLE_EXT | _HDF5_EXT)}"
    )


# ----------------------------------------------------------------------
# writer
# ----------------------------------------------------------------------

class ResultWriter:
    """Accumulate solve steps in memory; write one result file on save.

    Nothing touches the filesystem until `save()`. That is deliberate:
    holding a file handle open for a multi-minute solve means a crashed
    run leaves the path locked (Win32 sharing violation) and the next run
    cannot create it -- a failure mode `TransientVTKHDFExporter` already
    had to work around.
    """

    def __init__(
        self,
        mesh,
        materials: Optional[Dict] = None,
        constraints: Optional[Dict] = None,
        meta: Optional[Dict] = None,
    ):
        node_ids = mesh.node_ids()
        self.n_nodes = len(node_ids)

        pts = np.asarray(mesh.nodes_array(), dtype=np.float64)
        points = np.zeros((self.n_nodes, 3), dtype=np.float64)
        points[:, :2] = pts

        nid_to_idx = mesh.node_id_to_index()
        conn: List[int] = []
        offs: List[int] = [0]
        types: List[int] = []
        eids: List[int] = []
        pids: List[int] = []
        etypes: List[str] = []
        for eid, elem in mesh.elements.items():
            types.append(_vtk_type(elem))
            conn.extend(nid_to_idx[n] for n in elem.node_ids)
            offs.append(offs[-1] + elem.n_nodes)
            eids.append(eid)
            pids.append(getattr(elem, "pid", 0))
            etypes.append(str(getattr(elem, "elem_type", "")))

        self.n_cells = len(eids)
        self._geometry = {
            "Points": points,
            "Connectivity": np.asarray(conn, dtype=np.int64),
            "Offsets": np.asarray(offs, dtype=np.int64),
            "Types": np.asarray(types, dtype=np.uint8),
        }
        self._model = {
            "NodeIds": np.asarray(node_ids, dtype=np.int64),
            "ElementIds": np.asarray(eids, dtype=np.int64),
            "ElementPid": np.asarray(pids, dtype=np.int64),
            "ElementTypes": etypes,
            "Materials": _plain(materials or {}),
            "Constraints": _plain(constraints or {}),
            "Meta": _plain(meta or {}),
        }

        self._times: List[float] = []
        self._scalars: Dict[str, List[float]] = {}
        self._point_data: Dict[str, List[np.ndarray]] = {}
        self._cell_data: Dict[str, List[np.ndarray]] = {}
        self._arrays: Dict[str, np.ndarray] = {}

    # -- accumulation ---------------------------------------------------

    def update_meta(self, **kw) -> None:
        """Add/overwrite run metadata (e.g. whether it reached the target).

        Meta is written at `save()` time, so this can be called after the
        solve loop with things only known at the end.
        """
        self._model["Meta"].update(_plain(kw))

    def add_array(self, name: str, arr) -> None:
        """Attach a run-level array that is neither per-node nor per-cell.

        For derived histories such as the per-layer interlayer-slip table
        (n_steps x n_layers), which has no home in point/cell data but
        should travel with the result.
        """
        self._arrays[name] = np.asarray(arr)

    def add_step(
        self,
        time: float,
        u: np.ndarray,
        point_data: Optional[Dict[str, np.ndarray]] = None,
        cell_data: Optional[Dict[str, np.ndarray]] = None,
        scalars: Optional[Dict[str, float]] = None,
    ) -> None:
        """Buffer one converged increment. No file I/O happens here.

        `scalars` holds per-step single values (drive angle, iteration
        count, ...) that belong to the step rather than to a node or an
        element.
        """
        u = np.asarray(u, dtype=np.float64).reshape(-1)
        if u.size < 2 * self.n_nodes:
            raise ValueError(
                f"displacement has {u.size} entries, need at least "
                f"{2 * self.n_nodes} for {self.n_nodes} nodes"
            )
        # Solvers carry extra DOFs (Lagrange multipliers, RBE2 rotations)
        # past the nodal block; keep only the nodal part.
        u3 = np.zeros((self.n_nodes, 3), dtype=np.float64)
        u3[:, :2] = u[: 2 * self.n_nodes].reshape(self.n_nodes, 2)

        self._times.append(float(time))
        self._point_data.setdefault("Displacement", []).append(u3)

        for name, arr in (point_data or {}).items():
            self._point_data.setdefault(name, []).append(np.asarray(arr, dtype=np.float64))
        for name, arr in (cell_data or {}).items():
            self._cell_data.setdefault(name, []).append(np.asarray(arr, dtype=np.float64))
        for name, val in (scalars or {}).items():
            self._scalars.setdefault(name, []).append(float(val))

    @property
    def n_steps(self) -> int:
        return len(self._times)

    # -- output ---------------------------------------------------------

    def payload(self) -> Dict[str, Any]:
        """The complete result as a plain dict (what actually gets saved)."""
        return {
            "format": ROOT,
            "version": FORMAT_VERSION,
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "n_nodes": self.n_nodes,
            "n_cells": self.n_cells,
            "n_steps": self.n_steps,
            "geometry": self._geometry,
            "model": self._model,
            "times": np.asarray(self._times, dtype=np.float64),
            "scalars": {k: np.asarray(v, dtype=np.float64)
                        for k, v in self._scalars.items()},
            "point_data": {k: np.concatenate(v, axis=0)
                           for k, v in self._point_data.items()},
            "cell_data": {k: np.concatenate(v, axis=0)
                          for k, v in self._cell_data.items()},
            "arrays": dict(self._arrays),
        }

    def save(self, filepath: str) -> str:
        """Write the buffered result. Backend chosen by file extension."""
        payload = self.payload()
        if _backend(filepath) == "pickle":
            _save_pickle(payload, filepath)
        else:
            _save_hdf5(payload, filepath)
        return str(filepath)

    def result(self) -> "Result":
        """The in-memory `Result` without a round trip through disk."""
        return Result(self.payload())


# ----------------------------------------------------------------------
# serializers
# ----------------------------------------------------------------------

def _save_pickle(payload: Dict[str, Any], filepath: str) -> None:
    with open(filepath, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(filepath: str) -> Dict[str, Any]:
    with open(filepath, "rb") as f:
        return pickle.load(f)


def _save_hdf5(payload: Dict[str, Any], filepath: str) -> None:
    import json

    import h5py as h5

    with h5.File(filepath, "w") as f:
        g = f.create_group(ROOT)
        g.attrs["Version"] = np.asarray(payload["version"], dtype=np.int32)
        g.attrs["Created"] = payload["created"]
        g.attrs["Type"] = "UnstructuredGrid"

        geo = g.create_group("Geometry")
        for k, v in payload["geometry"].items():
            geo.create_dataset(k, data=v)

        mdl = g.create_group("Model")
        for k in ("NodeIds", "ElementIds", "ElementPid"):
            mdl.create_dataset(k, data=payload["model"][k])
        mdl.create_dataset(
            "ElementTypes",
            data=np.asarray(payload["model"]["ElementTypes"],
                            dtype=h5.string_dtype("utf-8")),
        )
        for k in ("Materials", "Constraints", "Meta"):
            mdl.attrs[k] = json.dumps(payload["model"][k], ensure_ascii=True)

        steps = g.create_group("Steps")
        steps.attrs["NSteps"] = np.int32(payload["n_steps"])
        steps.create_dataset("Values", data=payload["times"])
        sc = steps.create_group("Scalars")
        for k, v in payload["scalars"].items():
            sc.create_dataset(k, data=v)

        pd = g.create_group("PointData")
        for k, v in payload["point_data"].items():
            pd.create_dataset(k, data=v)
        cd = g.create_group("CellData")
        for k, v in payload["cell_data"].items():
            cd.create_dataset(k, data=v)
        ar = g.create_group("Arrays")
        for k, v in payload.get("arrays", {}).items():
            ar.create_dataset(k, data=v)


def _load_hdf5(filepath: str) -> Dict[str, Any]:
    import json

    import h5py as h5

    with h5.File(filepath, "r") as f:
        if ROOT not in f:
            raise ValueError(f"{filepath!r} is not a {ROOT} file")
        g = f[ROOT]
        model = {
            k: np.asarray(g["Model"][k])
            for k in ("NodeIds", "ElementIds", "ElementPid")
        }
        model["ElementTypes"] = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in np.asarray(g["Model"]["ElementTypes"])
        ]
        for k in ("Materials", "Constraints", "Meta"):
            raw = g["Model"].attrs.get(k, "{}")
            model[k] = json.loads(raw.decode() if isinstance(raw, bytes) else raw)

        return {
            "format": ROOT,
            "version": tuple(int(v) for v in g.attrs["Version"]),
            "created": str(g.attrs.get("Created", "")),
            "n_nodes": int(np.asarray(g["Geometry"]["Points"]).shape[0]),
            "n_cells": int(len(model["ElementIds"])),
            "n_steps": int(g["Steps"].attrs["NSteps"]),
            "geometry": {k: np.asarray(v) for k, v in g["Geometry"].items()},
            "model": model,
            "times": np.asarray(g["Steps"]["Values"]),
            "scalars": {k: np.asarray(v)
                        for k, v in g["Steps"].get("Scalars", {}).items()},
            "point_data": {k: np.asarray(v) for k, v in g["PointData"].items()},
            "cell_data": {k: np.asarray(v) for k, v in g["CellData"].items()},
            "arrays": {k: np.asarray(v) for k, v in g.get("Arrays", {}).items()},
        }


# ----------------------------------------------------------------------
# reader
# ----------------------------------------------------------------------

class Result:
    """A finished (or aborted) solve, loaded from disk. Read-only.

    Thin accessor over the plain payload dict -- the dict, not this
    object, is what gets serialized, so this class can be changed freely
    without invalidating saved files.
    """

    def __init__(self, payload: Dict[str, Any], path: str = "<memory>"):
        if payload.get("format") != ROOT:
            raise ValueError(f"{path!r} is not a {ROOT} payload")
        self.path = str(path)
        self._p = payload

        self.version = tuple(payload["version"])
        self.created = payload["created"]
        self.n_nodes = int(payload["n_nodes"])
        self.n_cells = int(payload["n_cells"])
        self.n_steps = int(payload["n_steps"])

        geo = payload["geometry"]
        self.points = geo["Points"]
        self.connectivity = geo["Connectivity"]
        self.offsets = geo["Offsets"]
        self.types = geo["Types"]

        mdl = payload["model"]
        self.node_ids = mdl["NodeIds"]
        self.element_ids = mdl["ElementIds"]
        self.element_pid = mdl["ElementPid"]
        self.element_types = mdl["ElementTypes"]
        self.materials = mdl["Materials"]
        self.constraints = mdl["Constraints"]
        self.meta = mdl["Meta"]

        self.times = payload["times"]
        self.scalars = payload["scalars"]
        self._point_data = payload["point_data"]
        self._cell_data = payload["cell_data"]
        self.arrays = payload.get("arrays", {})

        self._nid_to_idx = {int(n): i for i, n in enumerate(self.node_ids)}

    # -- lookups --------------------------------------------------------

    def node_index(self, nid: int) -> int:
        return self._nid_to_idx[int(nid)]

    @property
    def point_data_names(self) -> List[str]:
        return sorted(self._point_data)

    @property
    def cell_data_names(self) -> List[str]:
        return sorted(self._cell_data)

    @property
    def scalar_names(self) -> List[str]:
        return sorted(self.scalars)

    @property
    def array_names(self) -> List[str]:
        return sorted(self.arrays)

    def _slab(self, store, name: str, step: int, rows: int) -> np.ndarray:
        if name not in store:
            raise KeyError(f"{name!r} not in this result; have {sorted(store)}")
        s = self.n_steps + step if step < 0 else step
        if not 0 <= s < self.n_steps:
            raise IndexError(f"step {step} out of range (n_steps={self.n_steps})")
        return store[name][s * rows: (s + 1) * rows]

    # -- fields ---------------------------------------------------------

    def point_data(self, name: str, step: int = -1) -> np.ndarray:
        return self._slab(self._point_data, name, step, self.n_nodes)

    def cell_data(self, name: str, step: int = -1) -> np.ndarray:
        return self._slab(self._cell_data, name, step, self.n_cells)

    def displacement(self, step: int = -1) -> np.ndarray:
        """(n_nodes, 2) in-plane displacement at `step` (default: last)."""
        return self.point_data("Displacement", step)[:, :2]

    def displacement_flat(self, step: int = -1) -> np.ndarray:
        """Solver-style flat (2*n_nodes,) displacement vector."""
        return self.displacement(step).reshape(-1)

    def deformed_points(self, step: int = -1) -> np.ndarray:
        """(n_nodes, 3) reference geometry + displacement at `step`."""
        out = self.points.copy()
        out[:, :2] += self.displacement(step)
        return out

    def elements_with_pid(self, pid: int) -> np.ndarray:
        return self.element_ids[self.element_pid == pid]

    def save(self, filepath: str) -> str:
        """Re-serialize (e.g. pkl -> h5) without going through a solve."""
        if _backend(filepath) == "pickle":
            _save_pickle(self._p, filepath)
        else:
            _save_hdf5(self._p, filepath)
        return str(filepath)

    def __repr__(self) -> str:
        t0 = self.times[0] if self.n_steps else float("nan")
        t1 = self.times[-1] if self.n_steps else float("nan")
        return (
            f"Result({os.path.basename(self.path)!r}: {self.n_nodes} nodes, "
            f"{self.n_cells} cells, {self.n_steps} steps, t={t0:.4g}..{t1:.4g}, "
            f"point_data={self.point_data_names}, scalars={self.scalar_names})"
        )


def load_result(path: str) -> Result:
    """Load a result written by `ResultWriter.save` (.pkl or .h5)."""
    payload = _load_pickle(path) if _backend(path) == "pickle" else _load_hdf5(path)
    return Result(payload, path=path)


# ----------------------------------------------------------------------
# projection to real VTKHDF (for ParaView)
# ----------------------------------------------------------------------

def to_vtkhdf(result: Result, filepath: str) -> str:
    """Write a loaded `Result` out as a genuine VTKHDF file.

    Because the payload layout above is VTKHDF-shaped this is mostly a
    regrouping: topology and the concatenated per-step slabs are copied
    as-is, `Points` is expanded to one deformed block per step (so
    ParaView shows the folded shape without a Warp By Vector filter), and
    the `model` block is dropped except for `Pid` -- VTKHDF has nowhere
    to keep node ids, materials or constraints.
    """
    import h5py as h5

    n_steps, n_nodes, n_cells = result.n_steps, result.n_nodes, result.n_cells
    deformed = np.concatenate(
        [result.deformed_points(s) for s in range(n_steps)], axis=0
    ).astype(np.float32)

    with h5.File(filepath, "w") as f:
        g = f.create_group("VTKHDF")
        g.attrs["Version"] = np.array([2, 2], dtype=np.int32)
        ascii_type = b"UnstructuredGrid"
        g.attrs.create("Type", ascii_type,
                       dtype=h5.string_dtype("ascii", len(ascii_type)))

        g.create_dataset("Points", data=deformed)
        g.create_dataset("Connectivity", data=result.connectivity.astype(np.int64))
        g.create_dataset("Offsets", data=result.offsets.astype(np.int64))
        g.create_dataset("Types", data=result.types.astype(np.uint8))
        g.create_dataset("NumberOfPoints", data=np.full(n_steps, n_nodes, dtype=np.int64))
        g.create_dataset("NumberOfCells", data=np.full(n_steps, n_cells, dtype=np.int64))
        g.create_dataset("NumberOfConnectivityIds",
                         data=np.full(n_steps, result.connectivity.size, dtype=np.int64))

        pd = g.create_group("PointData")
        for name in result.point_data_names:
            pd.create_dataset(name, data=np.concatenate(
                [result.point_data(name, s) for s in range(n_steps)], axis=0
            ).astype(np.float32))

        cd = g.create_group("CellData")
        for name in result.cell_data_names:
            cd.create_dataset(name, data=np.concatenate(
                [result.cell_data(name, s) for s in range(n_steps)], axis=0
            ).astype(np.float32))
        # pid is the one model array with a natural CellData home -- it is
        # what you colour by in ParaView to see the PET/PSA layup. Tiled
        # per step so it slabs like every other cell field.
        cd.create_dataset("Pid",
                          data=np.tile(result.element_pid.astype(np.int32), n_steps))

        steps = g.create_group("Steps")
        steps.attrs["NSteps"] = np.int32(n_steps)
        steps.create_dataset("Values", data=result.times.astype(np.float32))
        steps.create_dataset("PartOffsets", data=np.zeros(n_steps, dtype=np.int64))
        steps.create_dataset("PointOffsets",
                             data=np.arange(n_steps, dtype=np.int64) * n_nodes)
        steps.create_dataset("CellOffsets", data=np.zeros(n_steps, dtype=np.int64))
        steps.create_dataset("ConnectivityIdOffsets", data=np.zeros(n_steps, dtype=np.int64))

        p_off = np.arange(n_steps, dtype=np.int64) * n_nodes
        c_off = np.arange(n_steps, dtype=np.int64) * n_cells
        pdo = steps.create_group("PointDataOffsets")
        for name in result.point_data_names:
            pdo.create_dataset(name, data=p_off)
        cdo = steps.create_group("CellDataOffsets")
        for name in result.cell_data_names:
            cdo.create_dataset(name, data=c_off)
        cdo.create_dataset("Pid", data=c_off)

    return str(filepath)
