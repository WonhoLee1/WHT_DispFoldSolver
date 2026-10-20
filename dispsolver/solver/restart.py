"""Restart files: write a solve out, read it back into a NEW solver, continue.

What a restart carries is exactly what an increment of ``solve()`` mutates and the next one reads:
displacement, velocity, acceleration, the element state variables, the constraint states that expose
``capture_state()`` (the same protocol the cutback rollback uses), the NLGEOM state, the temperature
field, the step clock (name, step time, increment counters) and the time-increment controller.

The file is a plain dict of numpy arrays and builtin types, never a class instance, so refactoring the
solver cannot invalidate a restart written earlier (the same rule ``postprocess.result_io`` follows).
It does not carry the model: the mesh, materials and sections come from the input deck, and a model
fingerprint (sizes and hashes of the coordinate, element-type and property arrays) refuses to put the
state onto a different model.

Constraints without ``capture_state`` are recorded by name; ``apply`` returns them as warnings, because
whatever history they hold (friction slip, an active contact set) starts fresh.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np

VERSION = 1


def _hash(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _dim(solver) -> int:
    return int(solver.num_dofs // solver.num_nodes)


def fingerprint(solver) -> Dict[str, Any]:
    """Sizes and hashes that identify the model a state belongs to."""
    coords = np.asarray(solver.mesh.nodes_array(), dtype=np.float64)
    return {
        "dim": _dim(solver),
        "num_dofs": int(solver.num_dofs),
        "num_elems": int(solver.elem_mat_types.shape[0]),
        "coords": _hash(coords),
        "elements": _hash(solver.elem_mat_types, solver.elem_props),
    }


def _copy(a):
    return None if a is None else np.array(a, copy=True)


def capture(solver, step_name: str, step_time: float, step_inc: int, inc_no: int,
            total_iters: int, dt_ctrl: Optional[Any]) -> Dict[str, Any]:
    """Snapshot everything needed to continue ``step_name`` from ``step_time``."""
    constraint_states, stateless = [], []
    for i, c in enumerate(solver.constraints):
        name = getattr(c, "name", None) or type(c).__name__
        if hasattr(c, "capture_state"):
            constraint_states.append({"index": i, "name": str(name), "state": c.capture_state()})
        else:
            stateless.append(str(name))
    return {
        "version": VERSION,
        "fingerprint": fingerprint(solver),
        "step": {"name": str(step_name), "time": float(step_time), "inc": int(step_inc),
                 "inc_no": int(inc_no), "total_iters": int(total_iters)},
        "u": _copy(solver.u), "v": _copy(getattr(solver, "v", None)), "a": _copy(getattr(solver, "a", None)),
        "u_step_start": _copy(getattr(solver, "_u_step_start", None)),
        "elem_sdvs": _copy(getattr(solver, "elem_sdvs", None)),
        "nlgeom": bool(solver.nlgeom),
        "nlgeom_engaged": bool(getattr(solver, "_nlgeom_engaged", False)),
        "temperature": _copy(getattr(solver, "current_temperature", None))
        if not np.isscalar(getattr(solver, "current_temperature", None)) else getattr(solver, "current_temperature"),
        "dt_ctrl": dict(vars(dt_ctrl)) if dt_ctrl is not None else None,
        "constraint_states": constraint_states,
        "constraints_without_state": stateless,
    }


def apply(solver, state: Dict[str, Any]) -> List[str]:
    """Put ``state`` onto ``solver``; returns warnings (constraints whose history is not restored)."""
    if state.get("version") != VERSION:
        raise ValueError(f"restart file version {state.get('version')!r}, this solver reads {VERSION}")
    mine, theirs = fingerprint(solver), state["fingerprint"]
    for key in mine:
        if mine[key] != theirs[key]:
            raise ValueError(f"restart file belongs to a different model: {key} differs "
                             f"(file {theirs[key]!r}, this model {mine[key]!r})")
    solver.u = _copy(state["u"])
    if state["v"] is not None:
        solver.v = _copy(state["v"])
    if state["a"] is not None:
        solver.a = _copy(state["a"])
    if state["elem_sdvs"] is not None:
        solver.elem_sdvs = _copy(state["elem_sdvs"])
    if state["u_step_start"] is not None:
        solver._u_step_start = _copy(state["u_step_start"])
    solver._nlgeom_engaged = bool(state["nlgeom_engaged"])
    if bool(state["nlgeom"]) != bool(solver.nlgeom):
        solver.nlgeom = bool(state["nlgeom"])
        solver._build_kernel_groups()
    if state["temperature"] is not None:
        solver.current_temperature = _copy(state["temperature"]) if isinstance(state["temperature"], np.ndarray) \
            else state["temperature"]
    warnings = [f"constraint {n!r} has no capture_state: its history (friction slip, contact set) restarts fresh"
                for n in state["constraints_without_state"]]
    for cs in state["constraint_states"]:
        c = solver.constraints[cs["index"]] if cs["index"] < len(solver.constraints) else None
        if c is None or (getattr(c, "name", None) or type(c).__name__) != cs["name"] or not hasattr(c, "restore_state"):
            warnings.append(f"constraint {cs['name']!r} in the restart file has no counterpart in this model")
            continue
        c.restore_state(cs["state"])
    return warnings


def save(state: Dict[str, Any], path: str) -> None:
    """Write atomically: a crash mid-write must not leave a truncated file under the real name."""
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=4)
    os.replace(tmp, path)


def load(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def write_path(request: Dict[str, Any], step_name: str, inc: int) -> str:
    """Where a record goes: one overlaid file per job, or one file per (step, increment)."""
    stem = str(request["path"])
    if request.get("overlay"):
        return f"{stem}.rst"
    return f"{stem}_{step_name}_{inc:04d}.rst"


def maybe_write(solver, step, request: Dict[str, Any], step_time: float, step_inc: int, inc_no: int,
                total_iters: int, dt_ctrl: Any, t_total: float) -> Optional[str]:
    """Write a record after every ``frequency``-th converged increment and at the end of the step.

    ``inc_no`` is the number of the NEXT increment, so ``inc_no - 1`` increments have converged. Returns
    the path written, or None."""
    n_conv = int(inc_no) - 1
    freq = max(1, int(request.get("frequency", 1)))
    if n_conv % freq != 0 and step_time < t_total - 1e-12:
        return None
    state = capture(solver, step.name, step_time, step_inc, inc_no, total_iters, dt_ctrl)
    path = write_path(request, step.name, n_conv)
    save(state, path)
    return path


def find_record(oldjob: str, step_name: str, inc: Optional[int]) -> str:
    """Path of the record to read. ``inc`` None means the last one written for the step.

    Looks for ``<oldjob>_<step>_<inc:04d>.rst`` first, then the overlaid ``<oldjob>.rst`` (whose own step
    name and increment must match)."""
    import glob
    import re
    if inc is not None:
        exact = write_path({"path": oldjob, "overlay": False}, step_name, inc)
        if os.path.exists(exact):
            return exact
        overlay = f"{oldjob}.rst"
        if os.path.exists(overlay):
            st = load(overlay)["step"]
            if st["name"] == step_name and st["inc_no"] - 1 == inc:
                return overlay
        raise FileNotFoundError(f"no restart record for step {step_name!r} increment {inc}: expected {exact}")
    found = []
    for path in glob.glob(f"{glob.escape(oldjob)}_{glob.escape(step_name)}_*.rst"):
        m = re.search(r"_(\d+)\.rst$", path)
        if m:
            found.append((int(m.group(1)), path))
    overlay = f"{oldjob}.rst"
    if os.path.exists(overlay) and load(overlay)["step"]["name"] == step_name:
        found.append((load(overlay)["step"]["inc_no"] - 1, overlay))
    if not found:
        raise FileNotFoundError(f"no restart record for step {step_name!r}: expected "
                                f"{oldjob}_{step_name}_<inc>.rst or {overlay}")
    return max(found)[1]
