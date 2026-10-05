"""StepPlan -- the step's boundary conditions and loads, lowered once.

``lower_step`` is called at the start of each solver's ``solve()``; it
resolves every region / amplitude / surface of the step's active BCs and
loads to solver indices (raising, with the deck location, on anything that
does not resolve) and precomputes everything geometry-independent. Per
increment the solver then asks the plan for

  * ``prescribed(t, u_step_start)`` -> (dof indices, values) of the Dirichlet
    DOFs, and
  * ``f_ext(t, u)`` -> the external force vector.

One implementation serves the 2D and the 3D solver (``dim``), so twin decks
give the same vectors by construction.

Semantics (unchanged from the former per-solver code unless noted):
  * BC: ``UserFunctionBC`` per node; displacement BC = target * amplitude(t);
    without an amplitude a STATIC/QUASISTATIC step ramps from the step-start
    displacement, ``u0 + (target - u0) * clamp(t / time_period)``; other
    procedures apply the target at once.
  * CLOAD: nodal, times the Abaqus STATIC default ramp (no amplitude).
  * Pressure (element face P1.. on an elset, or an element-based surface):
    integrated through ``dispsolver.loads.face_pressure_forces``. With
    NLGEOM on it is a follower load integrated on X + u (u = the
    displacement passed in, the increment-start displacement, so a lagged
    follower with no load stiffness); with NLGEOM off, on X.
  * 3D gravity / body force: consistent weights (``body_force_weights``);
    GRAV is scaled by the element's own density and raises if it has none.
    2D body loads are still refused (the plane-thickness convention of the
    weights is unit thickness and is not settled).
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from dispsolver.loads import body_force_weights, face_pressure_forces, load_type
from dispsolver.model.regions import (
    describe as _describe,
    node_ids_of,
    resolve_elem_region,
    resolve_node_region,
)

_STATIC_PROCS = ("STATIC", "QUASISTATIC")


def _is_static(step: Any) -> bool:
    return str(getattr(step, "procedure", "STATIC")).upper() in _STATIC_PROCS


def _clamp_ramp(t: float, period: float) -> float:
    return min(max(t / max(period, 1e-12), 0.0), 1.0)


def _load_ramp(step: Any, load: Any, t: float, period: float) -> float:
    """Abaqus default amplitude: a load without AMPLITUDE= ramps linearly over
    a STATIC step and is applied at once otherwise."""
    if getattr(load, "amplitude", None) is None and _is_static(step):
        return _clamp_ramp(t, period)
    return 1.0


# ----------------------------------------------------------------- regions
def _node_gids(sys: Any, mesh: Any, region: Any, step: Any, what: str) -> List[int]:
    """Global node ids of a BC/load region; raises if it does not resolve."""
    if sys is not None and hasattr(sys, "global_nsets"):
        return node_ids_of(sys, resolve_node_region(sys, region, step, what=what))
    # model-less use (solver built directly from a mesh): mesh sets only
    if isinstance(region, str) and region.upper() == "ALL":
        return list(mesh.nodes.keys())
    for attr in ("node_sets", "sets"):
        for k, v in (getattr(mesh, attr, {}) or {}).items():
            if str(k).upper() == str(region).upper():
                return list(v.node_ids)
    if isinstance(region, (int, np.integer)) or (isinstance(region, str) and region.isdigit()):
        if int(region) in mesh.nodes:
            return [int(region)]
    raise ValueError(f"{what}: node region {region!r} does not resolve")


def _elements_of(sys: Any, mesh: Any, region: Any, what: str, allow_blank_all: bool = False) -> list:
    """Element objects of an element region (an elset, or a single element)."""
    elems = list(mesh.elements.values())
    if sys is not None and hasattr(sys, "global_elsets"):
        rows = resolve_elem_region(sys, region, what=what, allow_blank_all=allow_blank_all)
        return [elems[int(r)] for r in rows]
    if region in (None, "") or (isinstance(region, str) and region.upper() == "ALL"):
        if region in (None, "") and not allow_blank_all:
            raise ValueError(f"{what}: empty element region")
        return elems
    for k, v in (getattr(mesh, "element_sets", {}) or {}).items():
        if str(k).upper() == str(region).upper():
            return [mesh.elements[e] for e in v if e in mesh.elements]
    if isinstance(region, (int, np.integer)) or (isinstance(region, str) and region.isdigit()):
        if int(region) in mesh.elements:
            return [mesh.elements[int(region)]]
    raise ValueError(f"{what}: element region {region!r} needs the model system or a mesh element set")


def _surface_faces(region: Any, sys: Any, model: Any):
    """Element-based surface -> [(global element id, 1-based face id)], or None
    when `region` is not a surface. Accepts an assembly surface name or an
    instance-qualified part surface ("INST.SURF")."""
    if model is None or sys is None or not isinstance(region, str):
        return None
    asm = getattr(model, "root_assembly", None)
    l2g = getattr(sys, "elem_local_to_global", {})
    out = []
    if asm is not None and region in getattr(asm, "surfaces", {}):
        for inst, local_eid, fid in getattr(asm.surfaces[region], "inst_faces", None) or []:
            gid = l2g.get((inst, local_eid))
            if gid is not None:
                out.append((gid, fid))
        return out or None
    if "." in region and asm is not None:
        inst_name, surf_name = region.split(".", 1)
        inst = getattr(asm, "instances", {}).get(inst_name)
        part = getattr(inst, "part", inst)
        surf = getattr(part, "surfaces", {}).get(surf_name) if part is not None else None
        if surf is None:
            return None
        for ef in getattr(surf, "faces", None) or []:
            eid = ef.element_id if hasattr(ef, "element_id") else ef[0]
            fid = ef.face_id if hasattr(ef, "face_id") else ef[1]
            gid = l2g.get((inst_name, eid))
            if gid is not None:
                out.append((gid, fid))
        return out or None
    return None


# ------------------------------------------------------------------ terms
class _UserBC:
    """UserFunctionBC: evaluated per node every call."""

    def __init__(self, bc: Any, label: str, nodes: np.ndarray, coords: np.ndarray, dim: int):
        self.bc, self.label, self.nodes, self.coords, self.dim = bc, label, nodes, coords, dim

    def prescribed(self, t: float, u0: Optional[np.ndarray], total_time: float):
        dofs: List[int] = []
        vals: List[float] = []
        for n_idx, xy in zip(self.nodes, self.coords):
            res = self.bc.evaluate_node(xy, t, step_time=t)
            if res is None:
                continue
            for d in range(self.dim):
                v = res[d] if len(res) > d else None
                if v is not None:
                    dofs.append(self.dim * int(n_idx) + d)
                    vals.append(float(v))
        return np.array(dofs, dtype=np.int64), np.array(vals, dtype=np.float64)


class _DispBC:
    """Displacement BC: fixed (dof, target) pairs plus amplitude / ramp."""

    def __init__(self, bc: Any, label: str, dofs: np.ndarray, targets: np.ndarray,
                 amp: Any, step: Any):
        self.bc, self.label, self.dofs, self.targets, self.amp = bc, label, dofs, targets, amp
        self.static = _is_static(step)

    def prescribed(self, t: float, u0: Optional[np.ndarray], total_time: float):
        if self.amp is not None:
            return self.dofs, self.targets * self.amp.evaluate(t, step_time=t, total_time=total_time)
        if self.static:
            s = _clamp_ramp(t, total_time)
            base = u0[self.dofs] if u0 is not None else np.zeros(len(self.dofs))
            return self.dofs, base + (self.targets - base) * s
        return self.dofs, self.targets * 1.0


class _NodalLoad:
    def __init__(self, load: Any, label: str, nodes: np.ndarray, dim: int):
        self.load, self.label, self.nodes, self.dim = load, label, nodes, dim


class _PressureLoad:
    """Face pressure: faces = [(node index array of the element, elem_type,
    face)]. `dofs`/`unit` hold the p = 1 reference forces when the load is
    not a follower load."""

    def __init__(self, load: Any, label: str, faces: list):
        self.load, self.label, self.faces = load, label, faces
        self.dofs: Optional[np.ndarray] = None
        self.unit: Optional[np.ndarray] = None


class _BodyLoad:
    def __init__(self, load: Any, label: str, is_grav: bool, node_idx: np.ndarray,
                 weights: np.ndarray, scale: np.ndarray):
        self.load, self.label, self.is_grav = load, label, is_grav
        self.node_idx, self.weights, self.scale = node_idx, weights, scale


# ------------------------------------------------------------------- plan
class StepPlan:
    def __init__(self, step: Any, dim: int, num_dofs: int, nlgeom: bool, total_time: float,
                 coords: np.ndarray):
        self.step = step
        self.dim = dim
        self.num_dofs = num_dofs
        self.nlgeom = bool(nlgeom)
        self.total_time = float(total_time)
        self.model = getattr(step, "parent_model", None)
        self._coords = coords
        self.bcs: list = []
        self.nodal: List[_NodalLoad] = []
        self.pressures: List[_PressureLoad] = []
        self.bodies: List[_BodyLoad] = []

    # -------------------------------------------------------------- BCs
    def prescribed(self, t: float, u_step_start: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """(DOF indices, prescribed values) at time t. Later BCs come later in
        the arrays, so a solver applying them in order lets them override."""
        dofs = [np.zeros(0, dtype=np.int64)]
        vals = [np.zeros(0, dtype=np.float64)]
        for term in self.bcs:
            d, v = term.prescribed(t, u_step_start, self.total_time)
            dofs.append(np.asarray(d, dtype=np.int64))
            vals.append(np.asarray(v, dtype=np.float64))
        return np.concatenate(dofs), np.concatenate(vals)

    # ------------------------------------------------------------ loads
    def f_ext(self, t: float, u: Optional[np.ndarray]) -> np.ndarray:
        """External force vector at time t. `u` (the increment-start
        displacement) is used only by follower pressure terms."""
        dim = self.dim
        f = np.zeros(self.num_dofs, dtype=np.float64)
        period = float(getattr(self.step, "time_period", 1.0))
        model = self.model

        for term in self.nodal:
            load = term.load
            f_vec = np.asarray(load.get_force_vector(t=t, step_time=t, model=model), dtype=np.float64)
            f_vec = f_vec * _load_ramp(self.step, load, t, period)
            for d in range(dim):
                np.add.at(f, dim * term.nodes + d, f_vec[d])

        for term in self.bodies:
            load = term.load
            if term.is_grav:
                vec = np.asarray(load.get_acceleration_vector(t=t, step_time=t, model=model), dtype=np.float64)
            else:
                vec = np.asarray(load.get_force_vector(t=t, step_time=t, model=model), dtype=np.float64)
            vec = vec * _load_ramp(self.step, load, t, period)
            contrib = (term.weights * term.scale)[:, None] * vec[None, :dim]
            F = f.reshape(-1, dim)
            np.add.at(F, term.node_idx, contrib)

        for term in self.pressures:
            load = term.load
            if hasattr(load, "get_magnitude"):
                p = float(load.get_magnitude(t=t, step_time=t, model=model))
            else:
                p = float(getattr(load, "magnitude", 0.0))
            p *= _load_ramp(self.step, load, t, period)
            if p == 0.0:
                continue
            if term.unit is not None:
                np.add.at(f, term.dofs, p * term.unit)
                continue
            # follower: integrate on X + u
            U = None if u is None else np.asarray(u, dtype=np.float64).reshape(-1, dim)
            for conn, etype, face in term.faces:
                Xe = self._coords[conn] + (0.0 if U is None else U[conn])
                loc, forces = self._face(Xe, etype, face, 1.0, term)
                nodes = conn[loc]
                for d in range(dim):
                    np.add.at(f, dim * nodes + d, p * forces[:, d])
        return f

    def _face(self, Xe, etype, face, p, term):
        try:
            return face_pressure_forces(Xe, etype, face, p)
        except (KeyError, IndexError) as exc:
            raise ValueError(f"{term.label}: {etype} has no face {face}") from exc


# ------------------------------------------------------------------ lower
def _face_number(load: Any, label: str) -> int:
    face_name = getattr(load, "face", getattr(load, "face_name", None))
    if not face_name:
        raise ValueError(f"{label}: element face is not specified (expected P1..P6 / S1..S6)")
    spec = load_type("DLOAD", str(face_name).strip())
    if spec.kind != "face_pressure" or spec.face is None:
        raise ValueError(f"{label}: element face {face_name!r} is not a face number (expected P1..P6 / S1..S6)")
    return int(spec.face)


def lower_step(step: Any, sys: Any, mesh: Any, dim: int, nlgeom: bool, *,
               total_time: Optional[float] = None,
               explicit_density: Optional[Callable[[int], float]] = None) -> StepPlan:
    """Resolve `step`'s active BCs and loads against `sys` / `mesh` once.

    total_time defaults to step.time_period. explicit_density(row) is the
    density the deck gives element `row` (0.0 when its material has none);
    needed only for *DLOAD GRAV.
    """
    if total_time is None:
        total_time = getattr(step, "time_period", 1.0)
    nid_map = mesh.node_id_to_index()
    coords = np.asarray(mesh.nodes_array(), dtype=np.float64)
    plan = StepPlan(step, dim, dim * len(nid_map), nlgeom, total_time, coords)
    model = plan.model
    names = ("u1", "u2", "u3")[:dim]

    # ---- BCs
    if hasattr(step, "get_active_bcs"):
        for bc_name, bc in step.get_active_bcs().items():
            label = _describe(bc, f"BC {bc_name}")
            gids = _node_gids(sys, mesh, bc.region, step, label)
            nodes = np.array([nid_map[g] for g in gids], dtype=np.int64)
            if hasattr(bc, "evaluate_node"):
                plan.bcs.append(_UserBC(bc, label, nodes, coords[nodes], dim))
            elif any(hasattr(bc, n) for n in names):
                amp = None
                if bc.amplitude is not None:
                    if isinstance(bc.amplitude, str):
                        amps = getattr(model, "amplitudes", None)
                        amp = amps.get(bc.amplitude, None) if amps is not None else None
                    elif hasattr(bc.amplitude, "evaluate"):
                        amp = bc.amplitude
                    if amp is None:
                        raise ValueError(f"{label}: amplitude {bc.amplitude!r} is not defined")
                dofs: List[int] = []
                targets: List[float] = []
                for n_idx in nodes:
                    for d, name in enumerate(names):
                        val = getattr(bc, name, None)
                        if val is not None:
                            dofs.append(dim * int(n_idx) + d)
                            targets.append(float(val))
                plan.bcs.append(_DispBC(bc, label, np.array(dofs, dtype=np.int64),
                                        np.array(targets, dtype=np.float64), amp, step))

    # ---- loads
    if hasattr(step, "get_active_loads"):
        for load_name, load in step.get_active_loads().items():
            is_grav = hasattr(load, "get_acceleration_vector")
            if is_grav or type(load).__name__ == "BodyForce" or hasattr(load, "b1"):
                if dim == 2:
                    raise NotImplementedError(f"load {load_name}: body/gravity loads are not implemented "
                                              f"in the 2D solver")
                label = _describe(load, f"load {load_name}")
                elems = _elements_of(sys, mesh, load.region, label, allow_blank_all=is_grav)
                row_of = {e.id if hasattr(e, "id") else e.eid: r for r, e in enumerate(mesh.elements.values())}
                node_idx: List[int] = []
                weights: List[float] = []
                scale: List[float] = []
                for e in elems:
                    conn = np.array([nid_map[n] for n in e.node_ids], dtype=np.int64)
                    w = body_force_weights(coords[conn], e.elem_type)
                    s = 1.0
                    if is_grav:
                        row = row_of[e.id if hasattr(e, "id") else e.eid]
                        s = float(explicit_density(row)) if explicit_density is not None else 0.0
                        if s <= 0.0:
                            raise ValueError(f"{label}: *DLOAD GRAV on element row {row} whose "
                                             f"material has no *DENSITY")
                    node_idx.extend(conn.tolist())
                    weights.extend(np.asarray(w, dtype=np.float64).tolist())
                    scale.extend([s] * len(conn))
                plan.bodies.append(_BodyLoad(load, label, is_grav, np.array(node_idx, dtype=np.int64),
                                             np.array(weights), np.array(scale)))

            elif hasattr(load, "get_force_vector"):
                label = _describe(load, f"load {load_name}")
                gids = _node_gids(sys, mesh, load.region, step, label)
                plan.nodal.append(_NodalLoad(load, label, np.array([nid_map[g] for g in gids], dtype=np.int64), dim))

            elif (hasattr(load, "get_magnitude") or hasattr(load, "magnitude")
                  or type(load).__name__ == "Pressure") and hasattr(load, "region"):
                label = _describe(load, f"pressure load {load_name!r}")
                faces: list = []
                surf = _surface_faces(load.region, sys, model)
                if surf is not None:
                    for gid, fid in surf:
                        e = mesh.elements[gid]
                        faces.append((np.array([nid_map[n] for n in e.node_ids], dtype=np.int64),
                                      e.elem_type, int(fid)))
                else:
                    face = _face_number(load, label)
                    for e in _elements_of(sys, mesh, load.region, label):
                        faces.append((np.array([nid_map[n] for n in e.node_ids], dtype=np.int64),
                                      e.elem_type, face))
                if not faces:
                    raise ValueError(f"{label}: region {load.region!r} resolved to no element faces")
                term = _PressureLoad(load, label, faces)
                if not plan.nlgeom:
                    d_all: List[np.ndarray] = []
                    f_all: List[np.ndarray] = []
                    for conn, etype, face in faces:
                        loc, forces = plan._face(coords[conn], etype, face, 1.0, term)
                        nodes = conn[loc]
                        d_all.append((dim * nodes[:, None] + np.arange(dim)[None, :]).ravel())
                        f_all.append(np.asarray(forces, dtype=np.float64).ravel())
                    term.dofs = np.concatenate(d_all)
                    term.unit = np.concatenate(f_all)
                plan.pressures.append(term)
    return plan
